"""Compare local tracks without changing the global autotagger configuration."""

import hashlib
import itertools
import json
import logging
import math
import os
import sqlite3

from beets import config, ui
from beets.autotag import match
from beets.autotag.hooks import TrackInfo
from beets.dbcore.query import InQuery, OrQuery
from beets.library import Item
from beets.plugins import BeetsPlugin
from unidecode import unidecode


DEFAULT_WEIGHTS = {
    "track_title": 4.0,
    "track_artist": 3.0,
    "track_length": 2.0,
    "track_id": 4.0,
    "track_index": 0.5,
    "medium": 0.5,
    "album": 2.0,
    "release_group": 6.0,
    "acoustic": 8.0,
    "data_source": 0.0,
}

REFRESH_FIELDS = (
    "title",
    "artist",
    "artists",
    "length",
    "track",
    "disc",
    "mb_trackid",
    "album",
    "mb_releasegroupid",
)
ACOUSTIC_REFRESH_FIELDS = ("acoustid_id", "acoustid_fingerprint")


def pair_key(first, second):
    """Canonical key for a non-self, persisted pair."""
    if not first or not second or first == second:
        raise ValueError("similarity pairs require two distinct persisted item IDs")
    return tuple(sorted((int(first), int(second))))


def effective_score(row, dissociate_penalty="to-zero"):
    """Manual choices survive recomputation and any valid threshold change."""
    if row["manual"]:
        if row["manual"] > 0:
            return 1.0
        penalty = row["score"] if dissociate_penalty == "to-zero" else dissociate_penalty
        return max(0.0, row["score"] - penalty)
    return row["score"]


class SimilarityStore:
    """Sparse, undirected pairs in the same SQLite database as beets."""

    def __init__(self, lib, refresh_state=False):
        self.lib = lib
        with lib.transaction() as tx:
            tx.mutate(
                "CREATE TABLE IF NOT EXISTS similarity ("
                "item_a INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE, "
                "item_b INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE, "
                "score REAL NOT NULL CHECK(score BETWEEN 0 AND 1), "
                "manual INTEGER NOT NULL DEFAULT 0 CHECK(manual IN (-1, 0, 1)), "
                "PRIMARY KEY(item_a, item_b), CHECK(item_a < item_b)) WITHOUT ROWID"
            )
            tx.mutate(
                "CREATE INDEX IF NOT EXISTS similarity_item_b ON similarity(item_b)"
            )
            if refresh_state:
                tx.mutate(
                    "CREATE TABLE IF NOT EXISTS similarity_refresh_state ("
                    "item_id INTEGER PRIMARY KEY "
                    "REFERENCES items(id) ON DELETE CASCADE, "
                    "signature TEXT NOT NULL)"
                )

    def get(self, first, second):
        key = pair_key(first, second)
        with self.lib.transaction() as tx:
            rows = tx.query(
                "SELECT * FROM similarity WHERE item_a=? AND item_b=?", key
            )
        return rows[0] if rows else None

    def put(self, first, second, score, manual=None):
        key = pair_key(first, second)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("similarity must be a finite number between 0 and 1")
        with self.lib.transaction() as tx:
            tx.mutate(
                "INSERT INTO similarity(item_a, item_b, score, manual) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(item_a, item_b) "
                "DO UPDATE SET score=excluded.score, "
                "manual=COALESCE(?, similarity.manual)",
                (*key, score, manual or 0, manual),
            )

    def neighbors(self, item_id):
        with self.lib.transaction() as tx:
            return tx.query(
                "SELECT * FROM similarity WHERE item_a=? OR item_b=?",
                (item_id, item_id),
            )

    def refresh_signature(self, item_id):
        with self.lib.transaction() as tx:
            rows = tx.query(
                "SELECT signature FROM similarity_refresh_state WHERE item_id=?",
                (item_id,),
            )
        return rows[0]["signature"] if rows else None

    def set_refresh_signature(self, item_id, signature):
        with self.lib.transaction() as tx:
            tx.mutate(
                "INSERT INTO similarity_refresh_state(item_id, signature) "
                "VALUES (?, ?) ON CONFLICT(item_id) "
                "DO UPDATE SET signature=excluded.signature",
                (item_id, signature),
            )

    def storage_size(self):
        """Allocated SQLite page bytes for the similarity table and its indexes."""
        with self.lib.transaction() as tx:
            try:
                rows = tx.query(
                    "SELECT COALESCE(SUM(pgsize), 0) AS size FROM dbstat "
                    "WHERE name IN ('similarity', 'similarity_item_b', "
                    "'similarity_refresh_state')"
                )
            except sqlite3.OperationalError as exc:
                if "no such table: dbstat" not in str(exc):
                    raise
                raise ui.UserError(
                    "Cannot measure similarity storage: this SQLite build "
                    "does not support dbstat (SQLITE_ENABLE_DBSTAT_VTAB)"
                ) from exc
        return rows[0]["size"]

    def remove(self, ids=None):
        with self.lib.transaction() as tx:
            if ids is None:
                tx.mutate("DELETE FROM similarity")
            else:
                has_refresh_state = tx.query(
                    "SELECT 1 FROM sqlite_schema "
                    "WHERE type='table' AND name='similarity_refresh_state'"
                )
                for item_id in ids:
                    tx.mutate(
                        "DELETE FROM similarity WHERE item_a=? OR item_b=?",
                        (item_id, item_id),
                    )
                    if has_refresh_state:
                        tx.mutate(
                            "DELETE FROM similarity_refresh_state WHERE item_id=?",
                            (item_id,),
                        )


class SimilarityPlugin(BeetsPlugin):
    def __init__(self):
        super().__init__()
        self.config.add(
            {
                "auto": False,
                "refresh_state": True,
                "remove_on_delete": False,
                "threshold": 0.8,
                "acoustic": True,
                "dissociate_penalty": "to-zero",
                "weights": DEFAULT_WEIGHTS,
                "warn_items": 1000,
                "warn_pairs": 500000,
                "fuzzy_artists": False,
                "bare_ascii_artists": False,
                "artist_threshold": 0.2,
            }
        )
        self._pending = {}
        self.register_listener("item_removed", self._item_removed)
        if self.config["auto"].get(bool):
            self.import_stages.append(self._import_stage)
            self.register_listener("database_change", self._database_change)
            self.register_listener("cli_exit", self._flush)

    def _number(self, name, minimum=0.0, maximum=1.0):
        value = self.config[name].as_number()
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ui.UserError(f"similarity.{name} must be in [{minimum}, {maximum}]")
        return value

    def _threshold(self, value=None):
        value = self.config["threshold"].as_number() if value is None else value
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ui.UserError("similarity threshold must be greater than 0 and at most 1")
        return value

    def _dissociate_penalty(self):
        value = self.config["dissociate_penalty"].get()
        if value == "to-zero":
            return value
        if isinstance(value, (str, bool)):
            raise ui.UserError(
                'similarity.dissociate_penalty must be a number in [0, 1] or "to-zero"'
            )
        try:
            penalty = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ui.UserError(
                'similarity.dissociate_penalty must be a number in [0, 1] or "to-zero"'
            ) from exc
        if not math.isfinite(penalty) or not 0 <= penalty <= 1:
            raise ui.UserError(
                'similarity.dissociate_penalty must be a number in [0, 1] or "to-zero"'
            )
        return penalty

    @staticmethod
    def _virtual(item):
        # Detached copies prevent scoring from changing or storing library items.
        return Item(
            title=item.title,
            artist=item.artist,
            length=item.length,
            track=item.track,
            disc=item.disc,
            mb_trackid=item.mb_trackid,
        )

    @staticmethod
    def _info(item):
        return TrackInfo(
            title=item.title,
            artist=item.artist,
            length=item.length,
            index=item.track,
            medium_index=item.track,
            medium=item.disc,
            track_id=item.mb_trackid or None,
        )

    def _acoustic_distance(self, first, second):
        fingerprints = [i.get("acoustid_fingerprint") for i in (first, second)]
        if all(fingerprints):
            try:
                import acoustid
                import chromaprint
            except ImportError as exc:
                raise ui.UserError(
                    "Cannot compare saved fingerprints: install "
                    "beets-similarity[acoustic] and libchromaprint, or set "
                    "similarity.acoustic: no"
                ) from exc
            try:
                left = (first.length, self._fingerprint_bytes(fingerprints[0]))
                right = (second.length, self._fingerprint_bytes(fingerprints[1]))
                similarities = [
                    acoustid.compare_fingerprints(left, right),
                    acoustid.compare_fingerprints(right, left),
                ]
            except (ValueError, RuntimeError, ZeroDivisionError, chromaprint.FingerprintError) as exc:
                raise ui.UserError(
                    "Cannot decode saved acoustic fingerprints; regenerate them with chroma"
                ) from exc
            if any(not math.isfinite(s) or not 0 <= s <= 1 for s in similarities):
                raise ui.UserError("Fingerprint comparison returned an invalid score")
            return 1.0 - sum(similarities) / 2
        ids = [i.get("acoustid_id") for i in (first, second)]
        if all(ids):
            return float(ids[0] != ids[1])
        return None

    @staticmethod
    def _fingerprint_bytes(value):
        return value.encode("ascii") if isinstance(value, str) else value

    def _log_distance(self, source, target, dist, weights):
        if config["verbose"].get(int) < 3 or not self._log.isEnabledFor(logging.DEBUG):
            return
        self._log.debug(
            "Similarity distance {} -> {}: raw penalty {} / total weight {}"
            " = {}; nonzero penalties below (final score impact includes / 2)",
            source.id, target.id, dist.raw_distance, dist.max_distance, float(dist),
        )
        fields = {
            "track_title": "title",
            "track_artist": "artist",
            "track_length": "length",
            "track_id": "mb_trackid",
            "track_index": "track",
            "medium": "disc",
            "album": "album",
            "release_group": "mb_releasegroupid",
            "data_source": "data_source",
        }
        for key, penalty in dist.items():
            if key == "acoustic":
                if all(i.get("acoustid_fingerprint") for i in (source, target)):
                    values = "saved fingerprints (both alignments averaged)"
                else:
                    values = (
                        f"acoustid_id: {source.get('acoustid_id')!r}"
                        f" vs {target.get('acoustid_id')!r}"
                    )
            else:
                field = fields[key]
                values = f"{field}: {source.get(field)!r} vs {target.get(field)!r}"
            self._log.debug(
                "Similarity penalty {} -> {}: {}; {}; weight={}; "
                "normalized distance={}; subtracts {} from final similarity",
                source.id, target.id, key, values, weights[key], penalty, penalty / 2,
            )

    def score(self, first, second, *, progress=None):
        """Return symmetric similarity in [0, 1]; never inspect audio quality."""
        weights = {
            key: self.config["weights"][key].as_number() for key in DEFAULT_WEIGHTS
        }
        if any(not math.isfinite(w) or w < 0 for w in weights.values()):
            raise ui.UserError("similarity weights must be finite and nonnegative")
        acoustic = (
            self._acoustic_distance(first, second)
            if self.config["acoustic"].get(bool)
            else None
        )
        distances = []
        for source, target in ((first, second), (second, first)):
            dist = match.track_distance(
                self._virtual(source), self._info(target), incl_artist=True
            )
            # Distance resolves weights on the instance before its class defaults.
            dist._weights = weights
            if source.album or target.album:
                dist.add_string("album", source.album, target.album)
            if source.mb_releasegroupid or target.mb_releasegroupid:
                dist.add_expr(
                    "release_group",
                    source.mb_releasegroupid != target.mb_releasegroupid,
                )
            if acoustic is not None:
                dist.add("acoustic", acoustic)
            if not dist.max_distance:
                raise ui.UserError("Enable at least one applicable similarity weight")
            distances.append(float(dist))
            self._log_distance(source, target, dist, weights)
        score = max(0.0, min(1.0, 1.0 - sum(distances) / 2))
        label = "Similarity score"
        if progress is not None:
            label += f" ({progress[0]}/{progress[1]})"
        if config["verbose"].get(int) >= 3:
            self._log.debug(
                label + " for items {} and {}: {} = 1 - ({} + {}) / 2",
                first.id, second.id, score, *distances,
            )
        else:
            self._log.debug(
                label + " for items {} and {}: {}", first.id, second.id, score
            )
        return score

    def similar_items(self, lib, item, threshold=None):
        """Public API for rating/playlist plugins: (Item, score), best first."""
        threshold = self._threshold(threshold)
        penalty = self._dissociate_penalty()
        results = []
        for row in SimilarityStore(lib).neighbors(item.id):
            score = effective_score(row, penalty)
            if score >= threshold:
                other_id = (
                    row["item_b"] if row["item_a"] == item.id else row["item_a"]
                )
                other = lib.get_item(other_id)
                if other is not None:
                    results.append((other, score))
        return sorted(results, key=lambda result: (-result[1], result[0].id))

    def _confirm(self, message):
        return ui.input_yn(message, require=True)

    def _warn_work(self, count, comparisons, unscoped=False, force=False):
        if (
            unscoped
            or count >= self.config["warn_items"].get(int)
            or comparisons >= self.config["warn_pairs"].get(int)
        ):
            self._log.warning(
                "Similarity: {} tracks, up to {} comparisons (quadratic work). "
                "Use an artist/genre query to restrict the scope.",
                count,
                comparisons,
            )
            return force or self._confirm("Proceed with this similarity computation?")
        return True

    def commands(self):
        command = ui.Subcommand("similar", help="compare tracks in the local library")
        parser = command.parser
        for action in ("add", "list", "remove"):
            parser.add_option(f"--{action}", action="store_true", default=False)
        parser.add_option(
            "--size", action="store_true", default=False,
            help="display similarity table and index storage in bytes",
        )
        for action in ("associate", "dissociate"):
            parser.add_option(f"--{action}", nargs=2, type="int", metavar="ID ID")
        parser.add_option("-f", "--force", action="store_true", default=False)
        parser.add_option("--matrix", action="store_true", default=False)
        parser.add_option("--debug", action="store_true", default=False)
        parser.add_option("--thres", type="float", help="minimum similarity (0, 1]")
        command.func = self._command
        return [command]

    def _command(self, lib, opts, args):
        actions = [
            name for name in ("add", "list", "remove", "associate", "dissociate", "size")
            if getattr(opts, name)
        ]
        if len(actions) > 1:
            raise ui.UserError("Choose only one similar action")
        action = actions[0] if actions else "list"
        if (opts.matrix or opts.debug) and action != "list":
            raise ui.UserError("--matrix and --debug require --list")
        if action == "size":
            if args or opts.thres is not None:
                raise ui.UserError("--size does not accept a query or --thres")
            ui.print_(f"Similarity storage: {SimilarityStore(lib).storage_size()} bytes")
            return
        threshold = self._threshold(opts.thres)
        store = SimilarityStore(lib)
        if action in ("associate", "dissociate"):
            if args:
                raise ui.UserError("Manual associations accept two item IDs, not a query")
            ids = getattr(opts, action)
            if ids[0] == ids[1] or any(lib.get_item(i) is None for i in ids):
                raise ui.UserError("Provide two distinct existing item IDs")
            old = store.get(*ids)
            if old and not opts.force and not self._confirm("Override existing pair?"):
                return
            score = old["score"] if old else self.score(*(lib.get_item(i) for i in ids))
            store.put(*ids, score, manual=1 if action == "associate" else -1)
            return
        items = sorted(lib.items(args), key=lambda i: i.id)
        if action == "add":
            count = len(items)
            if not self._warn_work(count, count * (count - 1) // 2, not args, opts.force):
                return
            selected = {i.id for i in items}
            existing = any(
                row["item_a"] in selected and row["item_b"] in selected
                for item in items for row in store.neighbors(item.id)
            )
            if existing and not opts.force and not self._confirm("Refresh existing scores?"):
                return
            total = count * (count - 1) // 2
            self._log.info(
                "Similarity refresh starts: {} pairs to score across {} tracks.",
                total, count,
            )
            for progress, (first, second) in enumerate(
                itertools.combinations(items, 2), 1
            ):
                store.put(first.id, second.id, self.score(first, second))
                self._log.info(
                    "Similarity progress: {}/{} pairs scored", progress, total
                )
            self._log.info(
                "Similarity complete: scored {} pairs across {} tracks", total, count
            )
        elif action == "remove":
            if not args:
                # Reset is destructive even when --force was supplied.
                if self._confirm("Reset the entire similarity matrix, including manual choices?"):
                    store.remove()
            elif opts.force or self._confirm("Remove scores touching the selected tracks?"):
                store.remove(i.id for i in items)
        else:
            self._list(store, items, threshold, opts.matrix, opts.debug)

    @staticmethod
    def _label(item, debug):
        if not debug:
            return str(item.id)
        path = os.fsdecode(item.path).replace("\\", "/")
        # Escape control characters so every output row remains one physical line.
        return "/".join(path.split("/")[-3:]).replace(
            "\t", "\\t"
        ).replace("\r", "\\r").replace("\n", "\\n")

    def _list(self, store, items, threshold, matrix, debug):
        labels = {item.id: self._label(item, debug) for item in items}
        penalty = self._dissociate_penalty()
        if matrix:
            label_width = max((len(label) for label in labels.values()), default=0)
            column_width = max(8, label_width)
            matrix_scores = {item.id: {} for item in items}
            for item in items:
                for row in store.neighbors(item.id):
                    other = row["item_b"] if row["item_a"] == item.id else row["item_a"]
                    if other in labels:
                        score = effective_score(row, penalty)
                        matrix_scores[item.id][other] = score
                        matrix_scores[other][item.id] = score
            ui.print_(
                " " * label_width
                + "  "
                + "  ".join(f"{labels[item.id]:>{column_width}}" for item in items)
            )
        for item in items:
            scores = {}
            if matrix:
                cells = []
                for other in items:
                    if other.id == item.id:
                        value = "1.000000"
                    elif other.id in matrix_scores[item.id]:
                        value = f"{matrix_scores[item.id][other.id]:.6f}"
                    else:
                        value = "-"
                    cells.append(f"{value:>{column_width}}")
                ui.print_(f"{labels[item.id]:<{label_width}}  " + "  ".join(cells))
            else:
                for row in store.neighbors(item.id):
                    other = row["item_b"] if row["item_a"] == item.id else row["item_a"]
                    if other in labels:
                        score = effective_score(row, penalty)
                        if score >= threshold:
                            scores[other] = score
                if debug:
                    ui.print_(labels[item.id] + ":")
                    for other, score in sorted(
                        scores.items(), key=lambda pair: (-pair[1], pair[0])
                    ):
                        formatted_score = ui.colorize("text_success", f"({score:.6f})")
                        ui.print_(f"\t\t{labels[other]}\t\t{formatted_score}")
                else:
                    cells = [
                        f"{labels[other]}{ui.colorize('text_success', f'({score:.6f})')}"
                        for other, score in sorted(
                            scores.items(), key=lambda pair: (-pair[1], pair[0])
                        )
                    ]
                    ui.print_(labels[item.id] + ":\t" + "; ".join(cells))

    @staticmethod
    def _artists(item):
        return (set(item.get("artists") or []) | {item.artist}) - {"", None}

    @staticmethod
    def _bare(artist):
        return "".join(c for c in unidecode(artist).casefold() if c.isalnum())

    def _refresh_candidates(self, lib, targets):
        """Match distinct artist values before fetching any track models."""
        from beets.autotag.distance import string_dist

        scopes = {frozenset(self._artists(item)) for item in targets.values()}
        artist_names = set().union(*scopes)
        if not artist_names:
            return {scope: [] for scope in scopes}
        fuzzy = self.config["fuzzy_artists"].get(bool)
        bare_ascii = self.config["bare_ascii_artists"].get(bool)
        threshold = self._number("artist_threshold") if fuzzy else 0.0

        library_names = {}
        with lib.transaction() as tx:
            for field in ("artist", "artists"):
                rows = tx.query(
                    f"SELECT DISTINCT {field} FROM items "
                    f"WHERE {field} IS NOT NULL AND {field} != ''"
                )
                for row in rows:
                    value = row[field]
                    names = (
                        Item._fields["artists"].from_sql(value)
                        if field == "artists"
                        else [value]
                    )
                    for name in names:
                        if name:
                            library_names.setdefault(name, set()).add((field, value))

        exact_index = {}
        bare_index = {}
        for name, values in library_names.items():
            exact_index.setdefault(name.casefold(), set()).update(values)
            if bare_ascii:
                bare = self._bare(name)
                if bare:
                    bare_index.setdefault(bare, set()).update(values)
        matches = {}
        for name in artist_names:
            values = set(exact_index.get(name.casefold(), ()))
            if bare_ascii:
                values.update(bare_index.get(self._bare(name), ()))
            if fuzzy:
                for other, other_values in library_names.items():
                    if string_dist(name, other) <= threshold:
                        values.update(other_values)
            matches[name] = values

        candidates = {}
        for scope in scopes:
            values = sorted(set().union(*(matches[name] for name in scope)))
            selected = {}
            # Bound SQL parameters and expression size for large artist batches.
            for start in range(0, len(values), 200):
                batch = values[start : start + 200]
                query = OrQuery(
                    [
                        InQuery(field, [value for key, value in batch if key == field])
                        for field in ("artist", "artists")
                        if any(key == field for key, _ in batch)
                    ]
                )
                selected.update((item.id, item) for item in lib.items(query))
            candidates[scope] = list(selected.values())
        return candidates

    def refresh_items(self, lib, items):
        """Refresh incident artist-scoped pairs once per batch, preserving overrides."""
        targets = {item.id: item for item in items if item.id}
        if not targets:
            return
        candidates = self._refresh_candidates(lib, targets)
        use_refresh_state = self.config["refresh_state"].get(bool)
        store = SimilarityStore(lib, refresh_state=use_refresh_state)
        estimate = sum(
            len(candidates[frozenset(self._artists(item))])
            for item in targets.values()
        )
        if (
            len(targets) >= self.config["warn_items"].get(int)
            or estimate >= self.config["warn_pairs"].get(int)
        ):
            self._log.warning(
                "Automatic similarity refresh: up to {} candidate comparisons", estimate
            )

        def matching_items(first):
            for second in candidates[frozenset(self._artists(first))]:
                # Both artist modes form a union, and targeted pairs run only once.
                if second.id != first.id and not (
                    second.id in targets and second.id < first.id
                ):
                    yield second

        total = sum(
            1 for first in targets.values() for second in matching_items(first)
        )
        artist_count = len(set().union(*(self._artists(i) for i in targets.values())))
        self._log.info(
            "Automatic similarity refresh: {} artists across {} items; "
            "{} similarity scores to recompute",
            artist_count, len(targets), total,
        )
        progress = 0
        for index, first in enumerate(targets.values(), 1):
            self._log.info("Item {}/{}: {}", index, len(targets), first.id)
            for second in matching_items(first):
                progress += 1
                store.put(
                    first.id, second.id,
                    self.score(first, second, progress=(progress, total)),
                )
        if use_refresh_state:
            for item in targets.values():
                store.set_refresh_signature(item.id, self._refresh_signature(item))

    def _refresh_signature(self, item):
        fields = REFRESH_FIELDS
        if self.config["acoustic"].get(bool):
            fields += ACOUSTIC_REFRESH_FIELDS
        values = [item.get(field) for field in fields]
        serialized = json.dumps(
            values, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _database_change(self, lib, model):
        if isinstance(model, Item) and model.id:
            if self.config["refresh_state"].get(bool):
                store = SimilarityStore(lib, refresh_state=True)
                signature = self._refresh_signature(model)
                if store.refresh_signature(model.id) == signature:
                    return
            self._pending.setdefault(lib, set()).add(model.id)

    def _flush(self, lib=None):
        libraries = [lib] if lib is not None else list(self._pending)
        for library in libraries:
            ids = self._pending.pop(library, set())
            if not ids:
                continue
            self.refresh_items(
                library,
                [item for i in ids if (item := library.get_item(i)) is not None],
            )

    def _import_stage(self, session, task):
        for item in task.imported_items():
            self._database_change(session.lib, item)
        self._flush(session.lib)

    def _item_removed(self, item):
        if not item.id or not self.config["remove_on_delete"].get(bool):
            return
        store = SimilarityStore(item.db)
        rows = store.neighbors(item.id)
        if not rows:
            return
        ui.print_("Similarity pairs touching the deleted item:")
        for row in sorted(
            rows,
            key=lambda pair: pair["item_b"] if pair["item_a"] == item.id
            else pair["item_a"],
        ):
            other_id = (
                row["item_b"] if row["item_a"] == item.id else row["item_a"]
            )
            other = item.db.get_item(other_id)
            other_label = (
                f"{other_id} ({other.artist} - {other.title})"
                if other is not None
                else str(other_id)
            )
            ui.print_(f"  {item.id} ({item.artist} - {item.title}) <-> {other_label}")
        if self._confirm("Remove these similarity pairs?"):
            store.remove([item.id])
