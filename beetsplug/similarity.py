"""Compare local tracks without changing the global autotagger configuration."""

import itertools
import math
import os

from beets import ui
from beets.autotag import match
from beets.autotag.hooks import TrackInfo
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


def pair_key(first, second):
    """Canonical key for a non-self, persisted pair."""
    if not first or not second or first == second:
        raise ValueError("similarity pairs require two distinct persisted item IDs")
    return tuple(sorted((int(first), int(second))))


def effective_score(row):
    """Manual choices survive recomputation and any valid threshold change."""
    if row["manual"]:
        return 1.0 if row["manual"] > 0 else 0.0
    return row["score"]


class SimilarityStore:
    """Sparse, undirected pairs in the same SQLite database as beets."""

    def __init__(self, lib):
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

    def remove(self, ids=None):
        with self.lib.transaction() as tx:
            if ids is None:
                tx.mutate("DELETE FROM similarity")
            else:
                for item_id in ids:
                    tx.mutate(
                        "DELETE FROM similarity WHERE item_a=? OR item_b=?",
                        (item_id, item_id),
                    )


class SimilarityPlugin(BeetsPlugin):
    def __init__(self):
        super().__init__()
        self.config.add(
            {
                "auto": False,
                "threshold": 0.8,
                "acoustic": True,
                "missing_acoustic_penalty": 0.5,
                "weights": DEFAULT_WEIGHTS,
                "warn_items": 1000,
                "warn_pairs": 500000,
                "fuzzy_artists": False,
                "bare_ascii_artists": False,
                "artist_threshold": 0.2,
            }
        )
        self._pending = {}
        self._warned_missing = False
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
        if not self._warned_missing:
            self._log.warning(
                "Missing acoustic evidence; using missing_acoustic_penalty. "
                "Populate fingerprints/AcoustID IDs with chroma for acoustic matching."
            )
            self._warned_missing = True
        return self._number("missing_acoustic_penalty")

    @staticmethod
    def _fingerprint_bytes(value):
        return value.encode("ascii") if isinstance(value, str) else value

    def score(self, first, second):
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
        return max(0.0, min(1.0, 1.0 - sum(distances) / 2))

    def similar_items(self, lib, item, threshold=None):
        """Public API for rating/playlist plugins: (Item, score), best first."""
        threshold = self._threshold(threshold)
        results = []
        for row in SimilarityStore(lib).neighbors(item.id):
            score = effective_score(row)
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
            name for name in ("add", "list", "remove", "associate", "dissociate")
            if getattr(opts, name)
        ]
        if len(actions) > 1:
            raise ui.UserError("Choose only one similar action")
        action = actions[0] if actions else "list"
        threshold = self._threshold(opts.thres)
        if (opts.matrix or opts.debug) and action != "list":
            raise ui.UserError("--matrix and --debug require --list")
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
            for first, second in itertools.combinations(items, 2):
                store.put(first.id, second.id, self.score(first, second))
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
        if matrix:
            label_width = max((len(label) for label in labels.values()), default=0)
            column_width = max(8, label_width)
            matrix_scores = {item.id: {} for item in items}
            for item in items:
                for row in store.neighbors(item.id):
                    other = row["item_b"] if row["item_a"] == item.id else row["item_a"]
                    if other in labels:
                        score = effective_score(row)
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
                        scores[other] = effective_score(row)
                cells = [
                    f"{labels[other]}({score:.6f})"
                    for other, score in sorted(scores.items())
                    if score >= threshold
                ]
                ui.print_(labels[item.id] + ":\t" + "; ".join(cells))

    @staticmethod
    def _artists(item):
        return (set(item.get("artists") or []) | {item.artist}) - {"", None}

    @staticmethod
    def _bare(artist):
        return "".join(c for c in unidecode(artist).casefold() if c.isalnum())

    def _artist_matches(self, first, second):
        # Reuse beets' string heuristic for optional fuzzy artist matching.
        from beets.autotag.distance import string_dist

        for left in self._artists(first):
            for right in self._artists(second):
                if left.casefold() == right.casefold():
                    return True
                if self.config["fuzzy_artists"].get(bool) and string_dist(
                    left, right
                ) <= self._number("artist_threshold"):
                    return True
                if self.config["bare_ascii_artists"].get(bool):
                    bare = self._bare(left)
                    if bare and bare == self._bare(right):
                        return True
        return False

    def refresh_items(self, lib, items):
        """Refresh incident artist-scoped pairs once per batch, preserving overrides."""
        targets = {item.id: item for item in items if item.id}
        if not targets:
            return
        candidates = list(lib.items())
        store = SimilarityStore(lib)
        estimate = len(targets) * max(0, len(candidates) - 1)
        if (
            len(targets) >= self.config["warn_items"].get(int)
            or estimate >= self.config["warn_pairs"].get(int)
        ):
            self._log.warning(
                "Automatic similarity refresh: up to {} candidate comparisons", estimate
            )
        for first in targets.values():
            for second in candidates:
                if first.id == second.id:
                    continue
                # Both artist modes form a union, and targeted pairs run only once.
                if second.id in targets and second.id < first.id:
                    continue
                if self._artist_matches(first, second):
                    store.put(first.id, second.id, self.score(first, second))

    def _database_change(self, lib, model):
        if isinstance(model, Item) and model.id:
            self._pending.setdefault(lib, set()).add(model.id)

    def _flush(self, lib=None):
        libraries = [lib] if lib is not None else list(self._pending)
        for library in libraries:
            ids = self._pending.pop(library, set())
            self.refresh_items(
                library,
                [item for i in ids if (item := library.get_item(i)) is not None],
            )

    def _import_stage(self, session, task):
        for item in task.imported_items():
            self._database_change(session.lib, item)
        self._flush(session.lib)

    def _item_removed(self, item):
        if item.id:
            SimilarityStore(item.db).remove([item.id])
