import contextlib
import io
import math
import os
import sqlite3
import tempfile
import types
import unittest
from unittest.mock import patch

from beets import config, ui
from beets.library import Item, Library

from beetsplug.similarity import (
    DEFAULT_WEIGHTS,
    SimilarityPlugin,
    SimilarityStore,
    effective_score,
    pair_key,
)


class SimilarityTest(unittest.TestCase):
    def setUp(self):
        config["similarity"].set(
            {
                "auto": False,
                "refresh_state": True,
                "remove_on_delete": False,
                "threshold": 0.8,
                "acoustic": True,
                "dissociate_penalty": "to-zero",
                "weights": dict(DEFAULT_WEIGHTS),
                "warn_items": 1000,
                "warn_pairs": 500000,
                "fuzzy_artists": False,
                "bare_ascii_artists": False,
                "artist_threshold": 0.2,
            }
        )
        self.plugin = SimilarityPlugin()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with contextlib.redirect_stdout(io.StringIO()):
            self.lib = Library(os.path.join(self.directory.name, "library.db"))
        self.addCleanup(self.lib._close)
        self.store = SimilarityStore(self.lib)

    def item(self, **values):
        defaults = {
            "title": "Around the World",
            "artist": "Daft Punk",
            "album": "Homework",
            "length": 429.0,
            "track": 7,
            "disc": 1,
            "acoustid_id": "shared-acoustic-id",
        }
        defaults.update(values)
        item = Item(**defaults)
        self.lib.add(item)
        return item

    def command(self, *args):
        command = self.plugin.commands()[0]
        opts, query = command.parser.parse_args(list(args))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            command.func(self.lib, opts, query)
        return output.getvalue()

    def test_identical_and_bitrate_invariance(self):
        first = self.item(bitrate=128000)
        second = self.item(bitrate=320000)
        self.assertEqual(self.plugin.score(first, second), 1.0)
        second.bitrate = 64000
        self.assertEqual(self.plugin.score(first, second), 1.0)

    def test_distance_uses_autotagger_and_is_symmetric(self):
        from beets.autotag import match

        first = self.item(mb_trackid="recording")
        second = self.item(
            title="Another Track", artist="Other Artist", length=80, mb_trackid=""
        )
        with patch.object(match, "track_distance", wraps=match.track_distance) as distance:
            forward = self.plugin.score(first, second)
        self.assertEqual(distance.call_count, 2)
        self.assertLess(forward, 1)
        self.assertEqual(forward, self.plugin.score(second, first))
        with self.assertRaisesRegex(ValueError, "no database"):
            _ = distance.call_args.args[0].db

    def test_weights_do_not_modify_global_match_config(self):
        before = config["match"].get()
        first, second = self.item(), self.item(title="different")
        default = self.plugin.score(first, second)
        self.plugin.config["weights"]["track_title"] = 100
        self.assertLess(self.plugin.score(first, second), default)
        self.assertEqual(config["match"].get(), before)

    def test_scoring_logs_pair_scores_and_bulk_progress(self):
        first, second, third = self.item(), self.item(), self.item()
        with (
            patch.object(self.plugin._log, "isEnabledFor", return_value=False),
            patch.object(self.plugin._log, "debug") as debug,
        ):
            self.plugin.score(first, second)
        debug.assert_called_once()
        self.assertIn("Similarity score", debug.call_args.args[0])
        self.assertEqual(debug.call_args.args[1:3], (first.id, second.id))

        with patch.object(self.plugin._log, "info") as info:
            self.command("--add", "-f", "artist:Daft Punk")
        self.assertEqual(info.call_count, 4)
        self.assertEqual(
            [call.args[1:] for call in info.call_args_list[:3]],
            [(1, 3), (2, 3), (3, 3)],
        )
        self.assertIn("complete", info.call_args.args[0])
        self.assertEqual(info.call_args.args[1:], (3, 3))

    def test_debug_penalties_explain_final_score(self):
        first = self.item(mb_trackid="recording", mb_releasegroupid="group-one")
        second = self.item(
            title="Another Track", artist="Other Artist", length=80,
            track=2, disc=2, mb_releasegroupid="group-two", album="Discovery",
            acoustid_id="different-acoustic-id",
        )
        with (
            patch.object(self.plugin._log, "isEnabledFor", return_value=True),
            patch.object(self.plugin._log, "debug") as debug,
        ):
            score = self.plugin.score(first, second)
        penalties = [
            call.args for call in debug.call_args_list
            if call.args[0].startswith("Similarity penalty")
        ]
        self.assertAlmostEqual(sum(args[-1] for args in penalties), 1 - score)
        forward = {args[3]: args for args in penalties if args[1] == first.id}
        reverse = {args[3]: args for args in penalties if args[1] == second.id}
        self.assertEqual(set(forward), set(DEFAULT_WEIGHTS) - {"data_source"})
        self.assertNotIn("track_id", reverse)
        self.assertIn("'Around the World' vs 'Another Track'", forward["track_title"][4])
        self.assertEqual(forward["track_length"][4], "length: 429.0 vs 80")
        self.assertIn("acoustid_id:", forward["acoustic"][4])
        for key, args in forward.items():
            self.assertEqual(args[5], DEFAULT_WEIGHTS[key])
            self.assertEqual(args[-1], args[-2] / 2)
        summary = debug.call_args.args
        self.assertEqual(summary[1:4], (first.id, second.id, score))
        self.assertAlmostEqual(score, 1 - (summary[4] + summary[5]) / 2)

    def test_debug_omits_zero_weight_and_missing_evidence(self):
        first = self.item(acoustid_id="")
        second = self.item(acoustid_id="", title="Other title")
        self.plugin.config["weights"]["track_title"] = 0
        with (
            patch.object(self.plugin._log, "isEnabledFor", return_value=True),
            patch.object(self.plugin._log, "debug") as debug,
        ):
            self.assertEqual(self.plugin.score(first, second), 1)
        self.assertFalse(any(
            call.args[0].startswith("Similarity penalty")
            for call in debug.call_args_list
        ))

    def test_debug_fingerprint_evidence_without_dumping_fingerprints(self):
        first = self.item(acoustid_fingerprint="fingerprint-a")
        second = self.item(acoustid_fingerprint="fingerprint-b")
        with (
            patch.object(self.plugin, "_acoustic_distance", return_value=0.75),
            patch.object(self.plugin._log, "isEnabledFor", return_value=True),
            patch.object(self.plugin._log, "debug") as debug,
        ):
            self.plugin.score(first, second)
        messages = [
            call.args[0].format(*call.args[1:]) for call in debug.call_args_list
        ]
        self.assertTrue(any("saved fingerprints" in message for message in messages))
        self.assertFalse(any("fingerprint-a" in message for message in messages))
        self.assertFalse(any("fingerprint-b" in message for message in messages))

    def test_release_group_and_album_penalties(self):
        first = self.item(mb_releasegroupid="group-one")
        same = self.item(mb_releasegroupid="group-one")
        other = self.item(mb_releasegroupid="group-two", album="Discovery")
        self.assertGreater(self.plugin.score(first, same), self.plugin.score(first, other))

    def test_acoustid_penalty_and_opt_out(self):
        first = self.item()
        second = self.item(acoustid_id="different-acoustic-id")
        self.assertLess(self.plugin.score(first, second), 1)
        self.plugin.config["acoustic"] = False
        with patch.object(self.plugin, "_acoustic_distance", side_effect=AssertionError):
            self.assertEqual(self.plugin.score(first, second), 1)

    def test_missing_acoustic_evidence_is_omitted(self):
        first, second = self.item(acoustid_id=""), self.item(acoustid_id="")
        self.assertIsNone(self.plugin._acoustic_distance(first, second))
        score_with_acoustic_enabled = self.plugin.score(first, second)
        self.plugin.config["acoustic"] = False
        self.assertEqual(score_with_acoustic_enabled, self.plugin.score(first, second))

    def test_fingerprints_are_compared_even_when_acoustid_ids_agree(self):
        first = self.item(acoustid_fingerprint="fingerprint-a")
        second = self.item(acoustid_fingerprint="fingerprint-b")
        fake_acoustid = types.SimpleNamespace(compare_fingerprints=lambda a, b: 0.25)
        fake_chromaprint = types.SimpleNamespace(FingerprintError=type("FPError", (Exception,), {}))
        with patch.dict("sys.modules", acoustid=fake_acoustid, chromaprint=fake_chromaprint):
            with patch.object(fake_acoustid, "compare_fingerprints", return_value=0.25) as compare:
                self.assertEqual(self.plugin._acoustic_distance(first, second), 0.75)
            self.assertEqual(compare.call_count, 2)
            self.assertEqual(compare.call_args_list[0].args[0][1], b"fingerprint-a")
            self.assertEqual(compare.call_args_list[0].args[1][1], b"fingerprint-b")

    def test_asymmetric_fingerprint_alignment_is_averaged(self):
        first = self.item(acoustid_fingerprint="a")
        second = self.item(acoustid_fingerprint="b")
        fake = types.SimpleNamespace(
            compare_fingerprints=lambda a, b: 0.8 if a[1] < b[1] else 0.2
        )
        fake_chromaprint = types.SimpleNamespace(FingerprintError=type("FPError", (Exception,), {}))
        with patch.dict("sys.modules", acoustid=fake, chromaprint=fake_chromaprint):
            self.assertEqual(self.plugin._acoustic_distance(first, second), 0.5)
            self.assertEqual(self.plugin.score(first, second), self.plugin.score(second, first))

    def test_fingerprint_failures_are_actionable(self):
        first = self.item(acoustid_fingerprint="bad")
        second = self.item(acoustid_fingerprint="bad")
        fake = types.SimpleNamespace(compare_fingerprints=lambda a, b: 1)
        error = type("FPError", (Exception,), {})
        with patch.dict("sys.modules", acoustid=fake, chromaprint=types.SimpleNamespace(FingerprintError=error)):
            for failure in (ValueError(), ZeroDivisionError(), error()):
                with patch.object(fake, "compare_fingerprints", side_effect=failure):
                    with self.assertRaisesRegex(ui.UserError, "regenerate"):
                        self.plugin.score(first, second)
        with patch.dict("sys.modules", chromaprint=None):
            with self.assertRaisesRegex(ui.UserError, "libchromaprint"):
                self.plugin.score(first, second)

    def test_invalid_configuration_rejected(self):
        first, second = self.item(), self.item()
        self.plugin.config["weights"]["track_title"] = -1
        with self.assertRaises(ui.UserError):
            self.plugin.score(first, second)
        for value in (0, -0.5, 1.1, math.nan, math.inf):
            with self.assertRaises(ui.UserError):
                self.plugin._threshold(value)
        self.plugin.config["weights"]["track_title"] = 4
        for key in self.plugin.config["weights"].keys():
            self.plugin.config["weights"][key] = 0
        with self.assertRaisesRegex(ui.UserError, "applicable"):
            self.plugin.score(first, second)

    def test_store_canonical_and_persistent(self):
        first, second = self.item(), self.item()
        self.store.put(second.id, first.id, 0.9)
        self.assertEqual(self.store.get(first.id, second.id)["score"], 0.9)
        reopened = Library(self.lib.path)
        self.addCleanup(reopened._close)
        self.assertEqual(SimilarityStore(reopened).get(second.id, first.id)["score"], 0.9)
        self.assertEqual(len(self.store.neighbors(first.id)), 1)
        with self.assertRaises(ValueError):
            pair_key(first.id, first.id)

    def test_storage_size_includes_table_and_index_only(self):
        empty_size = self.store.storage_size()
        with self.lib.transaction() as tx:
            page_size = tx.query("PRAGMA page_size")[0][0]
            tx.mutate("CREATE TABLE unrelated (payload BLOB)")
            tx.mutate("INSERT INTO unrelated VALUES (zeroblob(1048576))")
        self.assertEqual(empty_size, 2 * page_size)
        self.assertEqual(self.store.storage_size(), empty_size)

        items = [self.item() for _ in range(50)]
        with self.lib.transaction() as tx:
            for first in items:
                for second in items:
                    if first.id < second.id:
                        self.store.put(first.id, second.id, 0.9)
        with self.lib.transaction() as tx:
            expected = sum(
                row["size"] for row in tx.query(
                    "SELECT SUM(pgsize) AS size FROM dbstat "
                    "WHERE name IN ('similarity', 'similarity_item_b')"
                )
            )
        self.assertGreater(expected, empty_size)
        self.assertEqual(self.store.storage_size(), expected)
        with patch.object(self.lib, "items", side_effect=AssertionError):
            self.assertEqual(
                self.command("--size"), f"Similarity storage: {expected} bytes\n"
            )
        self.assertEqual(len(self.store.neighbors(items[0].id)), 49)

    def test_storage_size_includes_refresh_state_table(self):
        item = self.item()
        similarity_size = self.store.storage_size()
        state_store = SimilarityStore(self.lib, refresh_state=True)
        state_store.set_refresh_signature(item.id, "signature")
        with self.lib.transaction() as tx:
            expected = tx.query(
                "SELECT SUM(pgsize) AS size FROM dbstat "
                "WHERE name IN ('similarity', 'similarity_item_b', "
                "'similarity_refresh_state')"
            )[0]["size"]
        self.assertEqual(state_store.storage_size(), expected)
        self.assertGreater(state_store.storage_size(), similarity_size)

    def test_size_does_not_require_valid_scoring_configuration(self):
        self.plugin.config["threshold"] = 0
        self.assertEqual(
            self.command("--size"),
            f"Similarity storage: {self.store.storage_size()} bytes\n",
        )

    def test_storage_size_reports_missing_dbstat(self):
        with self.lib.transaction() as tx, patch.object(
            self.lib, "transaction", return_value=contextlib.nullcontext(tx)
        ):
            with patch.object(
                tx, "query", side_effect=sqlite3.OperationalError("no such table: dbstat")
            ):
                with self.assertRaisesRegex(ui.UserError, "SQLITE_ENABLE_DBSTAT_VTAB"):
                    self.store.storage_size()
            with patch.object(
                tx, "query", side_effect=sqlite3.OperationalError("database is locked")
            ):
                with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                    self.store.storage_size()

    def test_manual_choices_survive_refresh_and_threshold_changes(self):
        first, second = self.item(), self.item()
        self.command("--associate", str(first.id), str(second.id))
        self.store.put(first.id, second.id, 0.1)
        self.assertEqual(effective_score(self.store.get(first.id, second.id)), 1)
        self.assertEqual(
            [(item.id, score) for item, score in self.plugin.similar_items(self.lib, first, 1)],
            [(second.id, 1)],
        )
        self.command("--dissociate", str(first.id), str(second.id), "--force")
        self.store.put(first.id, second.id, 1)
        self.assertEqual(effective_score(self.store.get(first.id, second.id)), 0)
        self.assertEqual(self.plugin.similar_items(self.lib, first, 0.01), [])

    def test_dissociate_penalty_is_configurable_and_preserves_base_score(self):
        first, second = self.item(), self.item(title="Different")
        base_score = 0.7
        self.store.put(first.id, second.id, base_score)
        self.command("--dissociate", str(first.id), str(second.id), "--force")

        self.plugin.config["dissociate_penalty"] = 0.2
        row = self.store.get(first.id, second.id)
        self.assertEqual(row["score"], base_score)
        self.assertAlmostEqual(effective_score(row, self.plugin._dissociate_penalty()), 0.5)
        results = self.plugin.similar_items(self.lib, first, threshold=0.4)
        self.assertEqual([item.id for item, _ in results], [second.id])
        self.assertAlmostEqual(results[0][1], 0.5)

        self.plugin.config["dissociate_penalty"] = 0.8
        self.assertEqual(effective_score(row, self.plugin._dissociate_penalty()), 0)
        self.plugin.config["dissociate_penalty"] = "to-zero"
        self.assertEqual(effective_score(row, self.plugin._dissociate_penalty()), 0)

    def test_invalid_dissociate_penalty_rejected(self):
        for value in ("invalid", -0.1, 1.1, math.nan, math.inf, True):
            self.plugin.config["dissociate_penalty"] = value
            with self.subTest(value=value), self.assertRaisesRegex(
                ui.UserError, "dissociate_penalty"
            ):
                self.plugin._dissociate_penalty()

    def test_query_scopes_add_and_overwrite_confirmation(self):
        first, second = self.item(), self.item()
        third = self.item(artist="Toto")
        self.command("--add", "-f", "artist:Daft Punk")
        self.assertIsNotNone(self.store.get(first.id, second.id))
        self.assertIsNone(self.store.get(first.id, third.id))
        with patch.object(self.plugin, "_confirm", return_value=False) as confirm:
            with patch.object(self.plugin, "score") as score:
                self.command("--add", "artist:Daft Punk")
            confirm.assert_called_once()
            score.assert_not_called()
        with patch.object(self.plugin, "_confirm", side_effect=AssertionError):
            self.command("--add", "-f", "artist:Daft Punk")

    def test_unscoped_and_large_computations_warn(self):
        self.item()
        self.item()
        with patch.object(self.plugin._log, "warning") as warning:
            with patch.object(self.plugin, "_confirm", return_value=False):
                self.command("--add")
            warning.assert_called_once()
        self.plugin.config["warn_items"] = 2
        with patch.object(self.plugin, "_confirm", return_value=False) as confirm:
            self.command("--add", "artist:Daft Punk")
            confirm.assert_called_once()
        with patch.object(self.plugin._log, "warning") as warning:
            self.command("--add", "-f")
            warning.assert_called_once()

    def test_lists_threshold_matrix_and_unknown_pairs(self):
        first, second, third = self.item(), self.item(), self.item()
        self.store.put(first.id, second.id, 0.9)
        self.store.put(first.id, third.id, 0.2)
        with patch.object(ui, "colorize", autospec=True, side_effect=lambda _, text: text):
            output = self.command("--list", "--thres", "0.8")
        self.assertIn(f"{first.id}:\t{second.id}(0.900000)", output)
        self.assertIn(f"{third.id}:\t\n", output)
        self.assertNotIn("(0.200000)", output)
        matrix = self.command("--list", "--matrix").splitlines()
        rows = [line.split() for line in matrix]
        self.assertEqual(rows[0], [str(first.id), str(second.id), str(third.id)])
        self.assertEqual(rows[1], [str(first.id), "1.000000", "0.900000", "0.200000"])
        self.assertEqual(rows[2], [str(second.id), "0.900000", "1.000000", "-"])
        self.assertEqual(rows[3], [str(third.id), "0.200000", "-", "1.000000"])
        self.assertNotIn("\t", matrix[0])

    def test_non_matrix_lists_order_neighbors_by_effective_score(self):
        first = self.item(path="album/first.flac")
        lower = self.item(path="album/lower.flac")
        higher = self.item(path="album/higher.flac")
        self.store.put(first.id, lower.id, 0.85)
        self.store.put(first.id, higher.id, 0.95)

        with patch.object(
            ui,
            "colorize",
            autospec=True,
            side_effect=lambda color_name, text: f"[{color_name}]{text}",
        ):
            regular = self.command("--list")
        debug = self.command("--list", "--debug")

        self.assertIn(
            f"{first.id}:\t{higher.id}[text_success](0.950000); "
            f"{lower.id}[text_success](0.850000)",
            regular,
        )
        self.assertLess(
            debug.index("Music/album/higher.flac"),
            debug.index("Music/album/lower.flac"),
        )

    def test_debug_paths_are_short_and_single_line(self):
        self.item(path=b"/music/Daft Punk/Homework/song\tname\n.flac")
        self.assertEqual(
            self.command("--list", "--debug"),
            "Daft Punk/Homework/song\\tname\\n.flac:\n",
        )

    def test_debug_list_indents_pairs_and_filters_by_threshold(self):
        first = self.item(path="album1/toto.flac")
        above = self.item(path="album1/toto.mp3")
        below = self.item(path="album2/toto.flac")
        self.store.put(first.id, above.id, 1)
        self.store.put(first.id, below.id, 0.7)
        with patch.object(
            ui,
            "colorize",
            autospec=True,
            side_effect=lambda color_name, text: f"[{color_name}]{text}",
        ):
            output = self.command("--list", "--debug")

        self.assertIn(
            "Music/album1/toto.flac:\n\t\tMusic/album1/toto.mp3\t\t[text_success](1.000000)",
            output,
        )
        self.assertNotIn("(0.700000)", output)
        self.assertIn("Music/album2/toto.flac:\n", output)

    def test_debug_list_honors_configured_threshold_and_cli_override(self):
        first = self.item(path="album1/first.flac")
        second = self.item(path="album1/second.flac")
        self.store.put(first.id, second.id, 0.7)
        self.plugin.config["threshold"] = 0.7

        for args, included in (
            ((), True),
            (("--thres", "0.8"), False),
            (("--thres", "0.6"), True),
        ):
            with self.subTest(args=args):
                output = self.command("--list", "--debug", *args)
                self.assertEqual("(0.700000)" in output, included)
                self.assertIn("Music/album1/first.flac:\n", output)
                self.assertIn("Music/album1/second.flac:\n", output)

    def test_debug_list_uses_beets_color_configuration(self):
        first = self.item(path="album1/toto.flac")
        above = self.item(path="album1/toto.mp3")
        below = self.item(path="album2/toto.flac")
        self.store.put(first.id, above.id, 0.8)
        self.store.put(first.id, below.id, 0.7)
        original_color = config["ui"]["color"].get()
        self.addCleanup(config["ui"]["color"].set, original_color)

        with patch.dict(os.environ):
            os.environ.pop("NO_COLOR", None)
            for enabled in (True, False):
                with self.subTest(color=enabled):
                    config["ui"]["color"] = enabled
                    output = self.command("--list", "--debug")
                    self.assertIn(
                        ui.colorize("text_success", "(0.800000)"), output
                    )
                    self.assertNotIn("(0.700000)", output)
                    self.assertEqual("\x1b[" in output, enabled)

            config["ui"]["color"] = True
            os.environ["NO_COLOR"] = "1"
            output = self.command("--list", "--debug")
            self.assertIn("(0.800000)", output)
            self.assertNotIn("(0.700000)", output)
            self.assertNotIn("\x1b[", output)

    def test_scoped_remove_and_reset_confirmation_even_with_force(self):
        first, second = self.item(), self.item(artist="Toto")
        third = self.item(artist="Toto")
        self.store.put(first.id, second.id, 1)
        self.store.put(second.id, third.id, 1)
        self.command("--remove", "-f", f"id:{first.id}")
        self.assertIsNone(self.store.get(first.id, second.id))
        with patch.object(self.plugin, "_confirm", return_value=False) as confirm:
            self.command("--remove", "-f")
        confirm.assert_called_once()
        self.assertIsNotNone(self.store.get(second.id, third.id))
        with patch.object(self.plugin, "_confirm", return_value=True):
            self.command("--remove", "-f")
        self.assertIsNone(self.store.get(second.id, third.id))

    def test_invalid_commands(self):
        first = self.item()
        for args in (
            ("--add", "--list"),
            ("--add", "--matrix"),
            ("--size", "--list"),
            ("--size", "--matrix"),
            ("--size", "--debug"),
            ("--size", "artist:Daft Punk"),
            ("--size", "--thres", "0.8"),
            ("--associate", str(first.id), str(first.id)),
            ("--associate", str(first.id), "999"),
            ("--associate", str(first.id), "999", "artist:Daft Punk"),
            ("--thres", "nan"),
        ):
            with self.subTest(args=args), self.assertRaises(ui.UserError):
                self.command(*args)

    def test_artist_modes_deduplicate_both_queries_and_targets(self):
        first = self.item(artist="Beyoncé")
        second = self.item(artist="Beyonce")
        self.item(artist="Other")
        self.plugin.config["fuzzy_artists"] = True
        self.plugin.config["bare_ascii_artists"] = True
        with patch.object(self.plugin, "score", wraps=self.plugin.score) as score:
            self.plugin.refresh_items(self.lib, [first, second, first])
        self.assertEqual(score.call_count, 1)
        self.assertEqual(len(self.store.neighbors(first.id)), 1)

    def test_exact_artist_scope_and_multi_artist_tracks(self):
        first = self.item()
        other = self.item(artist="Toto")
        collaboration = self.item(artist="Guest", artists=["Guest", "Daft Punk"])
        self.plugin.refresh_items(self.lib, [first])
        self.assertIsNone(self.store.get(first.id, other.id))
        self.assertIsNotNone(self.store.get(first.id, collaboration.id))

    def test_artist_edit_preserves_out_of_scope_scores_and_manual_choices(self):
        first, second, third = self.item(), self.item(), self.item()
        self.store.put(first.id, second.id, 1)
        self.store.put(first.id, third.id, 1, manual=-1)
        first.artist = "Toto"
        self.plugin.refresh_items(self.lib, [first])
        self.assertEqual(self.store.get(first.id, second.id)["score"], 1)
        self.assertEqual(self.store.get(first.id, third.id)["manual"], -1)

    def test_refresh_preserves_explicit_cross_artist_scores(self):
        first = self.item()
        second = self.item(artist="Toto")
        self.store.put(first.id, second.id, 0.6)
        first.title = "Updated title"
        self.plugin.refresh_items(self.lib, [first])
        self.assertEqual(self.store.get(first.id, second.id)["score"], 0.6)

    def test_singular_artist_edit_is_used_with_populated_plural_artists(self):
        first = self.item(artist="Old Artist", artists=["Old Artist"])
        second = self.item(artist="New Artist")
        first.artist = "New Artist"
        first.store()
        self.plugin.refresh_items(self.lib, [first])
        self.assertIsNotNone(self.store.get(first.id, second.id))

    def test_change_queue_import_stage_and_removed_item(self):
        first, second = self.item(), self.item()
        self.plugin._database_change(self.lib, first)
        self.plugin._database_change(self.lib, first)
        self.plugin._database_change(self.lib, second)
        task = types.SimpleNamespace(imported_items=lambda: [first, second])
        with patch.object(self.plugin, "score", wraps=self.plugin.score) as score:
            self.plugin._import_stage(types.SimpleNamespace(lib=self.lib), task)
            self.plugin._flush()
        self.assertEqual(score.call_count, 1)
        self.plugin.config["remove_on_delete"] = True
        with patch.object(self.plugin, "_confirm", return_value=True):
            self.plugin._item_removed(first)
        self.assertEqual(self.store.neighbors(second.id), [])

    def test_removed_item_pair_cleanup_is_opt_in_and_confirmed(self):
        first = self.item(title="Deleted track")
        second = self.item(title="Neighbor track")
        self.store.put(first.id, second.id, 0.9)

        with patch.object(self.plugin, "_confirm") as confirm:
            self.plugin._item_removed(first)
        confirm.assert_not_called()
        self.assertIsNotNone(self.store.get(first.id, second.id))

        self.plugin.config["remove_on_delete"] = True
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(
            self.plugin, "_confirm", return_value=False
        ) as confirm:
            self.plugin._item_removed(first)
        confirm.assert_called_once_with("Remove these similarity pairs?")
        self.assertIn(f"{first.id} (Daft Punk - Deleted track)", output.getvalue())
        self.assertIn(f"{second.id} (Daft Punk - Neighbor track)", output.getvalue())
        self.assertIsNotNone(self.store.get(first.id, second.id))

        with patch.object(self.plugin, "_confirm", return_value=True) as confirm:
            self.plugin._item_removed(first)
        confirm.assert_called_once()
        self.assertEqual(self.store.neighbors(second.id), [])

    def test_database_change_ignores_unrelated_metadata(self):
        first = self.item()
        self.plugin.refresh_items(self.lib, [first])

        first.composer = "New Composer"
        first.store()
        with patch.object(self.plugin, "refresh_items") as refresh:
            self.plugin._database_change(self.lib, first)
            self.plugin._flush(self.lib)
        refresh.assert_not_called()

        first.title = "Updated title"
        first.store()
        with patch.object(self.plugin, "refresh_items") as refresh:
            self.plugin._database_change(self.lib, first)
            self.plugin._flush(self.lib)
        refresh.assert_called_once()

    def test_refresh_state_can_be_disabled(self):
        first = self.item()
        self.plugin.config["refresh_state"] = False
        self.plugin.refresh_items(self.lib, [first])
        self.assertFalse(self._refresh_state_table_exists())

        with patch.object(self.plugin, "refresh_items") as refresh:
            self.plugin._database_change(self.lib, first)
            self.plugin._flush(self.lib)
        refresh.assert_called_once()

    def _refresh_state_table_exists(self):
        with self.lib.transaction() as tx:
            return bool(
                tx.query(
                    "SELECT 1 FROM sqlite_schema "
                    "WHERE type='table' AND name='similarity_refresh_state'"
                )
            )

    def test_auto_is_opt_in_and_registers_stage_and_events(self):
        self.assertEqual(self.plugin.import_stages, [])
        config["similarity"]["auto"] = True
        with patch.object(SimilarityPlugin, "register_listener") as register:
            automatic = SimilarityPlugin()
        self.assertEqual(automatic.import_stages, [automatic._import_stage])
        self.assertIn("database_change", [call.args[0] for call in register.call_args_list])
        self.assertIn("cli_exit", [call.args[0] for call in register.call_args_list])


if __name__ == "__main__":
    unittest.main()
