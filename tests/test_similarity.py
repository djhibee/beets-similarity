import contextlib
import io
import math
import os
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
                "threshold": 0.8,
                "acoustic": True,
                "missing_acoustic_penalty": 0.5,
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

    def test_missing_acoustic_evidence_has_explicit_penalty(self):
        first, second = self.item(acoustid_id=""), self.item(acoustid_id="")
        with patch.object(self.plugin._log, "warning") as warning:
            self.assertLess(self.plugin.score(first, second), 1)
            self.plugin.score(first, second)
        warning.assert_called_once()

    def test_fingerprints_are_compared_even_when_acoustid_ids_agree(self):
        first = self.item(acoustid_fingerprint="fingerprint-a")
        second = self.item(acoustid_fingerprint="fingerprint-b")
        fake_acoustid = types.SimpleNamespace(compare_fingerprints=lambda a, b: 0.25)
        fake_chromaprint = types.SimpleNamespace(FingerprintError=type("FPError", (Exception,), {}))
        with patch.dict("sys.modules", acoustid=fake_acoustid, chromaprint=fake_chromaprint):
            with patch.object(fake_acoustid, "compare_fingerprints", return_value=0.25) as compare:
                self.assertEqual(self.plugin._acoustic_distance(first, second), 0.75)
            self.assertEqual(compare.call_args.args[0][1], b"fingerprint-a")
            self.assertEqual(compare.call_args.args[1][1], b"fingerprint-b")

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
        output = self.command("--list", "--thres", "0.8")
        self.assertIn(f"{first.id}:\t{second.id}(0.900000)", output)
        self.assertIn(f"{third.id}:\t\n", output)
        self.assertNotIn("(0.200000)", output)
        matrix = self.command("--list", "--matrix").splitlines()
        self.assertEqual(matrix[0], f"\t{first.id}\t{second.id}\t{third.id}")
        self.assertEqual(matrix[1], f"{first.id}\t1.000000\t0.900000\t0.200000")
        self.assertEqual(matrix[2], f"{second.id}\t0.900000\t1.000000\t-")

    def test_debug_paths_are_short_and_single_line(self):
        self.item(path=b"/music/Daft Punk/Homework/song\tname\n.flac")
        self.assertEqual(
            self.command("--list", "--debug"),
            "Daft Punk/Homework/song\\tname\\n.flac:\t\n",
        )

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

    def test_artist_edit_removes_stale_scores_but_preserves_manual_choices(self):
        first, second, third = self.item(), self.item(), self.item()
        self.store.put(first.id, second.id, 1)
        self.store.put(first.id, third.id, 1, manual=-1)
        first.artist = "Toto"
        self.plugin.refresh_items(self.lib, [first])
        self.assertIsNone(self.store.get(first.id, second.id))
        self.assertEqual(self.store.get(first.id, third.id)["manual"], -1)

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
        self.plugin._item_removed(first)
        self.assertEqual(self.store.neighbors(second.id), [])

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
