# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Regression checks for selected-run identity and the refresh failure paths.

No model, environment, checkpoint, browser or GPU is constructed.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from mimo_bridge.explainer import current_state as CS
from mimo_bridge.explainer import refresh as RF
from mimo_bridge.tools import build_explainer as BX


class ExplainerRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "H84_preserved"
        self.run.mkdir()
        (self.run / "args.json").write_text("{}", encoding="utf-8")
        self.args = SimpleNamespace(need_gb=2.0, max_snapshots=12, every=30.0, stats_from=None)
        self.target = str(self.root / "explainer.html")

    def test_selection_uses_supervisor_record(self):
        (self.root / "training_continuation.json").write_text(
            json.dumps({"active_run": str(self.run)}), encoding="utf-8")
        self.assertEqual(CS.selected_production_run(self.root), str(self.run))
        (self.run / "args.json").unlink()
        self.assertEqual(CS.selected_production_run(self.root), str(self.run))

    def test_absent_null_off_and_on_are_distinct(self):
        self.assertFalse(CS.setting_record({}, "flag")["recorded"])
        self.assertTrue(CS.setting_record({"flag": None}, "flag")["recorded"])
        self.assertIn("null", CS.setting_record({"flag": None}, "flag")["label"])
        self.assertEqual(CS.setting_record({"flag": "off"}, "flag")["label"], "Off")
        self.assertEqual(CS.setting_record({"flag": "on"}, "flag")["label"], "On")

    def test_unreadable_production_selection_does_not_fall_back_to_newest_experiment(self):
        (self.root / "training_continuation.json").write_text('{"active_run":', encoding="utf-8")
        with self.assertRaises(ValueError):
            CS.selected_production_run(self.root)

    def test_run_pause_does_not_label_a_different_experiment_paused(self):
        (self.root / "training_continuation.json").write_text(
            json.dumps({"active_run": str(self.run)}), encoding="utf-8")
        (self.root / "supervisor_control.json").write_text('{"cmd":"pause"}', encoding="utf-8")
        stats = {"training": [{"training_step": 415075, "timestamp": 1790392283.79}]}
        state = CS.current_state(self.run, {"n_envs": 8}, stats, self.root, [])
        self.assertTrue(state["restart_paused"])
        self.assertIn("415,075", CS.current_state_html(state))
        other = CS.current_state(self.root / "candidate", {}, {}, self.root, [])
        self.assertFalse(other["restart_paused"])
        self.assertIn("experiment or earlier", CS.current_state_html(other))

    def test_user_supplied_flag_values_are_html_escaped(self):
        state = CS.current_state(self.run, {"motor_input_fold": "<script>"}, {}, self.root, [])
        self.assertNotIn("<script>", CS.current_state_html(state))
        self.assertIn("&lt;script&gt;", CS.current_state_html(state))

    def test_all_displayed_repair_settings_are_published(self):
        self.assertTrue({name for name, _ in CS.REPAIR_SETTINGS}.issubset(BX.PUBLISHED_ARGS))

    def test_enabled_repair_mode_does_not_imply_replacement_enabled(self):
        args = {"liveness_repair": "on", "liveness_rule": "none"}
        self.assertEqual(CS.setting_record(args, "liveness_repair")["label"], "On")
        self.assertIn("No automatic replacement", CS.setting_record(args, "liveness_rule")["label"])

    def test_display_build_time_is_ignored_without_ignoring_content(self):
        old = b'{"generated_at":"2026-09-27 22:00:00","generated_label":"Sep 27, 2026 at 10:00:00 PM PDT","run":"old"}'
        new = old.replace(b"22:00:00", b"22:00:01").replace(b"10:00:00", b"10:00:01")
        self.assertTrue(BX._same_but_build_time(old, new))
        self.assertFalse(BX._same_but_build_time(old, new.replace(b'"old"', b'"new"')))

    def test_stamp_tracks_source_control_and_run_selection(self):
        stamp = BX._sources_stamp(str(self.run))
        watched = {Path(p).name for p, *_ in stamp}
        self.assertTrue({"build_explainer.py", "schema.py", "walkthrough.py", "viewer.js",
                         "viewer.css", "current_state.py", "refresh.py", "control.json",
                         "training_continuation.json", "supervisor_control.json"}.issubset(watched))
        (self.run / "control.json").write_text('{"cmd":"snapshot_and_stop"}', encoding="utf-8")
        self.assertNotEqual(stamp, BX._sources_stamp(str(self.run)))

    def test_failed_build_retries_identical_inputs(self):
        calls = []
        def command(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=1 if len(calls) == 2 else 0, stdout="allowed", stderr="")
        RF.refresh_loop(self.args, lambda: str(self.run), lambda *a: "same", self.target,
                        run_command=command, sleep=lambda s: None, max_cycles=3)
        self.assertEqual(len(calls), 4)  # guard/build fail, guard/build pass; third cycle idle
        receipt = json.loads(Path(self.target + ".refresh.json").read_text())
        self.assertEqual(receipt["state"], "unchanged")
        self.assertIsNotNone(receipt["last_success_at"])

    def test_guard_refusal_retries_without_launching_builder(self):
        Path(self.target).write_text("last valid page", encoding="utf-8")
        calls = []
        def command(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=3, stdout="memory reserved for trial", stderr="")
        RF.refresh_loop(self.args, lambda: str(self.run), lambda *a: "same", self.target,
                        run_command=command, sleep=lambda s: None, max_cycles=2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all("commit_guard.py" in cmd[2] for cmd in calls))
        self.assertEqual(Path(self.target).read_text(encoding="utf-8"), "last valid page")

    def test_missing_selected_settings_retry_without_switching_to_an_experiment(self):
        (self.run / "args.json").unlink()
        checks = []
        calls = []
        def choose():
            checks.append(True)
            if len(checks) == 2:
                (self.run / "args.json").write_text("{}", encoding="utf-8")
            return str(self.run)
        def command(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="allowed", stderr="")
        RF.refresh_loop(self.args, choose, lambda *a: "same", self.target,
                        run_command=command, sleep=lambda s: None, max_cycles=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1][calls[-1].index("--run") + 1], str(self.run))

    def test_watcher_follows_production_resume_and_preserves_chart_source(self):
        other = self.root / "H84_resumed"
        other.mkdir()
        (other / "args.json").write_text("{}", encoding="utf-8")
        selected = iter([str(self.run), str(other)])
        self.args.stats_from = "separate chart run"
        calls = []
        def command(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="allowed", stderr="")
        RF.refresh_loop(self.args, lambda: next(selected), lambda *a: "same", self.target,
                        run_command=command, sleep=lambda s: None, max_cycles=2)
        children = [cmd for cmd in calls if "build_explainer.py" in cmd[2]]
        self.assertEqual([cmd[cmd.index("--run") + 1] for cmd in children], [str(self.run), str(other)])
        self.assertTrue(all("--watch-child" in cmd and "--stats-from" in cmd for cmd in children))

    # THE OUTPUT FENCE (the decision of Oct 15, 2026, arm1-failure-reconciled-output-fence-20261015): the watcher
    # wrote <run>/explainer.html into a protected run folder; --no-run-copy writes only the requested output, reaches
    # the fresh child builder, and refuses an output inside the run under any spelling
    def test_no_run_copy_reaches_the_child_builder_and_the_old_mode_keeps_both_targets(self):
        _, child = RF.build_commands(self.args, str(self.run), self.target)
        self.assertNotIn("--no-run-copy", child)
        self.assertEqual(RF.destinations(self.args, str(self.run), self.target),
                         [str(self.run / "explainer.html"), self.target])
        self.args.no_run_copy = True
        _, child = RF.build_commands(self.args, str(self.run), self.target)
        self.assertIn("--no-run-copy", child)
        self.assertEqual(RF.destinations(self.args, str(self.run), self.target), [self.target])

    def test_an_output_inside_the_run_is_found_under_every_spelling(self):
        import os
        inside = [self.run / "explainer.html", self.run, str(self.run) + os.sep,
                  self.run / "sub" / ".." / "explainer.html", str(self.run / "explainer.html").upper(),
                  os.path.relpath(self.run / "explainer.html")]
        # the spellings realpath does not rewrite (the independent recheck of Oct 15, 2026, finding A-M1): the
        # extended-length prefix, and the administrative and loopback shares of the run's drive where this machine
        # serves them; the identity test (os.path.samefile) finds each
        full = os.path.abspath(self.run / "explainer.html")
        inside.append("\\\\?\\" + full)
        drive, rest = os.path.splitdrive(full)
        if os.name == "nt" and drive.endswith(":"):
            for host in ("localhost", "127.0.0.1"):
                share = "\\\\%s\\%s$" % (host, drive[0])
                if os.path.exists(share + "\\"):
                    inside.append(share + rest)
        for spelling in inside:
            self.assertTrue(RF.output_inside_run(spelling, self.run), spelling)
            self.assertTrue(RF.output_inside_run(spelling, str(self.run).lower()), spelling)
        for outside in (self.target, self.root / "H84_preserved_other" / "explainer.html", self.root,
                        "\\\\?\\" + os.path.abspath(self.target)):
            self.assertFalse(RF.output_inside_run(outside, self.run), outside)

    def test_the_chart_run_of_stats_from_is_fenced_too_and_an_unchoosable_run_refuses(self):
        # the independent recheck's findings A-m2 and A-m1
        import os
        charts = self.root / "chart_run"
        charts.mkdir()
        (charts / "args.json").write_text("{}", encoding="utf-8")
        inside = str(charts / "page.html")
        self.assertEqual(RF.inputs_holding(inside, self.run, str(charts)), [str(charts)])
        self.assertEqual(RF.inputs_holding(self.target, self.run, str(charts)), [])
        with self.assertRaises(ValueError):
            BX.build(str(self.run), inside, 12, False, 30.0, stats_from=str(charts), run_copy=False)
        self.assertEqual(BX.main(["--run", str(self.run), "--out", inside, "--stats-from", str(charts),
                                  "--no-run-copy"]), 2)
        self.args.no_run_copy, self.args.stats_from = True, str(charts)
        calls = []
        RF.refresh_loop(self.args, lambda: str(self.run), lambda *a: "same", inside,
                        run_command=lambda cmd, **k: calls.append(cmd), sleep=lambda s: None, max_cycles=1)
        self.assertEqual(calls, [])
        self.assertEqual(sorted(p.name for p in charts.iterdir()), ["args.json"])
        # following the production record (no --run) with the record unreadable: refused before the lock is made
        original = BX.selected_production_run
        def unreadable(_saved):
            raise ValueError("the production record cannot be read")
        BX.selected_production_run = unreadable
        try:
            out = str(self.root / "elsewhere" / "page.html")
            self.assertEqual(BX.main(["--watch", "--out", out, "--no-run-copy"]), 2)
            self.assertFalse(os.path.exists(out + ".refresh.lock"))
        finally:
            BX.selected_production_run = original

    def test_no_run_copy_refuses_an_output_inside_the_run_before_building(self):
        self.args.no_run_copy = True
        inside = str(self.run / "sub" / ".." / "page.html")
        calls = []
        def command(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="allowed", stderr="")
        RF.refresh_loop(self.args, lambda: str(self.run), lambda *a: "same", inside,
                        run_command=command, sleep=lambda s: None, max_cycles=1)
        self.assertEqual(calls, [])
        # nothing written inside the run, not even the receipt that sits beside the output
        self.assertFalse(Path(inside + ".refresh.json").exists())
        with self.assertRaises(ValueError):
            BX.build(str(self.run), inside, 12, False, 30.0, run_copy=False)
        self.assertEqual(sorted(p.name for p in self.run.iterdir()), ["args.json"])
        self.assertEqual(BX.main(["--run", str(self.run), "--out", inside, "--no-run-copy"]), 2)
        self.assertEqual(sorted(p.name for p in self.run.iterdir()), ["args.json"])

    def test_output_lock_excludes_duplicate_and_releases(self):
        lock = self.target + ".lock"
        with RF.output_lock(lock):
            with self.assertRaises(RuntimeError):
                with RF.output_lock(lock):
                    self.fail("duplicate accepted")
        with RF.output_lock(lock):
            pass


if __name__ == "__main__":
    unittest.main()


def test_the_header_carries_no_dated_decision_and_links_the_dashboards_now_page(tmp_path):
    # the decided design of the Architecture tab, Oct 7, 2026, D-E7: the typed note "H84 decision, September 28" was
    # printed for every run whose folder name starts with H84_, and its link opened a status page last rewritten on
    # Oct 3; the header now links the development dashboard's Now page, computed on every request
    run = tmp_path / "H84_fresh_drivers_20260918_seg41c_1629"
    run.mkdir()
    (tmp_path / "training_continuation.json").write_text(json.dumps({"active_run": str(run)}), encoding="utf-8")
    page = CS.current_state_html(CS.current_state(run, {}, {}, tmp_path, []))
    assert '<a href="http://127.0.0.1:8766/#now">' in page
    for gone in ("H84 decision", "September 28", "H84_CURRENT_STATUS_AND_BLOCKERS"):
        assert gone not in page, gone


def test_a_repair_run_is_named_as_one(tmp_path):
    # the relaunch review of Oct 5, 2026, finding F10: the repair run trains under a pause that names it
    run = tmp_path / "H84_repair"
    run.mkdir()
    (tmp_path / "training_continuation.json").write_text(json.dumps({"active_run": str(run)}), encoding="utf-8")
    (tmp_path / "supervisor_control.json").write_text(json.dumps({"cmd": "pause", "repair_run": str(run)}),
                                                      encoding="utf-8")
    state = CS.current_state(run, {}, {}, tmp_path, [])
    assert state["restart_paused"] and state["repair_run"]
    assert "Repair run: bounded, with no automatic restart when it ends" in CS.current_state_html(state)
    (tmp_path / "supervisor_control.json").write_text(json.dumps({"cmd": "pause"}), encoding="utf-8")
    state = CS.current_state(run, {}, {}, tmp_path, [])
    assert not state["repair_run"] and "Production restart is paused" in CS.current_state_html(state)
