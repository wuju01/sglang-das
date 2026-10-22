# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests; all mutation targets are disposable installation fixtures."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
MODULE_SPEC = importlib.util.spec_from_file_location("config_patch", HERE / "config_patch.py")
tool = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(tool)
CONFIRM = ["--yes", "--services-stopped", "--confirm-gfx936-80cu"]


class TestConfigPatch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config_smooth_gemm.py"
        self.backup = self.path.with_name(self.path.name + tool.BACKUP_SUFFIX)
        self.lock = self.path.with_name(self.path.name + tool.LOCK_SUFFIX)
        self.spec = copy.deepcopy(tool.load_spec())
        self.original = (
            "class Config:\n    def tables():\n"
            "        w8a8_smooth_quant_config_dict_80 = {\n"
            "            'M1N256K256': (1, 16, 256, 1, False),\n"
            "        }\n        other_table = {'untouched': True}\n"
        ).encode()
        additions = "".join(
            f"            {key!r}: {tuple(value)!r},\n"
            for key, value in self.spec["entries"].items()
        ).encode()
        self.expected = self.original.replace(b"        }\n", additions + b"        }\n")
        self.spec.update(original_sha256=tool.sha(self.original), patched_sha256=tool.sha(self.expected))
        self.path.write_bytes(self.original)
        dist = SimpleNamespace(version=self.spec["lightop_version"], locate_file=lambda _: self.path)
        self.dist = dist
        self.addCleanup(patch.stopall)
        patch.object(tool, "load_spec", return_value=self.spec).start()
        patch.object(tool.metadata, "distribution", return_value=dist).start()

    def run_cli(self, action, flags=CONFIRM):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = tool.main([action, *flags])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_exact_entry_scope(self):
        entries = self.spec["entries"]
        observed = {(1280,5120):(5,10,15), (1792,5120):(6,12),
                    (4096,1280):(6,12), (4608,5120):(5,6,10,12,15),
                    (5120,2304):(5,6,10,12,15), (25600,6144):(6,12),
                    (32768,1280):(5,6,10,12,15)}
        keys = {f"M{m}N512K5120" for m in range(1,17)}
        keys.update(f"M{m}N{n}K{k}" for (n,k), ms in observed.items() for m in ms)
        self.assertEqual(set(entries), keys)
        self.assertEqual(len(entries), 40)

    def test_only_expected_additions(self):
        result = tool.patched_bytes(self.original, self.spec)
        self.assertEqual(result, self.expected)
        self.assertEqual(result.count(b"other_table = {'untouched': True}"), 1)
        self.assertEqual(result.count(b"'M1N256K256': (1, 16, 256, 1, False)"), 1)

    def test_check_is_read_only(self):
        self.assertEqual(self.run_cli("check", [])[0], 0)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_confirmation_required(self):
        for flags in ([], CONFIRM[:1], CONFIRM[:2]):
            with self.subTest(flags=flags), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    tool.main(["apply", *flags])
                self.assertEqual(error.exception.code, 2)
        self.assertFalse(self.backup.exists())

    def test_apply_idempotency_and_restore(self):
        self.assertEqual(self.run_cli("apply")[0], 0)
        self.assertEqual(self.path.read_bytes(), self.expected)
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertEqual(json.loads(self.run_cli("apply")[1])["state"], "already_applied")
        self.assertEqual(self.run_cli("restore")[0], 0)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertEqual(json.loads(self.run_cli("restore")[1])["state"], "already_original")

    def test_unknown_file_rejected(self):
        self.path.write_bytes(self.original + b"# user edit\n")
        self.assertEqual(self.run_cli("apply")[0], 2)
        self.assertEqual(self.path.read_bytes(), self.original + b"# user edit\n")
        self.assertFalse(self.backup.exists())

    def test_wrong_version_rejected(self):
        self.dist.version = "0.0.0"
        self.assertEqual(self.run_cli("apply")[0], 2)
        self.assertFalse(self.backup.exists())

    def test_bad_payload_does_not_write(self):
        self.spec["entries"]["M1N512K5120"][0] = 8
        self.assertEqual(self.run_cli("apply")[0], 2)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse(self.backup.exists())

    def test_conflicting_backup_preserved(self):
        self.backup.write_bytes(b"user backup")
        self.assertEqual(self.run_cli("apply")[0], 2)
        self.assertEqual(self.backup.read_bytes(), b"user backup")
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_restore_refuses_later_edits(self):
        self.assertEqual(self.run_cli("apply")[0], 0)
        edited = self.expected + b"# subsequent edit\n"
        self.path.write_bytes(edited)
        self.assertEqual(self.run_cli("restore")[0], 2)
        self.assertEqual(self.path.read_bytes(), edited)

    def test_restore_refuses_corrupt_backup(self):
        self.assertEqual(self.run_cli("apply")[0], 0)
        self.backup.write_bytes(b"bad backup")
        self.assertEqual(self.run_cli("restore")[0], 2)
        self.assertEqual(self.path.read_bytes(), self.expected)

    def test_existing_patch_without_backup_is_not_restorable(self):
        self.path.write_bytes(self.expected)
        self.assertEqual(self.run_cli("check", [])[0], 0)
        self.assertEqual(self.run_cli("apply")[0], 0)
        self.assertFalse(self.backup.exists())
        self.assertEqual(self.run_cli("restore")[0], 2)

    def test_existing_lock_is_not_removed(self):
        self.lock.write_bytes(b"another installer")
        self.assertEqual(self.run_cli("apply")[0], 2)
        self.assertEqual(self.lock.read_bytes(), b"another installer")
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_replace_failure_preserves_original_and_backup(self):
        with patch.object(tool.os, "replace", side_effect=OSError("injected failure")):
            self.assertEqual(self.run_cli("apply")[0], 2)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.backup.read_bytes(), self.original)
        self.assertEqual(set(self.path.parent.iterdir()), {self.path, self.backup})


if __name__ == "__main__":
    unittest.main()
