#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Explicit, offline LightOp configuration patch; never imported by serving."""

import argparse
import ast
from contextlib import contextmanager
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import stat
import sys
import tempfile

BACKUP_SUFFIX = ".sglang-small-m40.bak"
LOCK_SUFFIX = ".sglang-small-m40.lock"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def load_spec():
    path = Path(__file__).with_name("configs.json")
    return json.loads(path.read_text(), object_pairs_hook=unique_object)


def patched_bytes(original, spec):
    if sha(original) != spec["original_sha256"]:
        raise ValueError("Unknown original configuration; refusing to patch")
    source = original.decode("utf-8")
    nodes = [
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
        and any(isinstance(t, ast.Name) and t.id == spec["table"] for t in node.targets)
    ]
    if len(nodes) != 1 or len(spec["entries"]) != 40:
        raise ValueError("Expected one target table and exactly 40 additions")
    node = nodes[0]
    existing = ast.literal_eval(node.value)
    if set(existing).intersection(spec["entries"]):
        raise ValueError("Refusing to replace existing configuration entries")
    lines = source.splitlines(keepends=True)
    closing = node.value.end_lineno - 1
    if lines[closing].strip() != "}":
        raise ValueError("Unexpected table layout")
    indent = " " * (node.col_offset + 4)
    additions = "".join(
        f"{indent}{key!r}: {tuple(value)!r},\n"
        for key, value in spec["entries"].items()
    )
    changed = ("".join(lines[:closing]) + additions + "".join(lines[closing:])).encode()
    if sha(changed) != spec["patched_sha256"]:
        raise ValueError("Patch does not reproduce the validated configuration hash")
    compile(changed, "config_smooth_gemm.py", "exec")
    return changed


def regular_bytes(path):
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"Expected a regular file, not a symlink: {path}")
    return path.read_bytes()


@contextmanager
def mutation_lock(path):
    lock = path.with_name(path.name + LOCK_SUFFIX)
    # Exclusive creation refuses existing files and symlinks. Never steal a lock.
    with lock.open("xb") as handle:
        handle.write(f"pid={os.getpid()}\n".encode())
    try:
        yield
    finally:
        lock.unlink()


def atomic_replace(path, before, after):
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(after)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        if regular_bytes(path) != before:
            raise ValueError("Configuration changed concurrently; refusing to overwrite")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def change(path, spec, action):
    with mutation_lock(path):
        current = regular_bytes(path)
        digest = sha(current)
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if action == "apply":
            if digest == spec["patched_sha256"]:
                return "already_applied"
            changed = patched_bytes(current, spec)
            if backup.exists() or backup.is_symlink():
                if sha(regular_bytes(backup)) != spec["original_sha256"]:
                    raise ValueError("Existing backup differs; it will not be overwritten")
            else:
                with backup.open("xb") as handle:
                    handle.write(current)
                    handle.flush()
                    os.fsync(handle.fileno())
            atomic_replace(path, current, changed)
            return "applied"
        if digest == spec["original_sha256"]:
            return "already_original"
        if digest != spec["patched_sha256"]:
            raise ValueError("Configuration changed since patching; refusing to restore")
        original = regular_bytes(backup)
        if sha(original) != spec["original_sha256"]:
            raise ValueError("Backup hash mismatch; refusing to restore")
        atomic_replace(path, current, original)
        return "restored"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply", "restore"))
    parser.add_argument("--yes", action="store_true", help="Authorize the requested mutation")
    parser.add_argument("--services-stopped", action="store_true", help="Confirm all users of this LightOp installation are stopped")
    parser.add_argument("--confirm-gfx936-80cu", action="store_true", help="Confirm the validated hardware target; not an automatic GPU check")
    args = parser.parse_args(argv)
    if args.action != "check" and not (args.yes and args.services_stopped and args.confirm_gfx936_80cu):
        parser.error("apply/restore require --yes --services-stopped --confirm-gfx936-80cu")
    try:
        spec = load_spec()
        dist = metadata.distribution("lightop")  # No torch/lightop import or GPU use.
        if dist.version != spec["lightop_version"]:
            raise ValueError(f"Unsupported LightOp version {dist.version}; expected {spec['lightop_version']}")
        path = Path(dist.locate_file("lightop/config_smooth_gemm.py")).absolute()
        digest = sha(regular_bytes(path))
        if args.action == "check":
            states = {spec["original_sha256"]: "original", spec["patched_sha256"]: "patched"}
            if digest not in states:
                raise ValueError("Unknown configuration hash; manual review required")
            state = states[digest]
        else:
            state = change(path, spec, args.action)
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        backup_valid = (not backup.is_symlink() and backup.is_file()
                        and sha(backup.read_bytes()) == spec["original_sha256"])
        print(json.dumps({"state": state, "path": str(path), "target": spec["target"],
                          "sha256": sha(regular_bytes(path)), "backup_valid": backup_valid,
                          "restart_required": args.action != "check"}, indent=2))
        return 0
    except (OSError, ValueError, metadata.PackageNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
