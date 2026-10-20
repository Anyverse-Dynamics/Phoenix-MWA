"""Thin, immutable Robot36 index referencing the frozen raw-BI package."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from wam.resources import resource_path

from wam.datasets.local_robot_contracts import sha256
from wam.datasets.local_robot_package import LocalRobotPackage, read_json, unpack
from wam.datasets.local_robot_raw_bi_package import LocalRobotRawBIPackage
from wam.datasets.local_robot36_contracts import SCHEMA


class _CachedRawBIBase(LocalRobotRawBIPackage):
    _root = lru_cache(maxsize=32)(LocalRobotPackage._root)


class Robot36Package(LocalRobotPackage):
    def __init__(self, root, *, audit=False, verify_index=True):
        self.root = resource_path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        from wam.runtime_contracts import recorded_manifest_id, require_file
        self.fingerprint = recorded_manifest_id(self.root)
        if self.manifest["schema"] != SCHEMA:
            raise ValueError("Robot36 manifest/schema drift")
        if not audit and self.manifest["purpose"] != "training_candidate":
            raise ValueError("bounded Robot36 audit package cannot enter formal training")
        require_file(self.root / "episodes.sqlite")
        base = self.manifest["base_package"]
        self.base = _CachedRawBIBase(base["root"], audit=True, verify_index=verify_index)
        if self.base.fingerprint != base["manifest_sha256"]:
            raise ValueError("Robot36 frozen base changed")
        self._db, self._pid = None, None

    @lru_cache(maxsize=64)
    def _bindings(self, root_id):
        return [unpack(row[0]) for row in self.db.execute(
            "SELECT payload FROM bindings WHERE root_id=? ORDER BY task_index", (root_id,))]

    @lru_cache(maxsize=32)
    def _root(self, root_id):
        return unpack(self.db.execute("SELECT payload FROM roots WHERE id=?", (root_id,)).fetchone()[0])

    @lru_cache(maxsize=32)
    def episode(self, episode_id):
        row = self.db.execute("SELECT profile,split,unit,ranges,record,root_id,base_episode_id FROM episodes WHERE id=?",
                              (int(episode_id),)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        profile, split, unit, ranges, record, root_id, base_id = row
        if base_id is not None:
            original = self.base.episode(int(base_id))
            record, root = copy.deepcopy(original["record"]), original["root"]
            if original["unit"] != unit or original["split"] != split or original["ranges"] != unpack(ranges):
                raise ValueError("Robot36 must preserve inherited split/unit/starts")
        else:
            record = unpack(record)
            root = self._root(root_id)
        if record.get("t5_episode_binding") is None:
            record["t5_task_bindings"] = self._bindings(root_id)
        return dict(id=int(episode_id), profile=profile, split=split, unit=unit,
                    ranges=unpack(ranges), record=record, root=root)

    def close(self):
        super().close()
        self.base.close()
