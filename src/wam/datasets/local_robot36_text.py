"""Explicit provider routing, including generic-pick episode-language3 caches."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from wam.resources import resource_path, repository_root

import pyarrow.parquet as pq

from wam.datasets.local_robot_contracts import digest, sha256
from wam.datasets.local_robot_package import read_json
from wam.datasets.local_robot_text import LocalRobotTextCache, _array

CACHE_ROOT = repository_root() / "resources/text_cache"
REGISTRIES = (
    "registry_robot11_language_aug_v3_v1.json",
    "registry_galaxea_english_delta_v2.json",
    "registry_openneo5_language_aug_v3_v1.json",
    "registry_new12_language_aug_20261001_v1.json",
    "registry_collections5_language_20261003_v1.json",
    "registry_interndata_generic_pick_20260927_language3_v1.json",
)
PROVIDER_ALIASES = {
    "task_generalization": "interndata_task_generalization",
    "generic_pick": "interndata_generic_pick_20260927",
    "arx456_xt_motion_dedup_20260916": "arx456_xt_motion_dedup",
    "arx_machine_named_20260916": "arx_machine_named",
    "arx_incremental920_20260917": "arx_incremental920",
}
ALLOWED_PROVIDERS = frozenset({"intern_data_a1", "agibot_gripper", "robocoin", "robomind2",
    "cosmos3_droid", "galaxea_upperbody", "abc130k", "openneo_aloha", "openneo_arx5",
    "openneo_arx5_single", "openneo_flexiv", "openneo_ur", "consolidated_pickup",
    "random_pick_and_place", "object_to_container", "relational_pick_and_place",
    "stack_blocks", "cover_blocks", "arrange_largest_number", "realsource", "rwrl_g2a",
    "rwrl_r1lite", "rwrl_piperx", "habit", *PROVIDER_ALIASES.values()})


class EpisodeLanguage3Cache(LocalRobotTextCache):
    def __init__(self, registry_path, source, expected_hash):
        path = resource_path(registry_path)
        registry = read_json(path)
        if (registry.get("schema_version") != "source_routed_t5_v1" or
                registry.get("context_len") != 128 or registry.get("dtype") != "bfloat16" or
                registry.get("encoder_id") != "wan22ti2v5b"):
            raise ValueError("unsupported episode T5 registry")
        provider = registry["providers"][source]
        self.root = resource_path(provider["cache_root"])
        self.manifest = read_json(self.root / "manifest.json")
        if (self.manifest["context_len"], self.manifest["context_dim"], self.manifest["dtype"]) != (128, 4096, "bfloat16"):
            raise ValueError("episode T5 geometry mismatch")
        provenance = self.manifest["provenance"]
        if provenance["original_in_sampling"] is not False or provenance["route_kind"] != "physical_dataset_uid_plus_episode_index":
            raise ValueError("generic-pick must use only added episode descriptions")
        bindings = self.root / "episode_bindings.parquet"
        rows = pq.read_table(bindings).to_pylist()
        self.routes = {(r["physical_dataset_uid"], int(r["episode_index"])): r for r in rows}
        if len(self.routes) != len(rows):
            raise ValueError("duplicate physical episode language binding")
        self.catalog = read_json(self.root / "catalog/manifest.json")
        if self.catalog["prompt_set_fingerprint"] != self.manifest["catalog_prompt_set_fingerprint"]:
            raise ValueError("episode T5 catalog drift")
        if any(self.manifest[k] != registry[k] for k in ("encoder_id", "tokenizer_id")):
            raise ValueError("episode T5 encoder mismatch")
        catroot = self.root / "catalog"
        src = self.catalog["sources"][source]
        if src["index_fingerprint"] != registry["sources"][source]["index_fingerprint"]:
            raise ValueError("episode catalog source fingerprint mismatch")
        self.offsets, self.ids = _array(catroot, src["episode_offsets"]), _array(catroot, src["prompt_ids"])
        self.prompt_offsets = _array(catroot, self.catalog["prompt_offsets"])
        desc = self.catalog["prompt_blob"]
        self.prompt_path = catroot / desc["file"]
        self.locators = {k: _array(self.root, d) for k, d in self.manifest["locators"].items()}
        self.parts, self.chunks = {}, OrderedDict()

    def resolve(self, binding, *, seed, identity, augment):
        row = self.routes[(binding["physical_dataset_uid"], int(binding["episode_index"]))]
        if (int(row["episode_row"]) != int(binding["route_row"]) or
                row["trajectory_sha256"] != binding["trajectory_sha256"] or
                row["effective_task_sha256"] != binding["effective_task_sha256"]):
            raise ValueError("episode T5 identity mismatch")
        begin, end = map(int, self.offsets[int(row["episode_row"]):int(row["episode_row"])+2])
        if end - begin != 3 or binding["variant_count"] != 3:
            raise ValueError("generic-pick requires exactly three added descriptions")
        for i, prompt in enumerate(row["instructions"]):
            if self._prompt(int(self.ids[begin+i])) != prompt:
                raise ValueError("generic-pick prompt order drift")
        variant = int(digest([seed, "episode_language3", identity]), 16) % 3 if augment else 0
        return self.resolve_prompt(int(self.ids[begin + variant]), variant)


class Robot36TextCatalog:
    """Build immutable bindings; no fuzzy task-name or basename matching."""
    def __init__(self, cache_root=CACHE_ROOT):
        self.caches, self.identities, self.by_path, self.by_uid = {}, {}, {}, {}
        for name in REGISTRIES:
            path = Path(cache_root) / name
            fingerprint, registry = sha256(path), read_json(path)
            self.identities[str(path)] = fingerprint
            for provider in sorted(set(registry["providers"]) & ALLOWED_PROVIDERS):
                episode = registry["schema_version"] == "source_routed_t5_v1"
                cache = (EpisodeLanguage3Cache if episode else LocalRobotTextCache)(path, provider, fingerprint)
                key = str(path), provider
                self.caches[key] = cache
                prompt_blob = cache.prompt_path.read_bytes()
                for identity, route in cache.routes.items():
                    binding = dict(registry=str(path), registry_sha256=fingerprint, provider_id=provider,
                        route_kind="episode" if episode else "task", physical_dataset_uid=identity[0],
                        route_row=int(route["episode_row"] if episode else route["route_row"]),
                        variant_count=3 if episode else int(route["variant_count"]))
                    if episode:
                        binding.update(episode_index=identity[1], trajectory_sha256=route["trajectory_sha256"],
                            effective_task_sha256=route["effective_task_sha256"])
                    else:
                        first = int(cache.ids[int(cache.offsets[binding["route_row"]])])
                        begin, end = map(int, cache.prompt_offsets[first:first+2])
                        binding.update(task_index=identity[1], source_fingerprint=route["source_fingerprint"],
                                       original_text=prompt_blob[begin:end].decode("utf-8"))
                    self.by_uid.setdefault((provider, *identity), []).append(binding)
                    self.by_path.setdefault((provider, str(Path(route["dataset_path_hint"])), identity[1]), []).append(binding)

    @staticmethod
    def choose(matches):
        if len(matches) == 1:
            return dict(matches[0])
        # The approved English delta supersedes unusable original Galaxea text.
        delta = [m for m in matches if Path(m["registry"]).name == REGISTRIES[1]]
        if len(delta) == 1 and all(m["provider_id"] == "galaxea_upperbody" for m in matches):
            return dict(delta[0])
        raise ValueError(f"no unique physical T5 route: {len(matches)} candidates")

    def bind(self, source, root, index, *, old_binding=None):
        provider = PROVIDER_ALIASES.get(source, source)
        if provider not in ALLOWED_PROVIDERS:
            raise ValueError("unselected T5 provider")
        matches = self.by_path.get((provider, str(Path(root)), int(index)), [])
        if not matches and old_binding:
            matches = self.by_uid.get((provider, old_binding["physical_dataset_uid"], int(index)), [])
        try:
            binding = self.choose(matches)
        except ValueError as exc:
            raise ValueError(f'{source}: root={root} index={index}: {exc}') from exc
        binding["bound_source_root"] = str(root)
        return binding
