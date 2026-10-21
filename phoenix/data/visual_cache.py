"""H32 frozen Joint96 windows in resumable, checksummed BF16 shards."""
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from phoenix.dataset import PhoenixDataset
from .common import atomic_json, check_sources, digest, directory_guard, read_json, sha256

SHAPE = (96, 3, 16, 48)
VIDEO_SHAPE = (3, 3, 9, 256, 256)


def jobs(manifest, chunk_rows):
    for ordinal, episode in enumerate(manifest["episodes"]):
        for start in range(0, episode["length"] - 4, chunk_rows):
            yield dict(key=f"e{ordinal:06d}_w{start:09d}", episode=ordinal, start=start,
                       count=min(chunk_rows, episode["length"] - 4 - start))


def preprocessing_identity():
    import inspect
    return dict(raw_window_frames=33, video_offsets=list(range(0, 33, 4)),
                video_shape=list(VIDEO_SHAPE), feature_shape=list(SHAPE), tail="repeat_last",
                decoder_sha256=hashlib.sha256(inspect.getsource(PhoenixDataset.decode_video).encode()).hexdigest(),
                writer_sha256=sha256(__file__), encoder_batch_size=1, posterior="mean",
                dtype="bfloat16", storage="uint16")


def build_contract(manifest_path, identity, chunk_rows):
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    manifest = read_json(manifest_path)
    if manifest["schema"] != "phoenix.windows.v1" or manifest["fps"] != 30:
        raise ValueError("Expected a frozen Phoenix H32 manifest")
    return dict(schema="phoenix.visual_cache.h32.v1", manifest_sha256=sha256(manifest_path),
                encoder=identity, preprocessing=preprocessing_identity(), chunk_rows=chunk_rows,
                windows=sum(e["length"] - 4 for e in manifest["episodes"]))


def mapping(manifest, job):
    return digest(dict(job=job, episode=manifest["episodes"][job["episode"]]))


def validate_shard(directory, contract, manifest, job, *, full_hash=False):
    directory = Path(directory)
    record = read_json(directory / "COMPLETE.json")
    if (record["contract_sha256"] != digest(contract) or record["mapping_sha256"] != mapping(manifest, job)
            or record["job"] != job or len(record["row_sha256"]) != job["count"]):
        raise ValueError("Visual shard contract or mapping differs")
    data = directory / "features.bf16"
    if data.stat().st_size != job["count"] * int(np.prod(SHAPE)) * 2:
        raise ValueError("Truncated visual shard")
    if full_hash and sha256(data) != record["data_sha256"]:
        raise ValueError("Visual shard hash differs")
    return record


def write_shard(output, contract, manifest, job, encode):
    directory = Path(output) / "shards" / job["key"]
    directory.mkdir(parents=True, exist_ok=True)
    with directory_guard(directory / "writer.owner"):
        if (directory / "COMPLETE.json").exists():
            return validate_shard(directory, contract, manifest, job, full_hash=True)
        partial = directory / "features.partial"
        shape = (job["count"], *SHAPE)
        values = np.memmap(partial, dtype=np.uint16, mode="w+", shape=shape)
        hashes = []
        episode = manifest["episodes"][job["episode"]]
        try:
            for row in range(job["count"]):
                video = PhoenixDataset.decode_video(episode, job["start"] + row)
                with torch.no_grad():
                    latent = encode(video).detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                if tuple(latent.shape) != SHAPE or not torch.isfinite(latent).all():
                    raise ValueError("Encoder must return finite BF16-compatible Joint96 H32 latents")
                raw = latent.view(torch.uint16).numpy()
                values[row] = raw
                hashes.append(hashlib.sha256(raw.tobytes()).hexdigest())
            values.flush()
        finally:
            del values
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, directory / "features.bf16")
        record = dict(schema="phoenix.visual_shard.v1", contract_sha256=digest(contract),
                      mapping_sha256=mapping(manifest, job), job=job, row_sha256=hashes,
                      data_sha256=sha256(directory / "features.bf16"))
        atomic_json(directory / "COMPLETE.json", record)
        return record


def finalize(output, manifest_path, *, expected_contract=None):
    output = Path(output)
    with directory_guard(output / "finalize.owner"):
        contract, manifest = read_json(output / "contract.json"), read_json(manifest_path)
        if expected_contract is not None and contract != expected_contract:
            raise ValueError("Visual encoder/configuration changed before finalization")
        if contract["manifest_sha256"] != sha256(manifest_path) or contract["preprocessing"] != preprocessing_identity():
            raise ValueError("Manifest/preprocessing changed before finalization")
        check_sources(manifest)
        records = {}
        for job in jobs(manifest, contract["chunk_rows"]):
            record = validate_shard(output / "shards" / job["key"], contract, manifest, job, full_hash=True)
            records[job["key"]] = digest(record)
        check_sources(manifest)
        index = dict(schema="phoenix.visual_cache.h32.v1", contract=contract, shards=records)
        if (output / "index.json").exists() and read_json(output / "index.json") != index:
            raise ValueError("Completed visual cache changed")
        atomic_json(output / "index.json", index)
        atomic_json(output / "READY.json", dict(index_sha256=sha256(output / "index.json"), windows=contract["windows"]))
    return dict(status="VISUAL_READY", index=str(output / "index.json"), windows=contract["windows"])


def build_visual_cache(config):
    from .encoders import visual_resources, visual_identity, load_visual
    rank, world = config["rank"], config["world_size"]
    if type(rank) is not int or type(world) is not int or not 0 <= rank < world:
        raise ValueError("Expected 0 <= rank < world_size")
    model_config, vae_path = visual_resources(config["model_definition"])
    contract = build_contract(config["manifest"], visual_identity(model_config, vae_path), config["chunk_rows"])
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    # Contract creation is idempotent: every rank must use identical content.
    # mkdir is nonblocking, so initialize with rank 0 before starting other ranks.
    if not (output / "contract.json").exists():
        with directory_guard(output / "contract.owner"):
            if (output / "contract.json").exists():
                if read_json(output / "contract.json") != contract:
                    raise ValueError("Visual cache contract differs")
            else:
                atomic_json(output / "contract.json", contract)
    if read_json(output / "contract.json") != contract:
        raise ValueError("Visual cache data/encoder/preprocessing differs")
    manifest = read_json(config["manifest"])
    check_sources(manifest)
    if config.get("initialize_only", False):
        return dict(status="VISUAL_INITIALIZED", windows=contract["windows"])
    vae, encoder = None, None
    def encode(video):
        nonlocal vae, encoder
        if encoder is None:
            vae, encoder = load_visual(model_config, vae_path, config["device"])
        value = video.unsqueeze(0).to(config["device"], dtype=torch.bfloat16)
        return encoder.encode(value, camera_is_pad=torch.zeros((1, 3), device=value.device, dtype=torch.bool))[0]
    written = 0
    for ordinal, job in enumerate(jobs(manifest, contract["chunk_rows"])):
        if ordinal % world == rank:
            write_shard(output, contract, manifest, job, encode)
            written += 1
    check_sources(manifest)
    if world == 1:
        return finalize(output, config["manifest"], expected_contract=contract)
    return dict(status="VISUAL_WORKER_COMPLETE", rank=rank, shards=written, ready=False)


class VisualCache:
    def __init__(self, index_path, manifest_path):
        self.root = Path(index_path).resolve().parent
        ready, self.index = read_json(self.root / "READY.json"), read_json(index_path)
        if ready["index_sha256"] != sha256(index_path) or self.index["schema"] != "phoenix.visual_cache.h32.v1":
            raise ValueError("Visual cache is incomplete or its index changed")
        self.contract = self.index["contract"]
        self.manifest = read_json(manifest_path)
        if (self.contract["manifest_sha256"] != sha256(manifest_path)
                or self.contract["preprocessing"] != preprocessing_identity()
                or ready["windows"] != self.contract["windows"]):
            raise ValueError("Visual cache belongs to different windows/preprocessing")
        check_sources(self.manifest)
        expected = {job["key"] for job in jobs(self.manifest, self.contract["chunk_rows"])}
        if set(self.index["shards"]) != expected:
            raise ValueError("Visual cache does not cover every episode window")
        self.ordinals = {(e["root_id"], e["episode_index"]): i for i, e in enumerate(self.manifest["episodes"])}
        self.open = OrderedDict()

    def validate_model(self, model, model_definition):
        from .encoders import model_settings, visual_identity
        identity = visual_identity(model_settings(model_definition), model.model_paths["vae"])
        if (self.contract["encoder"] != identity
                or model.joint_visual_encoder.current_conditioning != identity["current_conditioning"]):
            raise ValueError("Visual cache and training model use different frozen encoders/conditioning")

    def get(self, episode, start):
        ordinal = self.ordinals[(episode["root_id"], episode["episode_index"])]
        chunk_rows = self.contract["chunk_rows"]
        block = start // chunk_rows * chunk_rows
        job = dict(key=f"e{ordinal:06d}_w{block:09d}", episode=ordinal, start=block,
                   count=min(chunk_rows, episode["length"] - 4 - block))
        key = job["key"]
        if key not in self.open:
            record = validate_shard(self.root / "shards" / key, self.contract, self.manifest, job)
            if digest(record) != self.index["shards"][key]:
                raise ValueError("Visual shard receipt changed")
            data = np.memmap(self.root / "shards" / key / "features.bf16", mode="r", dtype=np.uint16,
                             shape=(job["count"], *SHAPE))
            if len(self.open) >= 4:
                self.open.popitem(last=False)
            self.open[key] = record, data
        record, data = self.open.pop(key)
        self.open[key] = record, data
        row = np.array(data[start - block], copy=True)
        if hashlib.sha256(row.tobytes()).hexdigest() != record["row_sha256"][start - block]:
            raise ValueError("Visual cache row is corrupt")
        latent = torch.from_numpy(row).view(torch.bfloat16)
        if not torch.isfinite(latent).all():
            raise ValueError("Nonfinite cached visual latents")
        return latent
