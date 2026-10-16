"""Save a reproducible, checkpoint-independent list of RoboCasa scene seeds."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or args.episodes < 1:
        parser.error("Require uint32 seed and positive episodes")
    rng = np.random.default_rng(args.seed)
    manifest = dict(protocol="robocasa_scene_seed_list_v1", generator="numpy.random.default_rng",
                    bit_generator=type(rng.bit_generator).__name__, generator_seed=args.seed,
                    numpy_version=np.__version__,
                    episode_seeds=rng.integers(0, 2**32, size=args.episodes).tolist())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
