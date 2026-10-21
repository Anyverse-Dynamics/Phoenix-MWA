"""Copy existing joint14 mean/std into Phoenix's shared train/inference schema."""
import argparse
import json
from pathlib import Path


def convert_normalization(source, output, *, gripper_unit):
    from .model import Normalizer
    from .trainer import file_sha256
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    content = json.loads(source.read_text())
    known_unit = content.get("gripper_unit")
    if known_unit is not None and known_unit != gripper_unit:
        raise ValueError("Unit conversion is not supported; select matching source units")
    # Preserve numeric values exactly. No statistics are fitted or rescaled.
    result = dict(schema="phoenix.joint14.normalization.v1", gripper_unit=gripper_unit,
                  state=content["state"], action=content["action"],
                  source_sha256=file_sha256(source))
    Normalizer(result, gripper_unit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(path=str(output), sha256=file_sha256(output), source_sha256=result["source_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gripper-unit", required=True, choices=["m", "rad"])
    args = parser.parse_args()
    print(json.dumps(convert_normalization(args.normalization, args.output, gripper_unit=args.gripper_unit), indent=2))


if __name__ == "__main__":
    main()
