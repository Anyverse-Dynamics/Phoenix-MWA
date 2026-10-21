"""Run one explicit data stage; stages never implicitly launch training."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

STAGES = {"convert": "conversion", "prepare": "preparation", "cache-text": "text_cache",
          "cache-visual": "visual_cache", "finalize-visual": "visual_cache"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[2] / "configs/phoenix_data.yaml"
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", default=str(default))
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = OmegaConf.merge(OmegaConf.load(default), OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    if config.schema != "phoenix.data_recipe.v1":
        raise ValueError("Expected Phoenix data recipe")
    # Only the selected stage's external resources are mandatory.
    options = OmegaConf.to_container(config[STAGES[args.stage]], resolve=True, throw_on_missing=True)
    if args.stage == "convert":
        from .convert import convert
        result = convert(options)
    elif args.stage == "prepare":
        from .prepare import prepare
        result = prepare(options)
    elif args.stage == "cache-text":
        from .text_cache import build_text_cache
        result = build_text_cache(options)
    elif args.stage == "cache-visual":
        from .visual_cache import build_visual_cache
        result = build_visual_cache(options)
    else:
        from .encoders import visual_resources, visual_identity
        from .visual_cache import build_contract, finalize
        model, vae = visual_resources(options["model_definition"])
        contract = build_contract(options["manifest"], visual_identity(model, vae), options["chunk_rows"])
        result = finalize(options["output"], options["manifest"], expected_contract=contract)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
