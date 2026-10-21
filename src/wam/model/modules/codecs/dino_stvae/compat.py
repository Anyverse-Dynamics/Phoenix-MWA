"""Read archived encoder options without loading external framework code."""


def check_legacy_encoder_options(options):
    # Archived policy configs include this obsolete source selector. The encoder
    # now lives in the installed Phoenix package, regardless of its old value.
    unknown = set(options) - {"starvla_source_path"}
    if unknown:
        raise TypeError(f"Unexpected encoder options: {sorted(unknown)}")
