"""Semantic compatibility checks, independent of file fingerprints."""


def semantic_contract(value):
    if isinstance(value, dict):
        return {key: semantic_contract(item) for key, item in value.items()
                if not any(tag in key.lower() for tag in ('sha256', 'checksum', 'fingerprint'))
                and key not in ('decision_document', 'source_commit', 'dinov3_source_commit')}
    if isinstance(value, (list, tuple)):
        return [semantic_contract(item) for item in value]
    return value
