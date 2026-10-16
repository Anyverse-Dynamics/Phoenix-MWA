# Source release

Run `python tools/audit_public_source.py` before packaging. Then run
`python tools/package_source.py --output dist/phoenix-mwa.tar.gz`.

The packager uses an explicit source allowlist, excludes runtime outputs and
rejects credentials, private deployment paths, symlinks and unreviewed binary
files. It emits a source archive manifest. This release-time scan is separate
from training/evaluation and does not gate model execution.

Training/evaluation do not check file digests, source commit hashes or historical
transfer receipts. File existence, shape/dtype, finite values, action convention,
normalization structure and task completeness remain checked. Same-size file
corruption is not detected by these structural checks. Supply matching resources
and run hardware smoke before relying on full runs.

Serialized format identifiers use the Phoenix namespace. Checkpoints, cache
manifests and resume metadata must use the matching schema version; changing the
namespace does not alter tensor names, shapes or numerical values. Original license copyright
holders are retained, even where they differ from the current project branding.
No model weights, datasets, experiment logs or private machine configurations are
included in the source archive.
