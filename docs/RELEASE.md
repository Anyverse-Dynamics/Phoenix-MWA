# Source release

Run `python tools/package_source.py --output dist/Phoenix-Encoder.tar.gz`.
The packager audits every selected file before creating the archive; local
resource packages and experiment directories are not scanned or included.

The packager uses an explicit source allowlist, excludes development regression
tests, migration-only tools and runtime outputs, and
rejects credentials, private deployment paths, symlinks and unreviewed binary
files. It emits a source archive manifest. This release-time scan is separate
from training/evaluation and does not gate model execution.

Training/evaluation do not check file digests, source commit hashes or historical
transfer receipts. File existence, shape/dtype, finite values, action convention,
normalization structure and task completeness remain checked. Same-size file
corruption is not detected by these structural checks. Supply matching resources
and run hardware smoke before relying on full runs.

Original license copyright holders are retained.
No model weights, datasets, experiment logs or private machine configurations are
included in the source archive.

Keep the documented bounded smoke entrypoints and runtime compatibility checks
in the public source. They validate the user's installation and resources and
are not part of the separate development regression suite.
