# Data Directory

This directory is used locally for dataset storage and preprocessing.

It is intentionally excluded from the GitHub upload package because it may contain:

- raw TAWOS SQL dumps
- processed TSV exports
- local MySQL sandbox files
- other large intermediate data files

Recommended public-repo policy:

- do not commit raw dataset files
- do not commit processed TSV exports if they are large or license-restricted
- keep only scripts and documentation needed to reproduce the data pipeline
