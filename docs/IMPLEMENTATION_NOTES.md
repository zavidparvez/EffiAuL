# Implementation notes

The source file is extracted from the **second populated cell** of the supplied notebook. The first populated cell was a near-identical single-seed variant; the second enables five training seeds. Duplicated notebook code was not executed twice.

The source is deliberately kept close to the uploaded implementation so that methodological details can be traced. The only functional packaging changes are environment-variable overrides for the dataset path, output directory, seeds, forget count, forget selection, and bootstrap iterations. The underlying experimental algorithms have not been replaced with simplified demonstrations.

## Important limitations

- The supplied implementation requires exactly four classes. It is not the paper's separate primary binary benchmark.
- Source data, checkpoints, and results were not supplied; published numerical results cannot be verified from the notebook alone.
- A fixed file-level stratified split does not by itself guarantee patient-level independence or absence of duplicate images across splits. Check patient identifiers and image hashes before reporting clinical generalisation.
- The membership-inference threshold is an empirical loss-based diagnostic; it does not prove privacy protection.
- AUC=1.0 or near-perfect F1 must be interpreted with confidence intervals, cohort characteristics, data provenance, and leakage checks.
- End-to-end runtime includes influence estimation and strategy-selection overhead and may exceed full retraining, as reported in the manuscript.
- A validation F1 utility gate is not a forgetting-success certificate. The fallback should be tracked separately from successful approximate updates.
- The code uses ImageDataGenerator and other APIs whose support can change across TensorFlow/Keras versions; dependencies should be tested and pinned for archival reproduction.
