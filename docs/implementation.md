# Formula module implementation ledger

Authority: user-approved spectrum-to-formula design in this conversation.

Tasks: (1) chemistry/candidate engine, (2) preprocessing/features/predictor,
(3) audit/preparation/training/evaluation, (4) CLI/docs/integration validation.

Ruling: workspace is a new non-Git directory, so implement here without creating
a worktree or committing. Preserve all original data. Use a project-local Python
3.11 virtual environment; do not modify the user's Conda environments.

Ruling: add Numba for CPU mass decomposition and PyArrow for bounded-memory
Parquet batch reads. Polars handles metadata grouping and reports.

Pre-flight: candidate engine exposes neutral formula vectors and exact masses;
feature extraction consumes these and normalized spectra; training and prediction
use the same feature extractor/configuration. Public inference never consumes labels.

Progress: implementation started; no validation claims yet.

Ruling: replace recursive Numba search with an iterative stack after first-run
chemistry tests passed but a fresh process crashed loading recursive cached code.
This preserves the search and pruning mathematics, avoids cached recursive pointers.

Ruling: add an explicit 250,000-result per-search resource ceiling. Searches over
this ceiling produce resource_limit, not a silently incomplete ranked list.
This is distinct from the planned deterministic 10,000-candidate ranking cap.
Training/evaluation count resource failures as candidate-recall failures. Users
can raise the resource ceiling in a versioned config for higher mass samples.

Tasks 1-2: 39 core/data integration tests implemented; chemistry, input validation,
fragment atom balance and model persistence exercised. Exact theoretical-ppm
window inversion replaces the symmetric approximation at boundary values.

Final review (independent agent): two P2 findings, partial CLI config recognition
and invalid instrument metadata; reproduced and fixed with regression tests.
Inference file API loads all groups in memory; accepted for current small test,
documented large-file callers should feed groups to predict. Training remains
partitioned. No mass/electron, fragment atom balance, structure leakage or truth
injection defect found in the bounded review. Pilot accuracy is verified separately.

Performance investigation: py-spy located Python per-assignment construction and
large fragment subsets in the first pilot (at molecule 23). Stopped that task-owned
process; retained raw/prepared data. Vectorized assignment construction, sorted
direct matches for early exit, and enumerated the smaller neutral-loss side when
candidate ion mass spread <=0.05 Da, with widened windows and candidate-specific
exact error checks. Independent exhaustive atom-subset tests cover equivalence.
Feature cache version bumped; pilot reruns with optimized implementation.

Additional real-data case: a 97-spectrum training molecule contained strong peaks
above all candidate precursor masses. These provably impossible atom subsets were
still searched. Added an exact parent-mass bound (keeping noise intensity in the
denominator), batched ion composition conversion, bounded candidate union storage,
and periodic resumable feature checkpoints. This is a performance/correct resource
accounting fix, covered by a failing resource-budget regression before the change.

Evaluation provenance fix: pilot seen/unseen strata must use formulas actually
included in fitted groups, not all formulas in the available training partition.
Persist that exact set in model metadata and use it for learned-model evaluation;
the baseline uses the available training partition. Regression test proves the
model's provenance remains authoritative when the pool's formula list changes.

Completed:
- Full raw-data audit and preparation: 2,539,608 -> 2,104,245 accepted spectra;
  214,672/26,632/26,951 structures in disjoint train/validation/test partitions.
- 400/400 public molecules through baseline and learned CLI; each 388 ok,12 truncated.
  Final baseline candidate lists match original exactly; summed molecule time
  1101.33 ->107.89 seconds (not parallel batch wall-clock).
- Pilot128 train/32 validation:122 fitted groups,119 fitted formulas,66 trees;
  validation MRR25 .66245 vs baseline .109375; real model saved and reloaded.
- Independent32-molecule holdout: Top1 .625, MRR25 .750841; all formula labels
  unseen in actual pilot fitting. Full fitting was not performed or claimed.
- Final pytest:46 passed in14.30s; pip check and compileall passed.
- README, runnable example, configs, pinned dependencies and validation report delivered.

Remaining research (not implementation blockers): larger training runs, stronger
chemical priors/baselines, higher-mass budget/coverage and domain-shift evaluation.
