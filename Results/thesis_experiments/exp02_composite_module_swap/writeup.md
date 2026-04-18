The composite module swap experiment was not run on 2026-04-18. The requested follow-up scope was limited to TGN-only work, while this experiment requires a multi-model routing setup. This host also did not have `kubectl` installed, so the `spec.activeModel` change could not be applied and the operator reconcile loop could not be observed.

All requested measurement fields were therefore recorded as `"not_run"` in `summary.json`. No claim was made about reconciliation time or the number of alias Services updated.
