The single-component swap experiment was not run on 2026-04-18. The requested follow-up scope was limited to TGN-only work, while this experiment requires a multi-model routing setup. This host also did not have `kubectl` installed, so the `ModelRouter` custom resource could not be applied and the MongoDB `routes` collection could not be inspected under a live reconcile loop.

All requested measurement fields were therefore recorded as `"not_run"` in `summary.json`. No latency or route-diff claim was made from static inspection alone.
