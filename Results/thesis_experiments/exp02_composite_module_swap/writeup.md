The composite module swap was measured by changing `spec.activeModel` from `tgn` to `tall` while both model stacks were present in the cluster. All six logical alias services changed targets, and the full alias update completed in `0.271` seconds.

The MongoDB routing summary reported `active_model` as `tall` after the reconcile, and the operator logs showed a reconcile loop with `aliases=6`. No claim beyond the observed alias retargeting and operator reconciliation was made.
