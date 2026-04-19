This directory contains the checked-in assets used by the kind-backed scaling experiments for `exp10_kubernetes_autoscaling_behavior` and `exp11_horizontal_scaling_concurrent_requests`.

The runner mounts `TGN/implementation/storage` into the kind node, applies the TGN `inference-service` and `text-processing-service`, and uses the files in `fixtures/` plus `loadgen.py` to drive in-cluster load while collecting HPA, replica, and `kubectl top` measurements.
