The autoscaling experiment was run on a local `kind` cluster with HPA configured for CPU target utilization `60%`. `inference-service` scaled to `4` replicas with first scale-up observed after `34.2` seconds; `text-processing-service` scaled to `3` replicas with first scale-up observed after `28.6` seconds.

Calibration and per-pod resource measurements were captured for both services. `inference-service` used calibrated concurrency `10`; `text-processing-service` used calibrated concurrency `20`.
