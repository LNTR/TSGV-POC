The concurrent horizontal-scaling experiment compared single-replica and HPA-enabled runs across the fixed concurrency sweep `1, 5, 10, 20, 40`. `inference-service` recorded baseline throughput `24.82` rps and HPA-enabled throughput `35.59` rps at concurrency `40`; `text-processing-service` recorded baseline throughput `33.31` rps and HPA-enabled throughput `47.78` rps at concurrency `40`.

Per-stage latency, throughput, non-2xx rate, replica maxima, and pod resource measurements were captured in `summary.json`.
