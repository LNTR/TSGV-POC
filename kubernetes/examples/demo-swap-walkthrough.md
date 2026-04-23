# Swap Demo Walkthrough

This walkthrough is intended for recording a short demo of:

- single-service swap
- whole-service swap

The demo uses per-service smoke backends in `router-smoke-backends.yaml`, so each logical route returns the concrete implementation name such as `tgn-text-processing-service` or `tall-inference-service`.

## Cluster Up

Create the `kind` cluster:

```bash
kind create cluster --name tsgv-swap-demo
```

Build the operator image into the local Docker environment:

```bash
docker build -t model-router-operator:latest -f kubernetes/operator/Dockerfile .
```

Load that local image into the `kind` cluster so Kubernetes does not try to pull it from Docker Hub:

```bash
kind load docker-image model-router-operator:latest --name tsgv-swap-demo
```

## One-Time Setup

Apply the shared router infrastructure and the smoke backends:

```bash
kubectl apply -f kubernetes/manifests/mongodb.yaml
kubectl apply -f kubernetes/manifests/modelrouter-crd.yaml
kubectl apply -f kubernetes/manifests/operator-rbac.yaml
kubectl apply -f kubernetes/manifests/operator-deployment.yaml
kubectl apply -f kubernetes/examples/router-smoke-backends.yaml
```

If you are reusing an existing demo cluster from an older version of the smoke setup, remove the previous shared backends first:

```bash
kubectl delete deployment tgn-backend tall-backend --ignore-not-found
```

Wait for the operator and all smoke deployments:

```bash
kubectl rollout status deployment/model-router-operator
kubectl wait --for=condition=available deployment -l app.kubernetes.io/part-of=router-smoke-demo --timeout=120s
```

## Recording Flow

### 1. Baseline

Apply the baseline router:

```bash
kubectl apply -f kubernetes/examples/model-router-demo-baseline.yaml
```

Show the baseline alias targets:

```bash
kubectl get svc text-processing-service inference-service -o wide
kubectl get configmap tsgv-router-active-routing -o yaml
kubectl logs deployment/model-router-operator --tail=40
```

Optional route check:

```bash
kubectl run curl-baseline --rm -i --restart=Never --image=busybox:1.36 --command -- \
  sh -lc "wget -qO- http://text-processing-service:8003; echo; wget -qO- http://inference-service:8005; echo"
```

Expected result:

- `text-processing-service` resolves to `tgn-text-processing-service`
- `inference-service` resolves to `tgn-inference-service`

### 2. Single-Service Swap

Apply the hybrid router:

```bash
kubectl apply -f kubernetes/examples/model-router-demo-single-service-swap.yaml
```

Show that only one logical alias changed:

```bash
kubectl get svc text-processing-service inference-service -o yaml
kubectl get configmap tsgv-router-active-routing -o yaml
kubectl logs deployment/model-router-operator --tail=40
```

Route checks:

```bash
kubectl run curl-single --rm -i --restart=Never --image=busybox:1.36 --command -- \
  sh -lc "wget -qO- http://text-processing-service:8003; echo; wget -qO- http://inference-service:8005; echo"
```

Expected result:

- `text-processing-service` now resolves to `tall-text-processing-service`
- `inference-service` still resolves to `tgn-inference-service`

This is the cleanest single-service swap shot for the demo.

### 3. Whole-Service Swap

Apply the full TALL router:

```bash
kubectl apply -f kubernetes/examples/model-router-demo-whole-service-swap.yaml
```

Show that all logical aliases moved:

```bash
kubectl get svc text-processing-service inference-service training-service evaluation-service -o yaml
kubectl get configmap tsgv-router-active-routing -o yaml
kubectl logs deployment/model-router-operator --tail=40
```

Route checks:

```bash
kubectl run curl-whole --rm -i --restart=Never --image=busybox:1.36 --command -- \
  sh -lc "wget -qO- http://text-processing-service:8003; echo; wget -qO- http://inference-service:8005; echo; wget -qO- http://training-service:8004; echo; wget -qO- http://evaluation-service:8006; echo"
```

Expected result:

- the aliases now resolve to their `tall-*` implementations
- the ConfigMap reflects the new active model and route map

## Best Things To Show On Screen

- the applied manifest name
- `kubectl get svc ... -o yaml` for the alias services
- `kubectl get configmap tsgv-router-active-routing -o yaml`
- `kubectl logs deployment/model-router-operator --tail=40`
- the `curl` result before and after each swap

## Suggested Narration

- Baseline: all aliases point to `tgn`
- Single-service swap: only `text-processing-service` changes, the rest remain on `tgn`
- Whole-service swap: all aliases move to `tall`
- The operator reconciles the `ModelRouter` CR and updates logical alias services without changing client-facing names

## Cluster Down

Delete the demo cluster when you are finished:

```bash
kind delete cluster --name tsgv-swap-demo
```
