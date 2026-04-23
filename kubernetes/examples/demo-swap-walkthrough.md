# Swap Demo Walkthrough

This walkthrough is intended for recording a short demo of:

- single-service swap
- whole-service swap

The demo uses the smoke backends in `router-smoke-backends.yaml` so the routed alias services return visibly different backend identities: `tgn-backend` and `tall-backend`.

## One-Time Setup

Apply the shared router infrastructure and the smoke backends:

```bash
kubectl apply -f kubernetes/manifests/mongodb.yaml
kubectl apply -f kubernetes/manifests/modelrouter-crd.yaml
kubectl apply -f kubernetes/manifests/operator-rbac.yaml
kubectl apply -f kubernetes/manifests/operator-deployment.yaml
kubectl apply -f kubernetes/examples/router-smoke-backends.yaml
```

Wait for the operator and the two smoke deployments:

```bash
kubectl rollout status deployment/model-router-operator
kubectl rollout status deployment/tgn-backend
kubectl rollout status deployment/tall-backend
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
kubectl run curl-baseline --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://text-processing-service:8003
```

Expected result:

- `text-processing-service` resolves to the TGN backend
- `inference-service` resolves to the TGN backend

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
kubectl run curl-single-text --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://text-processing-service:8003
kubectl run curl-single-infer --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://inference-service:8005
```

Expected result:

- `text-processing-service` now resolves to `tall-backend`
- `inference-service` still resolves to `tgn-backend`

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
kubectl run curl-whole-text --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://text-processing-service:8003
kubectl run curl-whole-infer --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://inference-service:8005
```

Expected result:

- both aliases now resolve to `tall-backend`
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
