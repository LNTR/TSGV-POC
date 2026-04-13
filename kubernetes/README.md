# Shared Kubernetes Model Router

This directory now holds the shared Kubernetes routing layer used by both the TGN and TALL model stacks.

## Layout

- `operator/` contains the Go-based `model-router-operator`
- `manifests/` contains the CRD, RBAC, operator deployment, and MongoDB manifest
- `examples/` contains copy-ready `ModelRouter` examples and smoke-test backends

## What The Operator Does

- watches `ModelRouter` custom resources
- reads `spec.activeModel` and the model-specific service mappings under `spec.models`
- creates logical alias `Service` objects such as `inference-service` and `training-service`
- points each alias at the active model implementation using `ExternalName` services
- writes the active routing summary to MongoDB when `MONGODB_URI` is configured
- publishes the current routing snapshot in a ConfigMap named `<router-name>-active-routing`

## Build The Operator Image

Build the shared Go operator image from the repo root:

```bash
docker build -t model-router-operator:latest kubernetes/operator
```

## Install The Shared Router

1. Apply the base infrastructure:

```bash
kubectl apply -f kubernetes/manifests/mongodb.yaml
kubectl apply -f kubernetes/manifests/modelrouter-crd.yaml
kubectl apply -f kubernetes/manifests/operator-rbac.yaml
kubectl apply -f kubernetes/manifests/operator-deployment.yaml
```

2. Deploy your model-specific services with distinct names such as `tgn-inference-service` or `tall-inference-service`.

3. Apply a router definition:

```bash
kubectl apply -f kubernetes/examples/model-router-tgn.yaml
```

4. Switch models later by applying another `ModelRouter` manifest:

```bash
kubectl apply -f kubernetes/examples/model-router-foobar.yaml
```

## Smoke Flow

```bash
kubectl apply -f kubernetes/examples/router-smoke-backends.yaml
kubectl apply -f kubernetes/examples/model-router-tgn.yaml
kubectl get modelrouter tsgv-router -o yaml
kubectl get svc inference-service text-processing-service
```

Switch to the alternate model:

```bash
kubectl apply -f kubernetes/examples/model-router-foobar.yaml
kubectl get modelrouter tsgv-router -o yaml
kubectl get svc inference-service text-processing-service
```

## MongoDB Collections

When MongoDB is enabled, the operator writes into the `service_registry` database by default:

- `model_router_state` stores one summary document per `ModelRouter`
- `routes` stores one document per logical alias route

## Notes

- the default manifests assume the `default` namespace
- if you deploy the operator elsewhere, update the `ServiceAccount` subject in `manifests/operator-rbac.yaml`
