# Kubernetes deployment manifests

This directory holds the GitOps-style Kubernetes manifests for the three
Milvus 3.0 demos plus the shared Milvus 3.0 stack. They mirror the layout of
`zilliztech/vdc-deploy` (`milvus-demo/apps/*` and `milvus-demo/shared-services/*`)
so the packages can be moved there and reviewed by SRE.

## Layout

```text
deploy/
├── shared-services/
│   └── milvus3-stack/     # one shared Milvus 3.0 + etcd + MinIO stack
└── apps/
    ├── function-chain-rerank/
    ├── structarray-search/
    └── embedding-list/
```

## Shared Milvus 3.0 stack

One `milvusdb/milvus:v3.0.0` standalone instance backs all three demos. Each
demo connects with `MILVUS_URI=http://milvus3:19530` and uses a distinct
collection, so they share the stack without interfering.

Resource budgets (requests / limits):

| Component | CPU request / limit | Memory request / limit | Storage |
| --- | --- | --- | --- |
| Milvus 3.0 | 2 / 8 | 4Gi / 16Gi | 10Gi PVC |
| etcd | 100m / 1 | 128Mi / 1Gi | 4Gi PVC |
| MinIO | 100m / 2 | 256Mi / 2Gi | 10Gi PVC |

These are generous for the tiny demo datasets; SRE may tune them down.

## Before deploying

1. **MinIO credentials**: create the `milvus3-minio-secret` Secret in the
   `demo` namespace with `access_key` and `secret_key` keys, via the approved
   offline SRE channel. The manifests reference it and never inline it.

2. **Image tags**: each app `kustomization.yaml` pins
   `harbor-us1.zilliz.cc/uso/milvus3-demos-*` with an immutable tag. Push the
   images there and bump the tag in the same PR.

3. **CoVLA data (structarray-search)**: the deployment mounts the approved
   CoVLA 30-video slice via NFS. Replace the placeholder
   `server: 10.1.2.68 / path: /covla-dataset` with the SRE-approved NFS
   location. Remove this volume once the demo switches to fully synthetic
   generated frames (planned follow-up).

4. **Sub-path routing**: each app is served at `demos.milvus.io/<name>` and
   built/run with `VITE_BASE_PATH` / `APP_ROOT_PATH` set to the same value.
   The images bake the model weights, so no model NFS mount is required.

## Public exposure

The Ingresses use `group.name: milvus-io` and `scheme: internet-facing`
(demos.milvus.io). Public exposure requires the SRE public-execution review
described in the vdc-deploy README before merge.
