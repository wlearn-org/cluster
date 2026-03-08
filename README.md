# @wlearn/cluster

Clustering library from scratch in C11. K-Means, Mini-batch K-Means, DBSCAN, Hierarchical Agglomerative, FastPAM k-medoids, plus validation indices (Silhouette, Calinski-Harabasz, Davies-Bouldin, Adjusted Rand Index).

No browser-native WASM clustering library exists. This fills that gap.

## Install

```bash
npm install @wlearn/cluster
```

## Quick Start

```js
const { ClusterModel, silhouette, adjustedRand } = require('@wlearn/cluster')

// K-Means
const km = await ClusterModel.create({ method: 'kmeans', k: 3 })
km.fit(X)
console.log(km.labels)       // Int32Array
console.log(km.inertia)      // within-cluster SSE
const pred = km.predict(X_new) // assign to nearest centroid
const sil = km.score(X)      // silhouette score

// DBSCAN
const db = await ClusterModel.create({ method: 'dbscan', eps: 0.5, minPts: 5 })
db.fit(X)
console.log(db.labels)       // -1 = noise
console.log(db.nNoise)
// db.predict() throws -- DBSCAN doesn't support predict

// FastPAM k-medoids
const fp = await ClusterModel.create({ method: 'fastpam', k: 3, metric: 'manhattan' })
fp.fit(X)
console.log(fp.medoidIndices) // indices into original data
console.log(fp.medoidCoords)  // cached medoid coordinates

// Hierarchical
const hc = await ClusterModel.create({ method: 'hierarchical', k: 3, linkage: 'ward' })
hc.fit(X)
console.log(hc.dendrogram)   // SciPy-compatible merge list

// Standalone validation
const s = silhouette(X, labels)
const ari = adjustedRand(trueLabels, predLabels)
```

## Algorithms

### K-Means

Lloyd's algorithm with K-means++ initialization. Euclidean distance only (K-Means optimizes within-cluster SSE, which is a Euclidean objective).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 3 | Number of clusters |
| `maxIter` | 300 | Maximum iterations |
| `nInit` | 10 | Random restarts, keep best |
| `init` | `'kmpp'` | `'kmpp'` or `'random'` |
| `tol` | 1e-4 | Convergence tolerance |
| `seed` | 42 | Random seed |

### Mini-batch K-Means

Sculley (2010) mini-batch variant. Faster convergence on large datasets.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 3 | Number of clusters |
| `batchSize` | 1024 | Mini-batch size |
| `maxIter` | 100 | Maximum iterations |
| `seed` | 42 | Random seed |

### DBSCAN

Density-based clustering. KD-tree accelerated for Euclidean/Manhattan, brute-force for cosine.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps` | 0.5 | Neighborhood radius |
| `minPts` | 5 | Minimum points for core point |
| `metric` | `'euclidean'` | `'euclidean'`, `'manhattan'`, or `'cosine'` |

Noise points get label -1. Does not support `predict()`.

### Hierarchical Agglomerative

Produces a SciPy-compatible dendrogram. Cut by k or by distance threshold.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` / `nClusters` | 2 | Number of clusters |
| `linkage` | `'ward'` | `'single'`, `'complete'`, `'average'`, `'ward'` |
| `metric` | `'euclidean'` | Distance metric (Ward requires Euclidean) |
| `distanceThreshold` | 0.0 | Cut by distance instead of k |

Memory: O(n^2). Practical limit: n < ~5000 in WASM.

### FastPAM (k-Medoids)

Schubert & Rousseeuw (2019). Medoids are actual data points, not means.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | 3 | Number of medoids |
| `maxIter` | 100 | Maximum SWAP iterations |
| `metric` | `'euclidean'` | Any metric (`'euclidean'`, `'manhattan'`, `'cosine'`) |
| `seed` | 42 | Random seed |

## Validation Indices

Available as standalone functions after calling `loadCluster()`:

```js
const { silhouette, calinskiHarabasz, daviesBouldin, adjustedRand } = require('@wlearn/cluster')

silhouette(X, labels)              // mean silhouette coefficient
calinskiHarabasz(X, labels)        // between/within variance ratio
daviesBouldin(X, labels)           // average cluster similarity (lower = better)
adjustedRand(trueLabels, predLabels) // pairwise agreement (1.0 = perfect)
```

## Input Format

X can be:
- `Array` of arrays: `[[1, 2], [3, 4], ...]`
- Flat `Float64Array` (row-major, must specify `nFeatures`)

## Save / Load

```js
const bytes = model.save()          // Uint8Array (CL01 binary format)
const model2 = await ClusterModel.load(bytes)
```

## Python

```python
import ctypes
lib = ctypes.CDLL('build/libcluster.so')
# See test/test_python.py for full ctypes bindings
```

## Build

```bash
# C tests
cd build && cmake .. -DBUILD_TESTING=ON && make && ./test_cluster

# WASM
bash scripts/build-wasm.sh

# JS tests
node test/test.js

# Python tests
python test/test_python.py
```

## Tests

- 33 C tests (352 assertions)
- 28 JS tests (WASM)
- 15 Python tests (6 with sklearn parity)

## License

Apache-2.0
