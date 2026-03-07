# Changelog

## 0.1.0 (2026-03-07)

Initial release.

### Algorithms

- K-Means (Lloyd's + K-means++ init, n_init restarts)
- Mini-batch K-Means (Sculley 2010, running average centroid update)
- DBSCAN (density-based, KD-tree accelerated for Euclidean/Manhattan, brute-force for cosine)
- Hierarchical agglomerative (single/complete/average/Ward linkage, SciPy-compatible dendrogram)
- FastPAM k-medoids (Schubert & Rousseeuw 2019, BUILD + SWAP)

### Validation Indices

- Silhouette (with ignore_noise flag)
- Calinski-Harabasz
- Davies-Bouldin
- Adjusted Rand Index

### Infrastructure

- C11 core (~1500 lines), no dependencies beyond libc/libm
- WASM build (Emscripten, single-file)
- JS wrapper with ClusterModel class
- Python ctypes bindings
- Binary serialization (CL01 format)
- Dendrogram cut utilities (by k or by distance)
- 33 C tests (352 assertions), 28 JS tests, 15 Python tests
