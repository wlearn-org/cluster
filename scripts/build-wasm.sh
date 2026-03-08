#!/bin/bash
set -euo pipefail

# Build Cluster C11 core as WASM via Emscripten

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='[
  "_wl_cluster_get_last_error",
  "_wl_cluster_fit_kmeans",
  "_wl_cluster_fit_minibatch",
  "_wl_cluster_fit_dbscan",
  "_wl_cluster_fit_hierarchical",
  "_wl_cluster_fit_fastpam",
  "_wl_cluster_predict",
  "_wl_cluster_dendro_cut_k",
  "_wl_cluster_dendro_cut_dist",
  "_wl_cluster_silhouette",
  "_wl_cluster_silhouette_samples",
  "_wl_cluster_calinski_harabasz",
  "_wl_cluster_davies_bouldin",
  "_wl_cluster_adjusted_rand",
  "_wl_cluster_get_n_samples",
  "_wl_cluster_get_n_features",
  "_wl_cluster_get_n_clusters",
  "_wl_cluster_get_algorithm",
  "_wl_cluster_get_metric",
  "_wl_cluster_get_n_iter",
  "_wl_cluster_get_n_noise",
  "_wl_cluster_get_inertia",
  "_wl_cluster_get_labels",
  "_wl_cluster_get_centers",
  "_wl_cluster_get_medoid_indices",
  "_wl_cluster_get_medoid_coords",
  "_wl_cluster_get_core_mask",
  "_wl_cluster_get_dendro_n_merges",
  "_wl_cluster_get_dendro_left",
  "_wl_cluster_get_dendro_right",
  "_wl_cluster_get_dendro_distance",
  "_wl_cluster_get_dendro_size",
  "_wl_cluster_save",
  "_wl_cluster_load",
  "_wl_cluster_free",
  "_wl_cluster_free_buffer",
  "_malloc",
  "_free"
]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32"]'

emcc \
  "${PROJECT_DIR}/csrc/cluster.c" \
  "${PROJECT_DIR}/csrc/wl_api.c" \
  -I "${PROJECT_DIR}/csrc" \
  -o "${OUTPUT_DIR}/cluster.js" \
  -std=c11 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s EXPORT_NAME=createCluster \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: none (C11 from scratch)
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c11 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/cluster.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
