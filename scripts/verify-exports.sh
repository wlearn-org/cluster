#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/cluster.cjs"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_cluster_get_last_error
  wl_cluster_fit_kmeans
  wl_cluster_fit_minibatch
  wl_cluster_fit_dbscan
  wl_cluster_fit_hierarchical
  wl_cluster_fit_fastpam
  wl_cluster_predict
  wl_cluster_dendro_cut_k
  wl_cluster_dendro_cut_dist
  wl_cluster_silhouette
  wl_cluster_silhouette_samples
  wl_cluster_calinski_harabasz
  wl_cluster_davies_bouldin
  wl_cluster_adjusted_rand
  wl_cluster_get_n_samples
  wl_cluster_get_n_features
  wl_cluster_get_n_clusters
  wl_cluster_get_algorithm
  wl_cluster_get_metric
  wl_cluster_get_n_iter
  wl_cluster_get_n_noise
  wl_cluster_get_inertia
  wl_cluster_get_labels
  wl_cluster_get_centers
  wl_cluster_get_medoid_indices
  wl_cluster_get_medoid_coords
  wl_cluster_get_core_mask
  wl_cluster_get_dendro_n_merges
  wl_cluster_get_dendro_left
  wl_cluster_get_dendro_right
  wl_cluster_get_dendro_distance
  wl_cluster_get_dendro_size
  wl_cluster_save
  wl_cluster_load
  wl_cluster_free
  wl_cluster_free_buffer
)

missing=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "_${fn}" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} exports missing from ${WASM_FILE}"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified."
