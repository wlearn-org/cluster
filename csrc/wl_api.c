/*
 * wl_api.c -- C ABI wrapper for cluster (WASM/FFI boundary)
 *
 * All params as primitives (no struct passing across ABI).
 */

#include "cluster.h"
#include <stdlib.h>

const char *wl_cluster_get_last_error(void) {
    return cl_get_error();
}

/* ========== Fit functions ========== */

cl_result_t *wl_cluster_fit_kmeans(
    const double *X, int nrow, int ncol,
    int k, int max_iter, int n_init, int init,
    double tol, int seed
) {
    cl_kmeans_params_t p;
    cl_kmeans_params_init(&p);
    p.k = k;
    p.max_iter = max_iter;
    p.n_init = n_init;
    p.init = init;
    p.tol = tol;
    p.seed = (uint32_t)seed;
    return cl_kmeans_fit(X, (int32_t)nrow, (int32_t)ncol, &p);
}

cl_result_t *wl_cluster_fit_minibatch(
    const double *X, int nrow, int ncol,
    int k, int max_iter, int n_init, int init,
    int batch_size, double tol, int seed
) {
    cl_minibatch_params_t p;
    cl_minibatch_params_init(&p);
    p.k = k;
    p.max_iter = max_iter;
    p.n_init = n_init;
    p.init = init;
    p.batch_size = batch_size;
    p.tol = tol;
    p.seed = (uint32_t)seed;
    return cl_minibatch_fit(X, (int32_t)nrow, (int32_t)ncol, &p);
}

cl_result_t *wl_cluster_fit_dbscan(
    const double *X, int nrow, int ncol,
    double eps, int min_pts, int metric
) {
    cl_dbscan_params_t p;
    cl_dbscan_params_init(&p);
    p.eps = eps;
    p.min_pts = min_pts;
    p.metric = metric;
    return cl_dbscan_fit(X, (int32_t)nrow, (int32_t)ncol, &p);
}

cl_result_t *wl_cluster_fit_hierarchical(
    const double *X, int nrow, int ncol,
    int n_clusters, double distance_threshold,
    int linkage, int metric
) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.n_clusters = n_clusters;
    p.distance_threshold = distance_threshold;
    p.linkage = linkage;
    p.metric = metric;
    return cl_hierarchical_fit(X, (int32_t)nrow, (int32_t)ncol, &p);
}

cl_result_t *wl_cluster_fit_fastpam(
    const double *X, int nrow, int ncol,
    int k, int max_iter, int init, int metric, int seed
) {
    cl_fastpam_params_t p;
    cl_fastpam_params_init(&p);
    p.k = k;
    p.max_iter = max_iter;
    p.init = init;
    p.metric = metric;
    p.seed = (uint32_t)seed;
    return cl_fastpam_fit(X, (int32_t)nrow, (int32_t)ncol, &p);
}

/* ========== Predict ========== */

int wl_cluster_predict(const cl_result_t *r, const double *X, int nrow, int ncol,
                        int *out) {
    return cl_predict(r, X, (int32_t)nrow, (int32_t)ncol, (int32_t *)out);
}

/* ========== Dendrogram utilities ========== */

int wl_cluster_dendro_cut_k(const cl_result_t *r, int n_clusters, int *out) {
    return cl_dendro_cut_k(r, (int32_t)n_clusters, (int32_t *)out);
}

int wl_cluster_dendro_cut_dist(const cl_result_t *r, double threshold, int *out) {
    return cl_dendro_cut_dist(r, threshold, (int32_t *)out);
}

/* ========== Validation ========== */

double wl_cluster_silhouette(const double *X, int nrow, int ncol,
                               const int *labels, int metric, int ignore_noise) {
    return cl_silhouette(X, (int32_t)nrow, (int32_t)ncol,
                          (const int32_t *)labels, (int32_t)metric, (int32_t)ignore_noise);
}

int wl_cluster_silhouette_samples(const double *X, int nrow, int ncol,
                                    const int *labels, int metric, int ignore_noise,
                                    double *out) {
    return cl_silhouette_samples(X, (int32_t)nrow, (int32_t)ncol,
                                  (const int32_t *)labels, (int32_t)metric,
                                  (int32_t)ignore_noise, out);
}

double wl_cluster_calinski_harabasz(const double *X, int nrow, int ncol,
                                      const int *labels) {
    return cl_calinski_harabasz(X, (int32_t)nrow, (int32_t)ncol, (const int32_t *)labels);
}

double wl_cluster_davies_bouldin(const double *X, int nrow, int ncol,
                                   const int *labels) {
    return cl_davies_bouldin(X, (int32_t)nrow, (int32_t)ncol, (const int32_t *)labels);
}

double wl_cluster_adjusted_rand(const int *labels_true, const int *labels_pred, int n) {
    return cl_adjusted_rand((const int32_t *)labels_true, (const int32_t *)labels_pred, (int32_t)n);
}

/* ========== Result accessors ========== */

int wl_cluster_get_n_samples(const cl_result_t *r) { return r ? r->n_samples : 0; }
int wl_cluster_get_n_features(const cl_result_t *r) { return r ? r->n_features : 0; }
int wl_cluster_get_n_clusters(const cl_result_t *r) { return r ? r->n_clusters : 0; }
int wl_cluster_get_algorithm(const cl_result_t *r) { return r ? r->algorithm : -1; }
int wl_cluster_get_metric(const cl_result_t *r) { return r ? r->metric : 0; }
int wl_cluster_get_n_iter(const cl_result_t *r) { return r ? r->n_iter : 0; }
int wl_cluster_get_n_noise(const cl_result_t *r) { return r ? r->n_noise : 0; }
double wl_cluster_get_inertia(const cl_result_t *r) { return r ? r->inertia : 0.0; }

const int *wl_cluster_get_labels(const cl_result_t *r) {
    return r ? (const int *)r->labels : NULL;
}

const double *wl_cluster_get_centers(const cl_result_t *r) {
    return r ? r->centers : NULL;
}

const int *wl_cluster_get_medoid_indices(const cl_result_t *r) {
    return r ? (const int *)r->medoid_indices : NULL;
}

const double *wl_cluster_get_medoid_coords(const cl_result_t *r) {
    return r ? r->medoid_coords : NULL;
}

const unsigned char *wl_cluster_get_core_mask(const cl_result_t *r) {
    return r ? r->core_mask : NULL;
}

/* Dendrogram accessors (per merge entry) */
int wl_cluster_get_dendro_n_merges(const cl_result_t *r) {
    return (r && r->dendrogram) ? r->n_samples - 1 : 0;
}

int wl_cluster_get_dendro_left(const cl_result_t *r, int i) {
    if (!r || !r->dendrogram || i < 0 || i >= r->n_samples - 1) return -1;
    return r->dendrogram[i].left;
}

int wl_cluster_get_dendro_right(const cl_result_t *r, int i) {
    if (!r || !r->dendrogram || i < 0 || i >= r->n_samples - 1) return -1;
    return r->dendrogram[i].right;
}

double wl_cluster_get_dendro_distance(const cl_result_t *r, int i) {
    if (!r || !r->dendrogram || i < 0 || i >= r->n_samples - 1) return 0.0;
    return r->dendrogram[i].distance;
}

int wl_cluster_get_dendro_size(const cl_result_t *r, int i) {
    if (!r || !r->dendrogram || i < 0 || i >= r->n_samples - 1) return 0;
    return r->dendrogram[i].size;
}

/* ========== Save/Load/Free ========== */

int wl_cluster_save(const cl_result_t *r, char **out_buf, int *out_len) {
    int32_t len32;
    int ret = cl_save(r, out_buf, &len32);
    if (ret == 0 && out_len) *out_len = (int)len32;
    return ret;
}

cl_result_t *wl_cluster_load(const char *buf, int len) {
    return cl_load(buf, (int32_t)len);
}

void wl_cluster_free(cl_result_t *r) {
    cl_free(r);
}

void wl_cluster_free_buffer(void *ptr) {
    cl_free_buffer(ptr);
}
