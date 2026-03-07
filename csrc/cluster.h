/*
 * cluster.h -- Clustering library (C11, from scratch)
 *
 * Algorithms:
 *   K-Means (Lloyd's + K-means++ init) + Mini-batch variant
 *   DBSCAN (density-based, KD-tree accelerated)
 *   Hierarchical agglomerative (single/complete/average/Ward)
 *   FastPAM k-medoids (Schubert & Rousseeuw 2019)
 *
 * Validation indices:
 *   Silhouette, Calinski-Harabasz, Davies-Bouldin, Adjusted Rand Index
 *
 * Binary serialization (CL01 format)
 */

#ifndef CLUSTER_H
#define CLUSTER_H

#include <stdint.h>
#include <stddef.h>

/* ========== Constants ========== */

/* Distance metrics */
enum {
    CL_DIST_EUCLIDEAN  = 0,
    CL_DIST_MANHATTAN  = 1,
    CL_DIST_COSINE     = 2
};

/* K-Means initialization */
enum {
    CL_INIT_KMPP       = 0,   /* K-means++ (default) */
    CL_INIT_RANDOM     = 1
};

/* Hierarchical linkage */
enum {
    CL_LINK_SINGLE     = 0,
    CL_LINK_COMPLETE   = 1,
    CL_LINK_AVERAGE    = 2,
    CL_LINK_WARD       = 3
};

/* Algorithm identifiers */
enum {
    CL_ALGO_KMEANS       = 0,
    CL_ALGO_MINIBATCH    = 1,
    CL_ALGO_DBSCAN       = 2,
    CL_ALGO_HIERARCHICAL = 3,
    CL_ALGO_FASTPAM      = 4
};

/* ========== Parameter structs ========== */

typedef struct {
    int32_t  k;          /* number of clusters (required, > 0) */
    int32_t  max_iter;   /* default 300 */
    int32_t  n_init;     /* random restarts, keep best (default 10) */
    int32_t  init;       /* CL_INIT_* (default CL_INIT_KMPP) */
    double   tol;        /* convergence: relative centroid shift (default 1e-4) */
    uint32_t seed;       /* default 42 */
} cl_kmeans_params_t;

typedef struct {
    int32_t  k;
    int32_t  max_iter;   /* default 100 */
    int32_t  n_init;     /* default 3 */
    int32_t  init;       /* default CL_INIT_KMPP */
    int32_t  batch_size; /* default 1024 */
    double   tol;        /* default 0.0 (run all iters) */
    uint32_t seed;
} cl_minibatch_params_t;

typedef struct {
    double   eps;        /* neighborhood radius (required, > 0) */
    int32_t  min_pts;    /* minimum points for core point (default 5) */
    int32_t  metric;     /* CL_DIST_* (default CL_DIST_EUCLIDEAN) */
} cl_dbscan_params_t;

typedef struct {
    int32_t  n_clusters;          /* number of clusters to cut (0 = full dendrogram) */
    double   distance_threshold;  /* cut distance (0.0 = use n_clusters) */
    int32_t  linkage;             /* CL_LINK_* (default CL_LINK_WARD) */
    int32_t  metric;              /* CL_DIST_* (default CL_DIST_EUCLIDEAN) */
} cl_hierarchical_params_t;

typedef struct {
    int32_t  k;          /* number of medoids (required, > 0) */
    int32_t  max_iter;   /* default 100 */
    int32_t  init;       /* CL_INIT_* (default CL_INIT_KMPP analog) */
    int32_t  metric;     /* CL_DIST_* (default CL_DIST_EUCLIDEAN) */
    uint32_t seed;
} cl_fastpam_params_t;

/* ========== Result types ========== */

/* Dendrogram merge entry (SciPy-compatible) */
typedef struct {
    int32_t  left;       /* left child cluster index */
    int32_t  right;      /* right child cluster index */
    double   distance;   /* merge distance */
    int32_t  size;       /* points in merged cluster */
} cl_merge_t;

/* Unified result for all algorithms */
typedef struct {
    /* Common */
    int32_t  *labels;        /* cluster assignment, length n_samples. -1 = noise (DBSCAN) */
    int32_t   n_samples;
    int32_t   n_features;
    int32_t   n_clusters;
    int32_t   algorithm;     /* CL_ALGO_* */
    int32_t   metric;

    /* K-Means / Mini-batch only */
    double   *centers;       /* k * n_features centroids (means). NULL otherwise */
    double    inertia;       /* within-cluster SSE. NAN if N/A */
    int32_t   n_iter;

    /* FastPAM only */
    int32_t  *medoid_indices;  /* k indices into original data. NULL otherwise */
    double   *medoid_coords;   /* k * n_features cached medoid coords. NULL otherwise */

    /* Hierarchical only */
    cl_merge_t *dendrogram;    /* (n_samples - 1) merges. NULL otherwise */

    /* DBSCAN only */
    uint8_t  *core_mask;       /* 1=core, 0=border/noise, length n_samples. NULL otherwise */
    int32_t   n_noise;

    uint32_t  seed;
} cl_result_t;

/* ========== Param initializers ========== */

void cl_kmeans_params_init(cl_kmeans_params_t *p);
void cl_minibatch_params_init(cl_minibatch_params_t *p);
void cl_dbscan_params_init(cl_dbscan_params_t *p);
void cl_hierarchical_params_init(cl_hierarchical_params_t *p);
void cl_fastpam_params_init(cl_fastpam_params_t *p);

/* ========== Fit ========== */

cl_result_t *cl_kmeans_fit(const double *X, int32_t nrow, int32_t ncol,
                            const cl_kmeans_params_t *params);

cl_result_t *cl_minibatch_fit(const double *X, int32_t nrow, int32_t ncol,
                               const cl_minibatch_params_t *params);

cl_result_t *cl_dbscan_fit(const double *X, int32_t nrow, int32_t ncol,
                             const cl_dbscan_params_t *params);

cl_result_t *cl_hierarchical_fit(const double *X, int32_t nrow, int32_t ncol,
                                   const cl_hierarchical_params_t *params);

cl_result_t *cl_fastpam_fit(const double *X, int32_t nrow, int32_t ncol,
                              const cl_fastpam_params_t *params);

/* ========== Predict (K-Means, Mini-batch, FastPAM only) ========== */

int cl_predict(const cl_result_t *result, const double *X, int32_t nrow, int32_t ncol,
               int32_t *out);

/* ========== Dendrogram utilities ========== */

int cl_dendro_cut_k(const cl_result_t *result, int32_t n_clusters, int32_t *out);

int32_t cl_dendro_cut_dist(const cl_result_t *result, double threshold, int32_t *out);

/* ========== Validation indices ========== */

/* Internal metrics. ignore_noise: 0=include, 1=exclude points with label -1 */
double cl_silhouette(const double *X, int32_t nrow, int32_t ncol,
                      const int32_t *labels, int32_t metric, int32_t ignore_noise);

int cl_silhouette_samples(const double *X, int32_t nrow, int32_t ncol,
                           const int32_t *labels, int32_t metric, int32_t ignore_noise,
                           double *out);

double cl_calinski_harabasz(const double *X, int32_t nrow, int32_t ncol,
                             const int32_t *labels);

double cl_davies_bouldin(const double *X, int32_t nrow, int32_t ncol,
                          const int32_t *labels);

/* External metric */
double cl_adjusted_rand(const int32_t *labels_true, const int32_t *labels_pred, int32_t n);

/* ========== Serialization ========== */

int cl_save(const cl_result_t *result, char **out_buf, int32_t *out_len);

cl_result_t *cl_load(const char *buf, int32_t len);

/* ========== Memory management ========== */

void cl_free(cl_result_t *result);
void cl_free_buffer(void *ptr);

/* ========== Error handling ========== */

const char *cl_get_error(void);

#endif /* CLUSTER_H */
