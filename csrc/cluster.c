/*
 * cluster.c -- Clustering library (C11, from scratch)
 *
 * K-Means, Mini-batch K-Means, DBSCAN, Hierarchical, FastPAM
 * + validation indices + serialization
 */

#define _POSIX_C_SOURCE 200809L

#include "cluster.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ========== Error handling ========== */

static _Thread_local char cl_error_buf[512] = "";

static void cl_set_error(const char *msg) {
    strncpy(cl_error_buf, msg, sizeof(cl_error_buf) - 1);
    cl_error_buf[sizeof(cl_error_buf) - 1] = '\0';
}

const char *cl_get_error(void) { return cl_error_buf; }

/* ========== PRNG (LCG, matches rf/gam) ========== */

static uint32_t cl_rng_next(uint32_t *state) {
    *state = (*state) * 1664525u + 1013904223u;
    return *state;
}

static double cl_rng_uniform(uint32_t *state) {
    return (cl_rng_next(state) & 0x7fffffffu) / (double)0x7fffffffu;
}

/* Random int in [0, n) */
static int32_t cl_rng_int(uint32_t *state, int32_t n) {
    return (int32_t)(cl_rng_uniform(state) * n);
}

/* ========== Param initializers ========== */

void cl_kmeans_params_init(cl_kmeans_params_t *p) {
    p->k = 3;
    p->max_iter = 300;
    p->n_init = 10;
    p->init = CL_INIT_KMPP;
    p->tol = 1e-4;
    p->seed = 42;
}

void cl_minibatch_params_init(cl_minibatch_params_t *p) {
    p->k = 3;
    p->max_iter = 100;
    p->n_init = 3;
    p->init = CL_INIT_KMPP;
    p->batch_size = 1024;
    p->tol = 0.0;
    p->seed = 42;
}

void cl_dbscan_params_init(cl_dbscan_params_t *p) {
    p->eps = 0.5;
    p->min_pts = 5;
    p->metric = CL_DIST_EUCLIDEAN;
}

void cl_hierarchical_params_init(cl_hierarchical_params_t *p) {
    p->n_clusters = 2;
    p->distance_threshold = 0.0;
    p->linkage = CL_LINK_WARD;
    p->metric = CL_DIST_EUCLIDEAN;
}

void cl_fastpam_params_init(cl_fastpam_params_t *p) {
    p->k = 3;
    p->max_iter = 100;
    p->init = CL_INIT_KMPP;
    p->metric = CL_DIST_EUCLIDEAN;
    p->seed = 42;
}

/* ========== Distance functions ========== */

static double dist_euclidean(const double *a, const double *b, int32_t d) {
    double s = 0.0;
    for (int32_t j = 0; j < d; j++) {
        double diff = a[j] - b[j];
        s += diff * diff;
    }
    return sqrt(s);
}

static double dist_euclidean_sq(const double *a, const double *b, int32_t d) {
    double s = 0.0;
    for (int32_t j = 0; j < d; j++) {
        double diff = a[j] - b[j];
        s += diff * diff;
    }
    return s;
}

static double dist_manhattan(const double *a, const double *b, int32_t d) {
    double s = 0.0;
    for (int32_t j = 0; j < d; j++) s += fabs(a[j] - b[j]);
    return s;
}

static double dist_cosine(const double *a, const double *b, int32_t d) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int32_t j = 0; j < d; j++) {
        dot += a[j] * b[j];
        na += a[j] * a[j];
        nb += b[j] * b[j];
    }
    double denom = sqrt(na) * sqrt(nb);
    if (denom < 1e-30) return 1.0;
    return 1.0 - dot / denom;
}

typedef double (*dist_fn_t)(const double *, const double *, int32_t);

static dist_fn_t get_dist_fn(int32_t metric) {
    switch (metric) {
        case CL_DIST_MANHATTAN: return dist_manhattan;
        case CL_DIST_COSINE:    return dist_cosine;
        default:                return dist_euclidean;
    }
}

/* ========== Result allocation / free ========== */

static cl_result_t *result_alloc(int32_t n, int32_t d, int32_t algo) {
    cl_result_t *r = (cl_result_t *)calloc(1, sizeof(cl_result_t));
    if (!r) return NULL;
    r->labels = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    if (!r->labels) { free(r); return NULL; }
    r->n_samples = n;
    r->n_features = d;
    r->algorithm = algo;
    r->inertia = NAN;
    r->n_noise = 0;
    r->n_iter = 0;
    return r;
}

void cl_free(cl_result_t *r) {
    if (!r) return;
    free(r->labels);
    free(r->centers);
    free(r->medoid_indices);
    free(r->medoid_coords);
    free(r->dendrogram);
    free(r->core_mask);
    free(r);
}

void cl_free_buffer(void *ptr) { free(ptr); }

/* ========== K-Means++ initialization ========== */

static void kmpp_init(const double *X, int32_t n, int32_t d, int32_t k,
                       double *centers, uint32_t *rng) {
    double *dists = (double *)malloc((size_t)n * sizeof(double));

    /* First center: random point */
    int32_t idx = cl_rng_int(rng, n);
    memcpy(centers, X + (size_t)idx * d, (size_t)d * sizeof(double));

    /* Initialize distances to first center */
    for (int32_t i = 0; i < n; i++)
        dists[i] = dist_euclidean_sq(X + (size_t)i * d, centers, d);

    for (int32_t c = 1; c < k; c++) {
        /* Cumulative distribution */
        double total = 0.0;
        for (int32_t i = 0; i < n; i++) total += dists[i];

        /* Sample proportional to D^2 */
        double r = cl_rng_uniform(rng) * total;
        double cum = 0.0;
        idx = n - 1;
        for (int32_t i = 0; i < n; i++) {
            cum += dists[i];
            if (cum >= r) { idx = i; break; }
        }

        memcpy(centers + (size_t)c * d, X + (size_t)idx * d, (size_t)d * sizeof(double));

        /* Update distances: min with new center */
        for (int32_t i = 0; i < n; i++) {
            double dd = dist_euclidean_sq(X + (size_t)i * d, centers + (size_t)c * d, d);
            if (dd < dists[i]) dists[i] = dd;
        }
    }

    free(dists);
}

static void random_init(const double *X, int32_t n, int32_t d, int32_t k,
                          double *centers, uint32_t *rng) {
    for (int32_t c = 0; c < k; c++) {
        int32_t idx = cl_rng_int(rng, n);
        memcpy(centers + (size_t)c * d, X + (size_t)idx * d, (size_t)d * sizeof(double));
    }
}

/* ========== K-Means (single run) ========== */

static double kmeans_single(const double *X, int32_t n, int32_t d, int32_t k,
                              int32_t max_iter, double tol, int32_t init_method,
                              uint32_t *rng,
                              int32_t *labels, double *centers, int32_t *out_iter) {
    /* Initialize centers */
    if (init_method == CL_INIT_RANDOM)
        random_init(X, n, d, k, centers, rng);
    else
        kmpp_init(X, n, d, k, centers, rng);

    int32_t *counts = (int32_t *)calloc((size_t)k, sizeof(int32_t));
    double *new_centers = (double *)calloc((size_t)k * d, sizeof(double));
    double inertia = 0.0;
    int32_t iter;

    for (iter = 0; iter < max_iter; iter++) {
        /* Assign points to nearest centroid */
        inertia = 0.0;
        for (int32_t i = 0; i < n; i++) {
            double best_dist = DBL_MAX;
            int32_t best_k = 0;
            for (int32_t c = 0; c < k; c++) {
                double dd = dist_euclidean_sq(X + (size_t)i * d, centers + (size_t)c * d, d);
                if (dd < best_dist) { best_dist = dd; best_k = c; }
            }
            labels[i] = best_k;
            inertia += best_dist;
        }

        /* Compute new centers */
        memset(new_centers, 0, (size_t)k * d * sizeof(double));
        memset(counts, 0, (size_t)k * sizeof(int32_t));
        for (int32_t i = 0; i < n; i++) {
            int32_t c = labels[i];
            counts[c]++;
            for (int32_t j = 0; j < d; j++)
                new_centers[(size_t)c * d + j] += X[(size_t)i * d + j];
        }
        for (int32_t c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int32_t j = 0; j < d; j++)
                    new_centers[(size_t)c * d + j] /= counts[c];
            }
        }

        /* Check convergence: max centroid shift */
        double max_shift = 0.0;
        for (int32_t c = 0; c < k; c++) {
            double shift = dist_euclidean_sq(centers + (size_t)c * d,
                                              new_centers + (size_t)c * d, d);
            if (shift > max_shift) max_shift = shift;
        }

        memcpy(centers, new_centers, (size_t)k * d * sizeof(double));

        if (max_shift <= tol * tol) { iter++; break; }
    }

    free(counts);
    free(new_centers);
    *out_iter = iter;
    return inertia;
}

/* ========== K-Means fit ========== */

cl_result_t *cl_kmeans_fit(const double *X, int32_t n, int32_t d,
                            const cl_kmeans_params_t *params) {
    if (!X || n <= 0 || d <= 0) { cl_set_error("invalid input"); return NULL; }
    int32_t k = params->k;
    if (k <= 0 || k > n) { cl_set_error("invalid k"); return NULL; }

    cl_result_t *r = result_alloc(n, d, CL_ALGO_KMEANS);
    if (!r) { cl_set_error("allocation failed"); return NULL; }
    r->metric = CL_DIST_EUCLIDEAN;
    r->seed = params->seed;

    r->centers = (double *)malloc((size_t)k * d * sizeof(double));
    if (!r->centers) { cl_free(r); cl_set_error("allocation failed"); return NULL; }

    double *tmp_centers = (double *)malloc((size_t)k * d * sizeof(double));
    int32_t *tmp_labels = (int32_t *)malloc((size_t)n * sizeof(int32_t));

    double best_inertia = DBL_MAX;
    int32_t best_iter = 0;
    uint32_t rng = params->seed;

    int32_t n_init = params->n_init > 0 ? params->n_init : 1;
    for (int32_t run = 0; run < n_init; run++) {
        int32_t iter;
        double inertia = kmeans_single(X, n, d, k, params->max_iter, params->tol,
                                        params->init, &rng, tmp_labels, tmp_centers, &iter);
        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_iter = iter;
            memcpy(r->labels, tmp_labels, (size_t)n * sizeof(int32_t));
            memcpy(r->centers, tmp_centers, (size_t)k * d * sizeof(double));
        }
    }

    free(tmp_centers);
    free(tmp_labels);

    r->n_clusters = k;
    r->inertia = best_inertia;
    r->n_iter = best_iter;
    return r;
}

/* ========== Mini-batch K-Means fit ========== */

cl_result_t *cl_minibatch_fit(const double *X, int32_t n, int32_t d,
                               const cl_minibatch_params_t *params) {
    if (!X || n <= 0 || d <= 0) { cl_set_error("invalid input"); return NULL; }
    int32_t k = params->k;
    if (k <= 0 || k > n) { cl_set_error("invalid k"); return NULL; }

    cl_result_t *r = result_alloc(n, d, CL_ALGO_MINIBATCH);
    if (!r) { cl_set_error("allocation failed"); return NULL; }
    r->metric = CL_DIST_EUCLIDEAN;
    r->seed = params->seed;

    r->centers = (double *)malloc((size_t)k * d * sizeof(double));
    if (!r->centers) { cl_free(r); cl_set_error("allocation failed"); return NULL; }

    uint32_t rng = params->seed;
    int32_t bs = params->batch_size > 0 ? params->batch_size : 1024;
    if (bs > n) bs = n;

    /* Initialize centers */
    if (params->init == CL_INIT_RANDOM)
        random_init(X, n, d, k, r->centers, &rng);
    else
        kmpp_init(X, n, d, k, r->centers, &rng);

    int32_t *counts = (int32_t *)calloc((size_t)k, sizeof(int32_t));
    int32_t *batch_idx = (int32_t *)malloc((size_t)bs * sizeof(int32_t));
    int32_t *batch_labels = (int32_t *)malloc((size_t)bs * sizeof(int32_t));

    int32_t iter;
    for (iter = 0; iter < params->max_iter; iter++) {
        /* Sample mini-batch */
        for (int32_t b = 0; b < bs; b++)
            batch_idx[b] = cl_rng_int(&rng, n);

        /* Assign batch to nearest centroid */
        for (int32_t b = 0; b < bs; b++) {
            int32_t idx = batch_idx[b];
            double best_dist = DBL_MAX;
            int32_t best_k = 0;
            for (int32_t c = 0; c < k; c++) {
                double dd = dist_euclidean_sq(X + (size_t)idx * d, r->centers + (size_t)c * d, d);
                if (dd < best_dist) { best_dist = dd; best_k = c; }
            }
            batch_labels[b] = best_k;
        }

        /* Update centers with running average */
        for (int32_t b = 0; b < bs; b++) {
            int32_t idx = batch_idx[b];
            int32_t c = batch_labels[b];
            counts[c]++;
            double lr = 1.0 / counts[c];
            for (int32_t j = 0; j < d; j++) {
                r->centers[(size_t)c * d + j] += lr * (X[(size_t)idx * d + j] - r->centers[(size_t)c * d + j]);
            }
        }
    }

    /* Final assignment of all points */
    double inertia = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double best_dist = DBL_MAX;
        int32_t best_k = 0;
        for (int32_t c = 0; c < k; c++) {
            double dd = dist_euclidean_sq(X + (size_t)i * d, r->centers + (size_t)c * d, d);
            if (dd < best_dist) { best_dist = dd; best_k = c; }
        }
        r->labels[i] = best_k;
        inertia += best_dist;
    }

    free(counts);
    free(batch_idx);
    free(batch_labels);

    r->n_clusters = k;
    r->inertia = inertia;
    r->n_iter = iter;
    return r;
}

/* ========== KD-tree (internal, for DBSCAN) ========== */

typedef struct kd_node {
    int32_t  split_dim;
    double   split_val;
    int32_t  point_idx;    /* -1 for internal nodes */
    struct kd_node *left, *right;
    /* Leaf: stores indices */
    int32_t *indices;
    int32_t  n_indices;
} kd_node_t;

#define KD_LEAF_SIZE 16

static kd_node_t *kd_build(const double *X, int32_t *idx, int32_t n, int32_t d, int32_t depth) {
    kd_node_t *node = (kd_node_t *)calloc(1, sizeof(kd_node_t));
    node->point_idx = -1;

    if (n <= KD_LEAF_SIZE) {
        node->indices = (int32_t *)malloc((size_t)n * sizeof(int32_t));
        memcpy(node->indices, idx, (size_t)n * sizeof(int32_t));
        node->n_indices = n;
        node->split_dim = -1;
        return node;
    }

    int32_t dim = depth % d;
    node->split_dim = dim;

    /* Find median via partial sort (simple selection) */
    /* Sort idx by X[idx][dim] -- insertion sort is fine for moderate n */
    for (int32_t i = 1; i < n; i++) {
        int32_t key = idx[i];
        double key_val = X[(size_t)key * d + dim];
        int32_t j = i - 1;
        while (j >= 0 && X[(size_t)idx[j] * d + dim] > key_val) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    int32_t mid = n / 2;
    node->split_val = X[(size_t)idx[mid] * d + dim];

    node->left = kd_build(X, idx, mid, d, depth + 1);
    node->right = kd_build(X, idx + mid, n - mid, d, depth + 1);
    return node;
}

static void kd_free(kd_node_t *node) {
    if (!node) return;
    kd_free(node->left);
    kd_free(node->right);
    free(node->indices);
    free(node);
}

/* Range query: find all points within eps distance */
static void kd_range_query(const kd_node_t *node, const double *X, int32_t d,
                             const double *query, double eps, int32_t metric,
                             int32_t *result, int32_t *count, int32_t max_count) {
    if (!node) return;

    /* Leaf node */
    if (node->split_dim < 0) {
        dist_fn_t dfn = get_dist_fn(metric);
        for (int32_t i = 0; i < node->n_indices; i++) {
            int32_t idx = node->indices[i];
            double dd = dfn(query, X + (size_t)idx * d, d);
            if (dd <= eps && *count < max_count) {
                result[(*count)++] = idx;
            }
        }
        return;
    }

    double diff = query[node->split_dim] - node->split_val;

    /* Visit closer side first */
    kd_node_t *first = diff <= 0 ? node->left : node->right;
    kd_node_t *second = diff <= 0 ? node->right : node->left;

    kd_range_query(first, X, d, query, eps, metric, result, count, max_count);

    /* Prune: can we skip the other side? */
    if (fabs(diff) <= eps) {
        kd_range_query(second, X, d, query, eps, metric, result, count, max_count);
    }
}

/* ========== DBSCAN fit ========== */

cl_result_t *cl_dbscan_fit(const double *X, int32_t n, int32_t d,
                             const cl_dbscan_params_t *params) {
    if (!X || n <= 0 || d <= 0) { cl_set_error("invalid input"); return NULL; }
    if (params->eps <= 0) { cl_set_error("eps must be > 0"); return NULL; }

    cl_result_t *r = result_alloc(n, d, CL_ALGO_DBSCAN);
    if (!r) { cl_set_error("allocation failed"); return NULL; }
    r->metric = params->metric;

    r->core_mask = (uint8_t *)calloc((size_t)n, sizeof(uint8_t));
    if (!r->core_mask) { cl_free(r); cl_set_error("allocation failed"); return NULL; }

    /* Initialize all labels to -1 (unvisited) */
    for (int32_t i = 0; i < n; i++) r->labels[i] = -1;

    /* Build KD-tree for Euclidean/Manhattan; brute-force for cosine */
    kd_node_t *tree = NULL;
    int32_t use_kdtree = (params->metric != CL_DIST_COSINE);
    if (use_kdtree) {
        int32_t *idx = (int32_t *)malloc((size_t)n * sizeof(int32_t));
        for (int32_t i = 0; i < n; i++) idx[i] = i;
        tree = kd_build(X, idx, n, d, 0);
        free(idx);
    }

    int32_t *neighbors = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *stack = (int32_t *)malloc((size_t)n * sizeof(int32_t));

    int32_t cluster_id = 0;
    dist_fn_t dfn = get_dist_fn(params->metric);

    for (int32_t i = 0; i < n; i++) {
        if (r->labels[i] != -1) continue;

        /* Find neighbors of i */
        int32_t nb_count = 0;
        if (use_kdtree) {
            kd_range_query(tree, X, d, X + (size_t)i * d, params->eps,
                            params->metric, neighbors, &nb_count, n);
        } else {
            for (int32_t j = 0; j < n; j++) {
                if (dfn(X + (size_t)i * d, X + (size_t)j * d, d) <= params->eps)
                    neighbors[nb_count++] = j;
            }
        }

        if (nb_count < params->min_pts) continue; /* noise for now */

        /* i is a core point, start new cluster */
        r->core_mask[i] = 1;
        r->labels[i] = cluster_id;

        /* Seed stack with neighbors */
        int32_t stack_top = 0;
        for (int32_t j = 0; j < nb_count; j++) {
            if (neighbors[j] != i) stack[stack_top++] = neighbors[j];
        }

        while (stack_top > 0) {
            int32_t q = stack[--stack_top];
            if (r->labels[q] == -1) {
                r->labels[q] = cluster_id;

                /* Find neighbors of q */
                int32_t q_nb_count = 0;
                int32_t *q_neighbors = (int32_t *)malloc((size_t)n * sizeof(int32_t));
                if (use_kdtree) {
                    kd_range_query(tree, X, d, X + (size_t)q * d, params->eps,
                                    params->metric, q_neighbors, &q_nb_count, n);
                } else {
                    for (int32_t j = 0; j < n; j++) {
                        if (dfn(X + (size_t)q * d, X + (size_t)j * d, d) <= params->eps)
                            q_neighbors[q_nb_count++] = j;
                    }
                }

                if (q_nb_count >= params->min_pts) {
                    r->core_mask[q] = 1;
                    for (int32_t j = 0; j < q_nb_count; j++) {
                        int32_t nb = q_neighbors[j];
                        if (r->labels[nb] == -1) {
                            stack[stack_top++] = nb;
                        }
                    }
                }
                free(q_neighbors);
            } else if (r->labels[q] >= 0) {
                /* Already assigned to a cluster -- skip */
            }
        }

        cluster_id++;
    }

    kd_free(tree);
    free(neighbors);
    free(stack);

    r->n_clusters = cluster_id;
    r->n_noise = 0;
    for (int32_t i = 0; i < n; i++) {
        if (r->labels[i] == -1) r->n_noise++;
    }
    return r;
}

/* ========== FastPAM fit ========== */

cl_result_t *cl_fastpam_fit(const double *X, int32_t n, int32_t d,
                              const cl_fastpam_params_t *params) {
    if (!X || n <= 0 || d <= 0) { cl_set_error("invalid input"); return NULL; }
    int32_t k = params->k;
    if (k <= 0 || k > n) { cl_set_error("invalid k"); return NULL; }

    dist_fn_t dfn = get_dist_fn(params->metric);

    cl_result_t *r = result_alloc(n, d, CL_ALGO_FASTPAM);
    if (!r) { cl_set_error("allocation failed"); return NULL; }
    r->metric = params->metric;
    r->seed = params->seed;

    r->medoid_indices = (int32_t *)malloc((size_t)k * sizeof(int32_t));
    r->medoid_coords = (double *)malloc((size_t)k * d * sizeof(double));
    if (!r->medoid_indices || !r->medoid_coords) {
        cl_free(r); cl_set_error("allocation failed"); return NULL;
    }

    /* BUILD phase: greedy medoid selection */
    /* First medoid: minimize total distance to all points */
    double *td_arr = (double *)calloc((size_t)n, sizeof(double));
    /* Compute sum of distances from each candidate to all others */
    {
        double best_td = DBL_MAX;
        int32_t best_idx = 0;
        for (int32_t c = 0; c < n; c++) {
            double total = 0.0;
            for (int32_t i = 0; i < n; i++)
                total += dfn(X + (size_t)c * d, X + (size_t)i * d, d);
            if (total < best_td) { best_td = total; best_idx = c; }
        }
        r->medoid_indices[0] = best_idx;
    }

    /* nearest[i] = distance to nearest medoid for point i */
    double *nearest = (double *)malloc((size_t)n * sizeof(double));
    /* second[i] = distance to second-nearest medoid */
    double *second_nearest = (double *)malloc((size_t)n * sizeof(double));
    /* assign[i] = index of nearest medoid (0..k-1) */
    int32_t *assign = (int32_t *)malloc((size_t)n * sizeof(int32_t));

    for (int32_t i = 0; i < n; i++) {
        nearest[i] = dfn(X + (size_t)i * d, X + (size_t)r->medoid_indices[0] * d, d);
        second_nearest[i] = DBL_MAX;
        assign[i] = 0;
    }

    /* Greedy selection of remaining medoids */
    for (int32_t m = 1; m < k; m++) {
        double best_gain = -DBL_MAX;
        int32_t best_idx = 0;
        for (int32_t c = 0; c < n; c++) {
            /* Skip if already a medoid */
            int skip = 0;
            for (int32_t j = 0; j < m; j++)
                if (r->medoid_indices[j] == c) { skip = 1; break; }
            if (skip) continue;

            double gain = 0.0;
            for (int32_t i = 0; i < n; i++) {
                double dc = dfn(X + (size_t)i * d, X + (size_t)c * d, d);
                if (dc < nearest[i]) gain += nearest[i] - dc;
            }
            if (gain > best_gain) { best_gain = gain; best_idx = c; }
        }
        r->medoid_indices[m] = best_idx;

        /* Update nearest/second-nearest */
        for (int32_t i = 0; i < n; i++) {
            double dc = dfn(X + (size_t)i * d, X + (size_t)best_idx * d, d);
            if (dc < nearest[i]) {
                second_nearest[i] = nearest[i];
                nearest[i] = dc;
                assign[i] = m;
            } else if (dc < second_nearest[i]) {
                second_nearest[i] = dc;
            }
        }
    }

    /* SWAP phase */
    for (int32_t iter = 0; iter < params->max_iter; iter++) {
        int improved = 0;

        for (int32_t m = 0; m < k; m++) {
            double best_delta = 0.0;
            int32_t best_swap = -1;

            for (int32_t c = 0; c < n; c++) {
                /* Skip existing medoids */
                int is_medoid = 0;
                for (int32_t j = 0; j < k; j++)
                    if (r->medoid_indices[j] == c) { is_medoid = 1; break; }
                if (is_medoid) continue;

                /* Compute delta TD for swapping medoid m with candidate c */
                double delta = 0.0;
                for (int32_t i = 0; i < n; i++) {
                    double dc = dfn(X + (size_t)i * d, X + (size_t)c * d, d);
                    if (assign[i] == m) {
                        /* Point's nearest medoid is being swapped out */
                        double new_nearest = fmin(dc, second_nearest[i]);
                        delta += new_nearest - nearest[i];
                    } else {
                        /* Point's nearest medoid stays */
                        if (dc < nearest[i]) {
                            delta += dc - nearest[i];
                        }
                    }
                }

                if (delta < best_delta) {
                    best_delta = delta;
                    best_swap = c;
                }
            }

            if (best_swap >= 0) {
                r->medoid_indices[m] = best_swap;
                improved = 1;

                /* Recompute nearest/second-nearest for all points */
                for (int32_t i = 0; i < n; i++) {
                    nearest[i] = DBL_MAX;
                    second_nearest[i] = DBL_MAX;
                    assign[i] = 0;
                    for (int32_t j = 0; j < k; j++) {
                        double dd = dfn(X + (size_t)i * d,
                                         X + (size_t)r->medoid_indices[j] * d, d);
                        if (dd < nearest[i]) {
                            second_nearest[i] = nearest[i];
                            nearest[i] = dd;
                            assign[i] = j;
                        } else if (dd < second_nearest[i]) {
                            second_nearest[i] = dd;
                        }
                    }
                }
                break; /* restart sweep after swap */
            }
        }

        r->n_iter = iter + 1;
        if (!improved) break;
    }

    /* Final assignment */
    double total_dev = 0.0;
    for (int32_t i = 0; i < n; i++) {
        r->labels[i] = assign[i];
        total_dev += nearest[i];
    }
    r->inertia = total_dev;
    r->n_clusters = k;

    /* Copy medoid coordinates */
    for (int32_t m = 0; m < k; m++) {
        memcpy(r->medoid_coords + (size_t)m * d,
               X + (size_t)r->medoid_indices[m] * d,
               (size_t)d * sizeof(double));
    }

    free(td_arr);
    free(nearest);
    free(second_nearest);
    free(assign);
    return r;
}

/* ========== Hierarchical fit ========== */

/* Condensed distance matrix index: i < j */
static inline size_t condensed_idx(int32_t i, int32_t j, int32_t n) {
    if (i > j) { int32_t tmp = i; i = j; j = tmp; }
    return (size_t)n * i - (size_t)i * (i + 1) / 2 + j - i - 1;
}

cl_result_t *cl_hierarchical_fit(const double *X, int32_t n, int32_t d,
                                   const cl_hierarchical_params_t *params) {
    if (!X || n <= 0 || d <= 0) { cl_set_error("invalid input"); return NULL; }
    if (n < 2) { cl_set_error("need at least 2 samples"); return NULL; }
    if (params->linkage == CL_LINK_WARD && params->metric != CL_DIST_EUCLIDEAN) {
        cl_set_error("Ward linkage requires Euclidean distance");
        return NULL;
    }

    /* Memory check: condensed matrix for n points */
    size_t cond_size = (size_t)n * (n - 1) / 2;
    if (cond_size > 50000000) { /* ~400MB limit */
        cl_set_error("too many samples for hierarchical clustering (n^2 memory)");
        return NULL;
    }

    dist_fn_t dfn = get_dist_fn(params->metric);

    cl_result_t *r = result_alloc(n, d, CL_ALGO_HIERARCHICAL);
    if (!r) { cl_set_error("allocation failed"); return NULL; }
    r->metric = params->metric;

    r->dendrogram = (cl_merge_t *)malloc((size_t)(n - 1) * sizeof(cl_merge_t));
    if (!r->dendrogram) { cl_free(r); cl_set_error("allocation failed"); return NULL; }

    /* Compute initial condensed distance matrix */
    double *dist_mat = (double *)malloc(cond_size * sizeof(double));
    if (!dist_mat) { cl_free(r); cl_set_error("allocation failed"); return NULL; }

    for (int32_t i = 0; i < n; i++) {
        for (int32_t j = i + 1; j < n; j++) {
            double dd;
            if (params->linkage == CL_LINK_WARD) {
                dd = dist_euclidean_sq(X + (size_t)i * d, X + (size_t)j * d, d);
            } else {
                dd = dfn(X + (size_t)i * d, X + (size_t)j * d, d);
            }
            dist_mat[condensed_idx(i, j, n)] = dd;
        }
    }

    /* Cluster sizes */
    int32_t *sizes = (int32_t *)malloc((size_t)(2 * n - 1) * sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) sizes[i] = 1;

    /* Active clusters: active[ii] is a condensed-matrix slot (0..n-1).
       cid[slot] maps slot -> current cluster ID (initially 0..n-1, then n+step after merges). */
    int32_t *active = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t *cid = (int32_t *)malloc((size_t)n * sizeof(int32_t));
    int32_t n_active = n;
    for (int32_t i = 0; i < n; i++) { active[i] = i; cid[i] = i; }

    for (int32_t step = 0; step < n - 1; step++) {
        /* Find minimum distance pair among active clusters */
        double min_dist = DBL_MAX;
        int32_t min_a = -1, min_b = -1;
        for (int32_t ii = 0; ii < n_active; ii++) {
            for (int32_t jj = ii + 1; jj < n_active; jj++) {
                int32_t si = active[ii], sj = active[jj];
                double dd = dist_mat[condensed_idx(si, sj, n)];
                if (dd < min_dist) {
                    min_dist = dd; min_a = ii; min_b = jj;
                }
            }
        }

        int32_t sa = active[min_a], sb = active[min_b]; /* condensed-matrix slots */
        int32_t id_a = cid[sa], id_b = cid[sb];          /* cluster IDs */
        int32_t new_cluster = n + step;
        sizes[new_cluster] = sizes[id_a] + sizes[id_b];

        double merge_dist = min_dist;
        if (params->linkage == CL_LINK_WARD) merge_dist = sqrt(2.0 * min_dist);

        r->dendrogram[step].left = id_a;
        r->dendrogram[step].right = id_b;
        r->dendrogram[step].distance = merge_dist;
        r->dendrogram[step].size = sizes[new_cluster];

        /* Update distances using Lance-Williams formula */
        for (int32_t ii = 0; ii < n_active; ii++) {
            int32_t sk = active[ii];
            if (sk == sa || sk == sb) continue;

            double da = dist_mat[condensed_idx(sa, sk, n)];
            double db = dist_mat[condensed_idx(sb, sk, n)];
            double new_d;

            switch (params->linkage) {
                case CL_LINK_SINGLE:
                    new_d = fmin(da, db);
                    break;
                case CL_LINK_COMPLETE:
                    new_d = fmax(da, db);
                    break;
                case CL_LINK_AVERAGE: {
                    double na = sizes[id_a], nb = sizes[id_b];
                    new_d = (na * da + nb * db) / (na + nb);
                    break;
                }
                case CL_LINK_WARD: {
                    int32_t id_k = cid[sk];
                    double na = sizes[id_a], nb = sizes[id_b], nk = sizes[id_k];
                    double dab = min_dist;
                    new_d = ((na + nk) * da + (nb + nk) * db - nk * dab) / (na + nb + nk);
                    break;
                }
                default:
                    new_d = fmin(da, db);
            }

            dist_mat[condensed_idx(sa, sk, n)] = new_d;
        }

        /* sa's slot now represents new_cluster */
        cid[sa] = new_cluster;

        /* Remove sb from active list */
        active[min_b] = active[n_active - 1];
        n_active--;
    }

    /* Cut dendrogram to get labels */
    int32_t n_clusters = params->n_clusters;
    if (params->distance_threshold > 0.0) {
        /* Cut by distance */
        n_clusters = cl_dendro_cut_dist(r, params->distance_threshold, r->labels);
    } else {
        if (n_clusters <= 0 || n_clusters > n) n_clusters = 2;
        cl_dendro_cut_k(r, n_clusters, r->labels);
    }
    r->n_clusters = n_clusters;

    free(dist_mat);
    free(sizes);
    free(active);
    free(cid);
    return r;
}

/* ========== Predict (K-Means, Mini-batch, FastPAM) ========== */

int cl_predict(const cl_result_t *r, const double *X, int32_t nrow, int32_t ncol,
               int32_t *out) {
    if (!r || !X || !out) { cl_set_error("null argument"); return -1; }
    if (ncol != r->n_features) { cl_set_error("feature count mismatch"); return -1; }

    const double *ref_centers = NULL;
    int32_t k = r->n_clusters;

    if (r->algorithm == CL_ALGO_KMEANS || r->algorithm == CL_ALGO_MINIBATCH) {
        ref_centers = r->centers;
    } else if (r->algorithm == CL_ALGO_FASTPAM) {
        ref_centers = r->medoid_coords;
    } else {
        cl_set_error("predict not supported for this algorithm");
        return -1;
    }

    if (!ref_centers) { cl_set_error("no centers available"); return -1; }

    dist_fn_t dfn = get_dist_fn(r->metric);
    int32_t d = ncol;

    for (int32_t i = 0; i < nrow; i++) {
        double best_dist = DBL_MAX;
        int32_t best_k = 0;
        for (int32_t c = 0; c < k; c++) {
            double dd = dfn(X + (size_t)i * d, ref_centers + (size_t)c * d, d);
            if (dd < best_dist) { best_dist = dd; best_k = c; }
        }
        out[i] = best_k;
    }
    return 0;
}

/* ========== Dendrogram utilities ========== */

int cl_dendro_cut_k(const cl_result_t *r, int32_t k, int32_t *out) {
    if (!r || !r->dendrogram || !out) { cl_set_error("invalid argument"); return -1; }
    int32_t n = r->n_samples;
    if (k <= 0 || k > n) { cl_set_error("invalid k"); return -1; }

    int32_t n_merges = n - 1;
    /* Labels for each node (0..2n-2) */
    int32_t *node_labels = (int32_t *)calloc((size_t)(2 * n - 1), sizeof(int32_t));

    /* Start: all nodes are their own cluster */
    for (int32_t i = 0; i < 2 * n - 1; i++) node_labels[i] = -1;

    /* Apply first (n - k) merges */
    int32_t merges_to_apply = n - k;
    for (int32_t i = 0; i < merges_to_apply; i++) {
        int32_t new_cluster = n + i;
        node_labels[new_cluster] = 0; /* mark as merged */
    }

    /* Assign cluster labels by traversing dendrogram */
    /* Each un-merged subtree root gets a unique label */
    int32_t label = 0;

    /* Walk from the latest merge backward to assign labels */
    /* A node is a "root" if it's not part of a later merge */
    int32_t *parent = (int32_t *)malloc((size_t)(2 * n - 1) * sizeof(int32_t));
    for (int32_t i = 0; i < 2 * n - 1; i++) parent[i] = -1;
    for (int32_t i = 0; i < n_merges; i++) {
        int32_t new_node = n + i;
        parent[r->dendrogram[i].left] = new_node;
        parent[r->dendrogram[i].right] = new_node;
    }

    /* Find the cut: apply first (n-k) merges, leave last (k-1) */
    /* Roots after cutting = nodes with no parent in the applied merges */
    /* Reset node_labels */
    for (int32_t i = 0; i < 2 * n - 1; i++) node_labels[i] = -1;

    /* Mark which merges are applied (first n-k) */
    /* The applied merges create nodes n..n+(n-k)-1 */
    /* Find roots: nodes that exist after applied merges but whose parent merge is NOT applied */
    label = 0;

    /* BFS/DFS from each root to assign leaf labels */
    /* A root is: a leaf (0..n-1) that has no parent in applied range,
       or an internal node (n..n+merges_to_apply-1) whose parent is NOT in applied range */
    int32_t *stack = (int32_t *)malloc((size_t)(2 * n) * sizeof(int32_t));
    for (int32_t node = 2 * n - 2; node >= 0; node--) {
        /* Skip if already labeled */
        if (node_labels[node] >= 0) continue;

        /* Is this node a root at the cut level? */
        int is_root = 0;
        if (node < n) {
            /* Leaf: root if parent merge index >= merges_to_apply (not applied) or no parent */
            if (parent[node] < 0) is_root = 1;
            else if (parent[node] - n >= merges_to_apply) is_root = 1;
        } else {
            int32_t merge_idx = node - n;
            if (merge_idx < merges_to_apply) {
                /* This internal node was created by an applied merge */
                /* It's a root if its parent merge is NOT applied */
                if (parent[node] < 0) is_root = 1;
                else if (parent[node] - n >= merges_to_apply) is_root = 1;
            }
        }

        if (!is_root) continue;

        /* DFS to assign this label to all descendant leaves */
        int32_t top = 0;
        stack[top++] = node;
        while (top > 0) {
            int32_t cur = stack[--top];
            if (cur < n) {
                /* Leaf */
                node_labels[cur] = label;
            } else {
                int32_t mi = cur - n;
                if (mi < n_merges) {
                    stack[top++] = r->dendrogram[mi].left;
                    stack[top++] = r->dendrogram[mi].right;
                }
            }
        }
        label++;
    }

    memcpy(out, node_labels, (size_t)n * sizeof(int32_t));

    free(node_labels);
    free(parent);
    free(stack);
    return 0;
}

int32_t cl_dendro_cut_dist(const cl_result_t *r, double threshold, int32_t *out) {
    if (!r || !r->dendrogram || !out) { cl_set_error("invalid argument"); return -1; }
    int32_t n = r->n_samples;

    /* Find how many merges to apply (those with distance <= threshold) */
    int32_t merges_to_apply = 0;
    for (int32_t i = 0; i < n - 1; i++) {
        if (r->dendrogram[i].distance <= threshold) merges_to_apply++;
        else break;
    }

    int32_t k = n - merges_to_apply;

    /* Use cl_dendro_cut_k with a temporary result that has n_clusters = k */
    cl_dendro_cut_k(r, k, out);
    return k;
}

/* ========== Validation: Silhouette ========== */

double cl_silhouette(const double *X, int32_t n, int32_t d,
                      const int32_t *labels, int32_t metric, int32_t ignore_noise) {
    double *sil = (double *)malloc((size_t)n * sizeof(double));
    if (!sil) return NAN;

    int ret = cl_silhouette_samples(X, n, d, labels, metric, ignore_noise, sil);
    if (ret != 0) { free(sil); return NAN; }

    double sum = 0.0;
    int32_t count = 0;
    for (int32_t i = 0; i < n; i++) {
        if (ignore_noise && labels[i] == -1) continue;
        if (!isnan(sil[i])) { sum += sil[i]; count++; }
    }
    free(sil);
    return count > 0 ? sum / count : NAN;
}

int cl_silhouette_samples(const double *X, int32_t n, int32_t d,
                           const int32_t *labels, int32_t metric, int32_t ignore_noise,
                           double *out) {
    if (!X || !labels || !out || n <= 0) { cl_set_error("invalid argument"); return -1; }

    dist_fn_t dfn = get_dist_fn(metric);

    /* Find max label */
    int32_t max_label = -1;
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] > max_label) max_label = labels[i];
    }
    if (max_label < 1) { cl_set_error("need at least 2 clusters"); return -1; }
    int32_t k = max_label + 1;

    int32_t *cluster_sizes = (int32_t *)calloc((size_t)k, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] >= 0) cluster_sizes[labels[i]]++;
    }

    for (int32_t i = 0; i < n; i++) {
        if (labels[i] < 0) { out[i] = NAN; continue; }

        /* Compute average distance to own cluster (a) and nearest other cluster (b) */
        double *cluster_dists = (double *)calloc((size_t)k, sizeof(double));
        for (int32_t j = 0; j < n; j++) {
            if (i == j) continue;
            if (ignore_noise && labels[j] == -1) continue;
            if (labels[j] < 0) continue;
            double dd = dfn(X + (size_t)i * d, X + (size_t)j * d, d);
            cluster_dists[labels[j]] += dd;
        }

        /* Use effective cluster sizes (may differ if ignore_noise changes counts) */
        int32_t own_cluster = labels[i];
        double a = (cluster_sizes[own_cluster] > 1)
            ? cluster_dists[own_cluster] / (cluster_sizes[own_cluster] - 1)
            : 0.0;

        double b = DBL_MAX;
        for (int32_t c = 0; c < k; c++) {
            if (c == own_cluster || cluster_sizes[c] == 0) continue;
            double avg = cluster_dists[c] / cluster_sizes[c];
            if (avg < b) b = avg;
        }

        free(cluster_dists);

        if (b == DBL_MAX) {
            out[i] = 0.0;
        } else {
            out[i] = (b - a) / fmax(a, b);
        }
    }

    free(cluster_sizes);
    return 0;
}

/* ========== Validation: Calinski-Harabasz ========== */

double cl_calinski_harabasz(const double *X, int32_t n, int32_t d,
                             const int32_t *labels) {
    if (!X || !labels || n <= 0 || d <= 0) return NAN;

    int32_t max_label = -1;
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] > max_label) max_label = labels[i];
    }
    if (max_label < 1) return NAN; /* need at least 2 clusters */
    int32_t k = max_label + 1;

    /* Global mean */
    double *global_mean = (double *)calloc((size_t)d, sizeof(double));
    int32_t n_valid = 0;
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] < 0) continue;
        n_valid++;
        for (int32_t j = 0; j < d; j++) global_mean[j] += X[(size_t)i * d + j];
    }
    for (int32_t j = 0; j < d; j++) global_mean[j] /= n_valid;

    /* Cluster means and sizes */
    double *cluster_means = (double *)calloc((size_t)k * d, sizeof(double));
    int32_t *cluster_sizes = (int32_t *)calloc((size_t)k, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] < 0) continue;
        int32_t c = labels[i];
        cluster_sizes[c]++;
        for (int32_t j = 0; j < d; j++)
            cluster_means[(size_t)c * d + j] += X[(size_t)i * d + j];
    }
    for (int32_t c = 0; c < k; c++) {
        if (cluster_sizes[c] > 0) {
            for (int32_t j = 0; j < d; j++)
                cluster_means[(size_t)c * d + j] /= cluster_sizes[c];
        }
    }

    /* Between-cluster dispersion (B) */
    double bgss = 0.0;
    for (int32_t c = 0; c < k; c++) {
        if (cluster_sizes[c] == 0) continue;
        bgss += cluster_sizes[c] * dist_euclidean_sq(cluster_means + (size_t)c * d, global_mean, d);
    }

    /* Within-cluster dispersion (W) */
    double wgss = 0.0;
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] < 0) continue;
        wgss += dist_euclidean_sq(X + (size_t)i * d, cluster_means + (size_t)labels[i] * d, d);
    }

    free(global_mean);
    free(cluster_means);
    free(cluster_sizes);

    if (wgss < 1e-30) return 0.0;
    return (bgss / (k - 1)) / (wgss / (n_valid - k));
}

/* ========== Validation: Davies-Bouldin ========== */

double cl_davies_bouldin(const double *X, int32_t n, int32_t d,
                          const int32_t *labels) {
    if (!X || !labels || n <= 0 || d <= 0) return NAN;

    int32_t max_label = -1;
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] > max_label) max_label = labels[i];
    }
    if (max_label < 1) return NAN;
    int32_t k = max_label + 1;

    /* Cluster means and sizes */
    double *means = (double *)calloc((size_t)k * d, sizeof(double));
    int32_t *sizes = (int32_t *)calloc((size_t)k, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] < 0) continue;
        int32_t c = labels[i];
        sizes[c]++;
        for (int32_t j = 0; j < d; j++) means[(size_t)c * d + j] += X[(size_t)i * d + j];
    }
    for (int32_t c = 0; c < k; c++) {
        if (sizes[c] > 0) {
            for (int32_t j = 0; j < d; j++) means[(size_t)c * d + j] /= sizes[c];
        }
    }

    /* Average intra-cluster distances (S_i) */
    double *scatter = (double *)calloc((size_t)k, sizeof(double));
    for (int32_t i = 0; i < n; i++) {
        if (labels[i] < 0) continue;
        int32_t c = labels[i];
        scatter[c] += dist_euclidean(X + (size_t)i * d, means + (size_t)c * d, d);
    }
    for (int32_t c = 0; c < k; c++) {
        if (sizes[c] > 0) scatter[c] /= sizes[c];
    }

    /* DB index: average of max R_ij for each cluster */
    double db_sum = 0.0;
    int32_t k_valid = 0;
    for (int32_t ci = 0; ci < k; ci++) {
        if (sizes[ci] == 0) continue;
        k_valid++;
        double max_r = 0.0;
        for (int32_t cj = 0; cj < k; cj++) {
            if (ci == cj || sizes[cj] == 0) continue;
            double d_ij = dist_euclidean(means + (size_t)ci * d, means + (size_t)cj * d, d);
            if (d_ij < 1e-30) continue;
            double r_ij = (scatter[ci] + scatter[cj]) / d_ij;
            if (r_ij > max_r) max_r = r_ij;
        }
        db_sum += max_r;
    }

    free(means);
    free(sizes);
    free(scatter);

    return k_valid > 0 ? db_sum / k_valid : NAN;
}

/* ========== Validation: Adjusted Rand Index ========== */

double cl_adjusted_rand(const int32_t *labels_true, const int32_t *labels_pred, int32_t n) {
    if (!labels_true || !labels_pred || n <= 0) return NAN;

    /* Find max labels */
    int32_t max_true = 0, max_pred = 0;
    for (int32_t i = 0; i < n; i++) {
        if (labels_true[i] > max_true) max_true = labels_true[i];
        if (labels_pred[i] > max_pred) max_pred = labels_pred[i];
    }
    int32_t n_true = max_true + 1;
    int32_t n_pred = max_pred + 1;

    /* Contingency table */
    int32_t *table = (int32_t *)calloc((size_t)n_true * n_pred, sizeof(int32_t));
    for (int32_t i = 0; i < n; i++) {
        if (labels_true[i] < 0 || labels_pred[i] < 0) continue;
        table[labels_true[i] * n_pred + labels_pred[i]]++;
    }

    /* Row and column sums */
    int64_t *row_sums = (int64_t *)calloc((size_t)n_true, sizeof(int64_t));
    int64_t *col_sums = (int64_t *)calloc((size_t)n_pred, sizeof(int64_t));
    for (int32_t i = 0; i < n_true; i++) {
        for (int32_t j = 0; j < n_pred; j++) {
            row_sums[i] += table[i * n_pred + j];
            col_sums[j] += table[i * n_pred + j];
        }
    }

    /* Compute index using combinatorial formula */
    /* sum of C(n_ij, 2) */
    int64_t sum_comb = 0;
    for (int32_t i = 0; i < n_true; i++) {
        for (int32_t j = 0; j < n_pred; j++) {
            int64_t nij = table[i * n_pred + j];
            sum_comb += nij * (nij - 1) / 2;
        }
    }

    int64_t sum_row_comb = 0;
    for (int32_t i = 0; i < n_true; i++)
        sum_row_comb += row_sums[i] * (row_sums[i] - 1) / 2;

    int64_t sum_col_comb = 0;
    for (int32_t j = 0; j < n_pred; j++)
        sum_col_comb += col_sums[j] * (col_sums[j] - 1) / 2;

    int64_t n_comb = (int64_t)n * (n - 1) / 2;

    double expected = (double)sum_row_comb * sum_col_comb / n_comb;
    double max_index = 0.5 * (sum_row_comb + sum_col_comb);
    double denom = max_index - expected;

    free(table);
    free(row_sums);
    free(col_sums);

    if (fabs(denom) < 1e-30) return 0.0;
    return (sum_comb - expected) / denom;
}

/* ========== Serialization ========== */

#define CL_MAGIC "CL01"
#define CL_VERSION 1

/* Flags bitmask */
#define CL_FLAG_HAS_CENTERS    0x01
#define CL_FLAG_HAS_DENDROGRAM 0x02
#define CL_FLAG_HAS_MEDOIDS    0x04
#define CL_FLAG_HAS_CORE_MASK  0x08

static void write_i32(char **p, int32_t v) { memcpy(*p, &v, 4); *p += 4; }
static void write_u32(char **p, uint32_t v) { memcpy(*p, &v, 4); *p += 4; }
static void write_f64(char **p, double v) { memcpy(*p, &v, 8); *p += 8; }
static int32_t read_i32(const char **p) { int32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static uint32_t read_u32(const char **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static double read_f64(const char **p) { double v; memcpy(&v, *p, 8); *p += 8; return v; }

int cl_save(const cl_result_t *r, char **out_buf, int32_t *out_len) {
    if (!r || !out_buf || !out_len) { cl_set_error("null argument"); return -1; }

    int32_t n = r->n_samples, d = r->n_features, k = r->n_clusters;

    uint32_t flags = 0;
    if (r->centers) flags |= CL_FLAG_HAS_CENTERS;
    if (r->dendrogram) flags |= CL_FLAG_HAS_DENDROGRAM;
    if (r->medoid_indices) flags |= CL_FLAG_HAS_MEDOIDS;
    if (r->core_mask) flags |= CL_FLAG_HAS_CORE_MASK;

    /* Calculate total size */
    size_t size = 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 8 + 4; /* header: 52 bytes */
    size += (size_t)n * 4; /* labels */
    if (flags & CL_FLAG_HAS_CENTERS)    size += (size_t)k * d * 8;
    if (flags & CL_FLAG_HAS_MEDOIDS)    size += (size_t)k * 4 + (size_t)k * d * 8;
    if (flags & CL_FLAG_HAS_DENDROGRAM) size += (size_t)(n - 1) * (4 + 4 + 8 + 4); /* 20 per merge */
    if (flags & CL_FLAG_HAS_CORE_MASK)  size += (size_t)n;

    char *buf = (char *)malloc(size);
    if (!buf) { cl_set_error("allocation failed"); return -1; }
    char *p = buf;

    /* Header */
    memcpy(p, CL_MAGIC, 4); p += 4;
    write_i32(&p, CL_VERSION);
    write_i32(&p, r->algorithm);
    write_i32(&p, r->metric);
    write_i32(&p, n);
    write_i32(&p, d);
    write_i32(&p, k);
    write_i32(&p, r->n_iter);
    write_i32(&p, r->n_noise);
    write_u32(&p, r->seed);
    write_f64(&p, r->inertia);
    write_u32(&p, flags);

    /* Labels */
    memcpy(p, r->labels, (size_t)n * 4); p += (size_t)n * 4;

    /* Centers */
    if (flags & CL_FLAG_HAS_CENTERS) {
        memcpy(p, r->centers, (size_t)k * d * 8); p += (size_t)k * d * 8;
    }

    /* Medoids */
    if (flags & CL_FLAG_HAS_MEDOIDS) {
        memcpy(p, r->medoid_indices, (size_t)k * 4); p += (size_t)k * 4;
        memcpy(p, r->medoid_coords, (size_t)k * d * 8); p += (size_t)k * d * 8;
    }

    /* Dendrogram */
    if (flags & CL_FLAG_HAS_DENDROGRAM) {
        for (int32_t i = 0; i < n - 1; i++) {
            write_i32(&p, r->dendrogram[i].left);
            write_i32(&p, r->dendrogram[i].right);
            write_f64(&p, r->dendrogram[i].distance);
            write_i32(&p, r->dendrogram[i].size);
        }
    }

    /* Core mask */
    if (flags & CL_FLAG_HAS_CORE_MASK) {
        memcpy(p, r->core_mask, (size_t)n); p += (size_t)n;
    }

    *out_buf = buf;
    *out_len = (int32_t)(p - buf);
    return 0;
}

cl_result_t *cl_load(const char *buf, int32_t len) {
    if (!buf || len < 52) { cl_set_error("invalid buffer"); return NULL; }
    const char *p = buf;

    /* Check magic */
    if (memcmp(p, CL_MAGIC, 4) != 0) { cl_set_error("invalid magic"); return NULL; }
    p += 4;

    int32_t version = read_i32(&p);
    if (version != CL_VERSION) { cl_set_error("unsupported version"); return NULL; }

    int32_t algorithm = read_i32(&p);
    int32_t metric = read_i32(&p);
    int32_t n = read_i32(&p);
    int32_t d = read_i32(&p);
    int32_t k = read_i32(&p);
    int32_t n_iter = read_i32(&p);
    int32_t n_noise = read_i32(&p);
    uint32_t seed = read_u32(&p);
    double inertia = read_f64(&p);
    uint32_t flags = read_u32(&p);

    cl_result_t *r = result_alloc(n, d, algorithm);
    if (!r) { cl_set_error("allocation failed"); return NULL; }
    r->metric = metric;
    r->n_clusters = k;
    r->n_iter = n_iter;
    r->n_noise = n_noise;
    r->seed = seed;
    r->inertia = inertia;

    /* Labels */
    memcpy(r->labels, p, (size_t)n * 4); p += (size_t)n * 4;

    /* Centers */
    if (flags & CL_FLAG_HAS_CENTERS) {
        r->centers = (double *)malloc((size_t)k * d * 8);
        memcpy(r->centers, p, (size_t)k * d * 8); p += (size_t)k * d * 8;
    }

    /* Medoids */
    if (flags & CL_FLAG_HAS_MEDOIDS) {
        r->medoid_indices = (int32_t *)malloc((size_t)k * 4);
        memcpy(r->medoid_indices, p, (size_t)k * 4); p += (size_t)k * 4;
        r->medoid_coords = (double *)malloc((size_t)k * d * 8);
        memcpy(r->medoid_coords, p, (size_t)k * d * 8); p += (size_t)k * d * 8;
    }

    /* Dendrogram */
    if (flags & CL_FLAG_HAS_DENDROGRAM) {
        r->dendrogram = (cl_merge_t *)malloc((size_t)(n - 1) * sizeof(cl_merge_t));
        for (int32_t i = 0; i < n - 1; i++) {
            r->dendrogram[i].left = read_i32(&p);
            r->dendrogram[i].right = read_i32(&p);
            r->dendrogram[i].distance = read_f64(&p);
            r->dendrogram[i].size = read_i32(&p);
        }
    }

    /* Core mask */
    if (flags & CL_FLAG_HAS_CORE_MASK) {
        r->core_mask = (uint8_t *)malloc((size_t)n);
        memcpy(r->core_mask, p, (size_t)n); p += (size_t)n;
    }

    return r;
}
