/*
 * test_c.c -- C tests for cluster library
 */

#include "cluster.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;
static int assertions = 0;

#define ASSERT(cond, msg) do { \
    assertions++; \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        return 0; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    assertions++; \
    if (fabs((a) - (b)) > (tol)) { \
        printf("  FAIL: %s: got %g, expected %g (line %d)\n", msg, (double)(a), (double)(b), __LINE__); \
        return 0; \
    } \
} while(0)

#define RUN_TEST(fn) do { \
    tests_run++; \
    printf("  %s...", #fn); \
    if (fn()) { tests_passed++; printf(" OK\n"); } \
    else printf("\n"); \
} while(0)

/*
 * Test data: 3 well-separated clusters in 2D
 *   Cluster 0: around (0, 0)
 *   Cluster 1: around (10, 10)
 *   Cluster 2: around (20, 0)
 */
static const double X_3clusters[] = {
    /* cluster 0 */
    0.1, 0.2,
    0.3, -0.1,
    -0.2, 0.3,
    0.0, 0.0,
    0.4, 0.1,
    /* cluster 1 */
    10.0, 10.1,
    10.2, 9.8,
    9.9, 10.3,
    10.1, 10.0,
    10.3, 9.9,
    /* cluster 2 */
    20.0, 0.1,
    19.8, -0.2,
    20.1, 0.3,
    20.3, 0.0,
    19.9, 0.2,
};
static const int N_3C = 15;
static const int D_3C = 2;

/* Ground truth labels for the 3-cluster data */
static const int32_t true_labels_3c[] = {0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2};

/* ========== K-Means tests ========== */

static int test_kmeans_basic(void) {
    cl_kmeans_params_t p;
    cl_kmeans_params_init(&p);
    p.k = 3;
    p.seed = 42;

    cl_result_t *r = cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "kmeans fit returned NULL");
    ASSERT(r->n_clusters == 3, "n_clusters should be 3");
    ASSERT(r->n_samples == N_3C, "n_samples");
    ASSERT(r->centers != NULL, "centers should not be NULL");
    ASSERT(r->algorithm == CL_ALGO_KMEANS, "algorithm");
    ASSERT(!isnan(r->inertia), "inertia should not be NAN");
    ASSERT(r->inertia >= 0, "inertia >= 0");
    ASSERT(r->inertia < 5.0, "inertia should be small for well-separated clusters");

    /* Check that points in each original group got the same label */
    for (int i = 1; i < 5; i++) ASSERT(r->labels[i] == r->labels[0], "cluster 0 consistent");
    for (int i = 6; i < 10; i++) ASSERT(r->labels[i] == r->labels[5], "cluster 1 consistent");
    for (int i = 11; i < 15; i++) ASSERT(r->labels[i] == r->labels[10], "cluster 2 consistent");

    /* Three distinct labels */
    ASSERT(r->labels[0] != r->labels[5], "clusters 0 and 1 differ");
    ASSERT(r->labels[0] != r->labels[10], "clusters 0 and 2 differ");
    ASSERT(r->labels[5] != r->labels[10], "clusters 1 and 2 differ");

    cl_free(r);
    return 1;
}

static int test_kmeans_inertia_decreases(void) {
    cl_kmeans_params_t p;
    cl_kmeans_params_init(&p);
    p.k = 3;
    p.n_init = 1;
    p.max_iter = 1;
    p.seed = 42;

    cl_result_t *r1 = cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r1 != NULL, "fit with 1 iter");

    p.max_iter = 100;
    cl_result_t *r2 = cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r2 != NULL, "fit with 100 iters");
    ASSERT(r2->inertia <= r1->inertia + 1e-10, "more iters should not increase inertia");

    cl_free(r1);
    cl_free(r2);
    return 1;
}

static int test_kmeans_predict(void) {
    cl_kmeans_params_t p;
    cl_kmeans_params_init(&p);
    p.k = 3;
    p.seed = 42;

    cl_result_t *r = cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    /* Predict on new points near each cluster */
    double X_new[] = {0.5, 0.5, 10.0, 10.0, 20.0, 0.0};
    int32_t pred[3];
    int ret = cl_predict(r, X_new, 3, D_3C, pred);
    ASSERT(ret == 0, "predict succeeded");

    /* Each prediction should match original cluster label */
    ASSERT(pred[0] == r->labels[0], "predict near cluster 0");
    ASSERT(pred[1] == r->labels[5], "predict near cluster 1");
    ASSERT(pred[2] == r->labels[10], "predict near cluster 2");

    cl_free(r);
    return 1;
}

static int test_kmeans_n_init(void) {
    cl_kmeans_params_t p;
    cl_kmeans_params_init(&p);
    p.k = 3;
    p.n_init = 1;
    p.seed = 123;

    cl_result_t *r1 = cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p);
    double inertia1 = r1->inertia;
    cl_free(r1);

    p.n_init = 10;
    cl_result_t *r10 = cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r10->inertia <= inertia1 + 1e-10, "n_init=10 should be at least as good as n_init=1");
    cl_free(r10);
    return 1;
}

static int test_kmeans_invalid(void) {
    cl_kmeans_params_t p;
    cl_kmeans_params_init(&p);
    p.k = 0;
    ASSERT(cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p) == NULL, "k=0 rejected");

    p.k = 100;
    ASSERT(cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p) == NULL, "k>n rejected");

    ASSERT(cl_kmeans_fit(NULL, N_3C, D_3C, &p) == NULL, "null X rejected");
    return 1;
}

/* ========== Mini-batch K-Means tests ========== */

static int test_minibatch_basic(void) {
    cl_minibatch_params_t p;
    cl_minibatch_params_init(&p);
    p.k = 3;
    p.max_iter = 100;
    p.batch_size = 5;
    p.seed = 42;

    cl_result_t *r = cl_minibatch_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "minibatch fit");
    ASSERT(r->n_clusters == 3, "n_clusters");
    ASSERT(r->centers != NULL, "centers");
    ASSERT(r->algorithm == CL_ALGO_MINIBATCH, "algorithm");

    /* Should recover the three clusters */
    for (int i = 1; i < 5; i++) ASSERT(r->labels[i] == r->labels[0], "cluster 0 consistent");
    for (int i = 6; i < 10; i++) ASSERT(r->labels[i] == r->labels[5], "cluster 1 consistent");
    for (int i = 11; i < 15; i++) ASSERT(r->labels[i] == r->labels[10], "cluster 2 consistent");

    cl_free(r);
    return 1;
}

static int test_minibatch_predict(void) {
    cl_minibatch_params_t p;
    cl_minibatch_params_init(&p);
    p.k = 3;
    p.seed = 42;

    cl_result_t *r = cl_minibatch_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    double X_new[] = {0.5, 0.5, 10.0, 10.0, 20.0, 0.0};
    int32_t pred[3];
    int ret = cl_predict(r, X_new, 3, D_3C, pred);
    ASSERT(ret == 0, "predict ok");
    ASSERT(pred[0] == r->labels[0], "predict cluster 0");
    ASSERT(pred[1] == r->labels[5], "predict cluster 1");
    ASSERT(pred[2] == r->labels[10], "predict cluster 2");

    cl_free(r);
    return 1;
}

/* ========== DBSCAN tests ========== */

static int test_dbscan_basic(void) {
    cl_dbscan_params_t p;
    cl_dbscan_params_init(&p);
    p.eps = 2.0;
    p.min_pts = 3;
    p.metric = CL_DIST_EUCLIDEAN;

    cl_result_t *r = cl_dbscan_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "dbscan fit");
    ASSERT(r->n_clusters == 3, "found 3 clusters");
    ASSERT(r->algorithm == CL_ALGO_DBSCAN, "algorithm");
    ASSERT(r->core_mask != NULL, "core_mask allocated");
    ASSERT(r->n_noise == 0, "no noise with eps=2.0");

    /* Same-group consistency */
    for (int i = 1; i < 5; i++) ASSERT(r->labels[i] == r->labels[0], "cluster 0");
    for (int i = 6; i < 10; i++) ASSERT(r->labels[i] == r->labels[5], "cluster 1");
    for (int i = 11; i < 15; i++) ASSERT(r->labels[i] == r->labels[10], "cluster 2");

    cl_free(r);
    return 1;
}

static int test_dbscan_noise(void) {
    /* Add isolated noise points */
    double X_noise[] = {
        0.0, 0.0,
        0.1, 0.1,
        0.2, 0.0,
        10.0, 10.0,
        10.1, 10.1,
        10.2, 10.0,
        50.0, 50.0,  /* noise */
    };
    cl_dbscan_params_t p;
    cl_dbscan_params_init(&p);
    p.eps = 1.0;
    p.min_pts = 3;

    cl_result_t *r = cl_dbscan_fit(X_noise, 7, 2, &p);
    ASSERT(r != NULL, "fit");
    ASSERT(r->n_clusters == 2, "2 clusters");
    ASSERT(r->n_noise == 1, "1 noise point");
    ASSERT(r->labels[6] == -1, "last point is noise");
    ASSERT(r->core_mask[6] == 0, "noise not core");

    cl_free(r);
    return 1;
}

static int test_dbscan_predict_rejected(void) {
    cl_dbscan_params_t p;
    cl_dbscan_params_init(&p);
    p.eps = 2.0;
    p.min_pts = 3;

    cl_result_t *r = cl_dbscan_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    int32_t pred[1];
    double X_new[] = {0.0, 0.0};
    int ret = cl_predict(r, X_new, 1, D_3C, pred);
    ASSERT(ret == -1, "predict rejected for DBSCAN");

    cl_free(r);
    return 1;
}

static int test_dbscan_cosine(void) {
    /* Points that differ by direction, not magnitude */
    double X_cos[] = {
        1.0, 0.0,
        0.9, 0.1,
        0.95, 0.05,
        0.0, 1.0,
        0.1, 0.9,
        0.05, 0.95,
    };
    cl_dbscan_params_t p;
    cl_dbscan_params_init(&p);
    p.eps = 0.1;
    p.min_pts = 2;
    p.metric = CL_DIST_COSINE;

    cl_result_t *r = cl_dbscan_fit(X_cos, 6, 2, &p);
    ASSERT(r != NULL, "fit");
    ASSERT(r->n_clusters == 2, "2 directional clusters");

    /* Same direction -> same cluster */
    ASSERT(r->labels[0] == r->labels[1], "direction 0");
    ASSERT(r->labels[0] == r->labels[2], "direction 0b");
    ASSERT(r->labels[3] == r->labels[4], "direction 1");
    ASSERT(r->labels[3] == r->labels[5], "direction 1b");
    ASSERT(r->labels[0] != r->labels[3], "directions differ");

    cl_free(r);
    return 1;
}

/* ========== FastPAM tests ========== */

static int test_fastpam_basic(void) {
    cl_fastpam_params_t p;
    cl_fastpam_params_init(&p);
    p.k = 3;
    p.seed = 42;

    cl_result_t *r = cl_fastpam_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fastpam fit");
    ASSERT(r->n_clusters == 3, "n_clusters");
    ASSERT(r->algorithm == CL_ALGO_FASTPAM, "algorithm");
    ASSERT(r->medoid_indices != NULL, "medoid_indices allocated");
    ASSERT(r->medoid_coords != NULL, "medoid_coords allocated");
    ASSERT(r->centers == NULL, "no centroids for PAM");

    /* Medoid indices should be valid points */
    for (int i = 0; i < 3; i++) {
        ASSERT(r->medoid_indices[i] >= 0 && r->medoid_indices[i] < N_3C,
               "medoid index valid");
    }

    /* Medoid coords should match the data at those indices */
    for (int m = 0; m < 3; m++) {
        int idx = r->medoid_indices[m];
        for (int j = 0; j < D_3C; j++) {
            ASSERT_NEAR(r->medoid_coords[m * D_3C + j],
                        X_3clusters[idx * D_3C + j], 1e-10, "medoid coord match");
        }
    }

    /* Cluster consistency */
    for (int i = 1; i < 5; i++) ASSERT(r->labels[i] == r->labels[0], "cluster 0");
    for (int i = 6; i < 10; i++) ASSERT(r->labels[i] == r->labels[5], "cluster 1");
    for (int i = 11; i < 15; i++) ASSERT(r->labels[i] == r->labels[10], "cluster 2");

    cl_free(r);
    return 1;
}

static int test_fastpam_predict(void) {
    cl_fastpam_params_t p;
    cl_fastpam_params_init(&p);
    p.k = 3;
    p.seed = 42;

    cl_result_t *r = cl_fastpam_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    double X_new[] = {0.5, 0.5, 10.0, 10.0, 20.0, 0.0};
    int32_t pred[3];
    int ret = cl_predict(r, X_new, 3, D_3C, pred);
    ASSERT(ret == 0, "predict ok");
    ASSERT(pred[0] == r->labels[0], "predict cluster 0");
    ASSERT(pred[1] == r->labels[5], "predict cluster 1");
    ASSERT(pred[2] == r->labels[10], "predict cluster 2");

    cl_free(r);
    return 1;
}

static int test_fastpam_td_never_increases(void) {
    cl_fastpam_params_t p;
    cl_fastpam_params_init(&p);
    p.k = 3;
    p.max_iter = 1;
    p.seed = 42;

    cl_result_t *r1 = cl_fastpam_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r1 != NULL, "fit 1 iter");

    p.max_iter = 100;
    cl_result_t *r100 = cl_fastpam_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r100 != NULL, "fit 100 iters");

    /* Total deviation should not increase with more iterations */
    ASSERT(r100->inertia <= r1->inertia + 1e-10, "TD never increases");

    cl_free(r1);
    cl_free(r100);
    return 1;
}

static int test_fastpam_manhattan(void) {
    cl_fastpam_params_t p;
    cl_fastpam_params_init(&p);
    p.k = 3;
    p.metric = CL_DIST_MANHATTAN;
    p.seed = 42;

    cl_result_t *r = cl_fastpam_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit with manhattan");
    ASSERT(r->metric == CL_DIST_MANHATTAN, "metric stored");
    ASSERT(r->n_clusters == 3, "3 clusters");

    cl_free(r);
    return 1;
}

/* ========== Hierarchical tests ========== */

static int test_hierarchical_ward(void) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.n_clusters = 3;
    p.linkage = CL_LINK_WARD;

    cl_result_t *r = cl_hierarchical_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "hierarchical fit");
    ASSERT(r->n_clusters == 3, "3 clusters");
    ASSERT(r->algorithm == CL_ALGO_HIERARCHICAL, "algorithm");
    ASSERT(r->dendrogram != NULL, "dendrogram allocated");

    /* Cluster consistency */
    for (int i = 1; i < 5; i++) ASSERT(r->labels[i] == r->labels[0], "cluster 0");
    for (int i = 6; i < 10; i++) ASSERT(r->labels[i] == r->labels[5], "cluster 1");
    for (int i = 11; i < 15; i++) ASSERT(r->labels[i] == r->labels[10], "cluster 2");

    cl_free(r);
    return 1;
}

static int test_hierarchical_single(void) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.n_clusters = 3;
    p.linkage = CL_LINK_SINGLE;

    cl_result_t *r = cl_hierarchical_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");
    ASSERT(r->n_clusters == 3, "3 clusters");

    cl_free(r);
    return 1;
}

static int test_hierarchical_complete(void) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.n_clusters = 3;
    p.linkage = CL_LINK_COMPLETE;

    cl_result_t *r = cl_hierarchical_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");
    ASSERT(r->n_clusters == 3, "3 clusters");

    cl_free(r);
    return 1;
}

static int test_hierarchical_dendrogram_monotonic(void) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.n_clusters = 1;
    p.linkage = CL_LINK_WARD;

    cl_result_t *r = cl_hierarchical_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");
    ASSERT(r->dendrogram != NULL, "dendrogram");

    /* Merge distances should be non-decreasing */
    for (int i = 1; i < N_3C - 1; i++) {
        ASSERT(r->dendrogram[i].distance >= r->dendrogram[i-1].distance - 1e-10,
               "monotonic merge distances");
    }

    /* All merge sizes should be positive and sum correctly */
    for (int i = 0; i < N_3C - 1; i++) {
        ASSERT(r->dendrogram[i].size >= 2, "merge size >= 2");
    }
    ASSERT(r->dendrogram[N_3C - 2].size == N_3C, "last merge has all points");

    cl_free(r);
    return 1;
}

static int test_hierarchical_ward_rejects_cosine(void) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.linkage = CL_LINK_WARD;
    p.metric = CL_DIST_COSINE;
    p.n_clusters = 3;

    cl_result_t *r = cl_hierarchical_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r == NULL, "Ward + cosine rejected");
    return 1;
}

static int test_hierarchical_cut_k(void) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.n_clusters = 1;  /* full dendrogram */
    p.linkage = CL_LINK_WARD;

    cl_result_t *r = cl_hierarchical_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    int32_t labels_k2[15];
    int ret = cl_dendro_cut_k(r, 2, labels_k2);
    ASSERT(ret == 0, "cut_k succeeded");

    /* With k=2, the two closest clusters should merge */
    /* Count distinct labels */
    int32_t max_l = -1;
    for (int i = 0; i < N_3C; i++) {
        if (labels_k2[i] > max_l) max_l = labels_k2[i];
    }
    ASSERT(max_l >= 1, "at least 2 clusters for k=2");

    int32_t labels_k3[15];
    ret = cl_dendro_cut_k(r, 3, labels_k3);
    ASSERT(ret == 0, "cut_k=3");

    cl_free(r);
    return 1;
}

/* ========== Validation tests ========== */

static int test_silhouette_perfect(void) {
    double sil = cl_silhouette(X_3clusters, N_3C, D_3C, true_labels_3c,
                                CL_DIST_EUCLIDEAN, 0);
    ASSERT(!isnan(sil), "silhouette not NAN");
    ASSERT(sil > 0.8, "well-separated clusters have high silhouette");
    ASSERT(sil <= 1.0, "silhouette <= 1");
    return 1;
}

static int test_silhouette_samples(void) {
    double sil[15];
    int ret = cl_silhouette_samples(X_3clusters, N_3C, D_3C, true_labels_3c,
                                     CL_DIST_EUCLIDEAN, 0, sil);
    ASSERT(ret == 0, "silhouette_samples ok");
    for (int i = 0; i < N_3C; i++) {
        ASSERT(sil[i] >= -1.0 && sil[i] <= 1.0, "silhouette in [-1, 1]");
    }
    return 1;
}

static int test_silhouette_with_noise(void) {
    int32_t labels_with_noise[] = {0,0,0,0,0, 1,1,1,1,1, -1,-1,-1,-1,-1};

    double sil = cl_silhouette(X_3clusters, N_3C, D_3C, labels_with_noise,
                                CL_DIST_EUCLIDEAN, 1);
    ASSERT(!isnan(sil), "silhouette with noise not NAN");
    ASSERT(sil > 0.0, "silhouette positive for separated clusters");

    /* Per-sample: noise points should get NAN */
    double samples[15];
    int ret = cl_silhouette_samples(X_3clusters, N_3C, D_3C, labels_with_noise,
                                     CL_DIST_EUCLIDEAN, 1, samples);
    ASSERT(ret == 0, "silhouette_samples ok");
    for (int i = 10; i < 15; i++) {
        ASSERT(isnan(samples[i]), "noise points get NAN silhouette");
    }
    for (int i = 0; i < 10; i++) {
        ASSERT(!isnan(samples[i]), "non-noise points get valid silhouette");
    }
    return 1;
}

static int test_calinski_harabasz(void) {
    double ch = cl_calinski_harabasz(X_3clusters, N_3C, D_3C, true_labels_3c);
    ASSERT(!isnan(ch), "CH not NAN");
    ASSERT(ch > 100.0, "well-separated clusters have high CH");
    return 1;
}

static int test_davies_bouldin(void) {
    double db = cl_davies_bouldin(X_3clusters, N_3C, D_3C, true_labels_3c);
    ASSERT(!isnan(db), "DB not NAN");
    ASSERT(db >= 0.0, "DB >= 0");
    ASSERT(db < 0.5, "well-separated clusters have low DB");
    return 1;
}

static int test_adjusted_rand_perfect(void) {
    double ari = cl_adjusted_rand(true_labels_3c, true_labels_3c, N_3C);
    ASSERT_NEAR(ari, 1.0, 1e-10, "ARI of identical labels is 1.0");
    return 1;
}

static int test_adjusted_rand_symmetric(void) {
    int32_t pred[] = {1,1,1,1,1, 2,2,2,2,2, 0,0,0,0,0};
    double ari1 = cl_adjusted_rand(true_labels_3c, pred, N_3C);
    double ari2 = cl_adjusted_rand(pred, true_labels_3c, N_3C);
    ASSERT_NEAR(ari1, ari2, 1e-10, "ARI is symmetric");
    /* Permuted labels should still give ARI = 1.0 */
    ASSERT_NEAR(ari1, 1.0, 1e-10, "permuted labels ARI = 1");
    return 1;
}

static int test_adjusted_rand_random(void) {
    int32_t random_labels[] = {0,1,2,0,1, 2,0,1,2,0, 1,2,0,1,2};
    double ari = cl_adjusted_rand(true_labels_3c, random_labels, N_3C);
    /* ARI for random labels should be close to 0 */
    ASSERT(ari < 0.5, "random labels have low ARI");
    return 1;
}

/* ========== Serialization tests ========== */

static int test_save_load_kmeans(void) {
    cl_kmeans_params_t p;
    cl_kmeans_params_init(&p);
    p.k = 3;
    p.seed = 42;

    cl_result_t *r = cl_kmeans_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    char *buf = NULL;
    int32_t len = 0;
    int ret = cl_save(r, &buf, &len);
    ASSERT(ret == 0, "save ok");
    ASSERT(buf != NULL, "buf not null");
    ASSERT(len > 0, "len > 0");

    /* Check magic */
    ASSERT(memcmp(buf, "CL01", 4) == 0, "magic CL01");

    cl_result_t *r2 = cl_load(buf, len);
    ASSERT(r2 != NULL, "load ok");
    ASSERT(r2->n_samples == r->n_samples, "n_samples match");
    ASSERT(r2->n_features == r->n_features, "n_features match");
    ASSERT(r2->n_clusters == r->n_clusters, "n_clusters match");
    ASSERT(r2->algorithm == CL_ALGO_KMEANS, "algorithm match");
    ASSERT_NEAR(r2->inertia, r->inertia, 1e-10, "inertia match");

    /* Labels match */
    for (int i = 0; i < N_3C; i++) {
        ASSERT(r2->labels[i] == r->labels[i], "labels match");
    }

    /* Centers match */
    ASSERT(r2->centers != NULL, "loaded centers");
    for (int i = 0; i < 3 * D_3C; i++) {
        ASSERT_NEAR(r2->centers[i], r->centers[i], 1e-10, "centers match");
    }

    /* Predict with loaded model */
    double X_new[] = {0.5, 0.5};
    int32_t pred1[1], pred2[1];
    cl_predict(r, X_new, 1, D_3C, pred1);
    cl_predict(r2, X_new, 1, D_3C, pred2);
    ASSERT(pred1[0] == pred2[0], "predict matches after load");

    cl_free(r);
    cl_free(r2);
    cl_free_buffer(buf);
    return 1;
}

static int test_save_load_dbscan(void) {
    double X_noise[] = {
        0.0, 0.0, 0.1, 0.1, 0.2, 0.0,
        10.0, 10.0, 10.1, 10.1, 10.2, 10.0,
        50.0, 50.0,
    };
    cl_dbscan_params_t p;
    cl_dbscan_params_init(&p);
    p.eps = 1.0;
    p.min_pts = 3;

    cl_result_t *r = cl_dbscan_fit(X_noise, 7, 2, &p);
    ASSERT(r != NULL, "fit");

    char *buf = NULL;
    int32_t len = 0;
    int ret = cl_save(r, &buf, &len);
    ASSERT(ret == 0, "save");

    cl_result_t *r2 = cl_load(buf, len);
    ASSERT(r2 != NULL, "load");
    ASSERT(r2->n_noise == r->n_noise, "n_noise match");
    ASSERT(r2->core_mask != NULL, "core_mask loaded");

    for (int i = 0; i < 7; i++) {
        ASSERT(r2->labels[i] == r->labels[i], "labels match");
        ASSERT(r2->core_mask[i] == r->core_mask[i], "core_mask match");
    }

    cl_free(r);
    cl_free(r2);
    cl_free_buffer(buf);
    return 1;
}

static int test_save_load_hierarchical(void) {
    cl_hierarchical_params_t p;
    cl_hierarchical_params_init(&p);
    p.n_clusters = 3;
    p.linkage = CL_LINK_WARD;

    cl_result_t *r = cl_hierarchical_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    char *buf = NULL;
    int32_t len = 0;
    cl_save(r, &buf, &len);

    cl_result_t *r2 = cl_load(buf, len);
    ASSERT(r2 != NULL, "load");
    ASSERT(r2->dendrogram != NULL, "dendrogram loaded");

    for (int i = 0; i < N_3C - 1; i++) {
        ASSERT(r2->dendrogram[i].left == r->dendrogram[i].left, "left match");
        ASSERT(r2->dendrogram[i].right == r->dendrogram[i].right, "right match");
        ASSERT_NEAR(r2->dendrogram[i].distance, r->dendrogram[i].distance, 1e-10, "distance match");
        ASSERT(r2->dendrogram[i].size == r->dendrogram[i].size, "size match");
    }

    cl_free(r);
    cl_free(r2);
    cl_free_buffer(buf);
    return 1;
}

static int test_save_load_fastpam(void) {
    cl_fastpam_params_t p;
    cl_fastpam_params_init(&p);
    p.k = 3;
    p.seed = 42;

    cl_result_t *r = cl_fastpam_fit(X_3clusters, N_3C, D_3C, &p);
    ASSERT(r != NULL, "fit");

    char *buf = NULL;
    int32_t len = 0;
    cl_save(r, &buf, &len);

    cl_result_t *r2 = cl_load(buf, len);
    ASSERT(r2 != NULL, "load");
    ASSERT(r2->medoid_indices != NULL, "medoid_indices loaded");
    ASSERT(r2->medoid_coords != NULL, "medoid_coords loaded");

    for (int m = 0; m < 3; m++) {
        ASSERT(r2->medoid_indices[m] == r->medoid_indices[m], "medoid idx match");
        for (int j = 0; j < D_3C; j++) {
            ASSERT_NEAR(r2->medoid_coords[m * D_3C + j],
                        r->medoid_coords[m * D_3C + j], 1e-10, "medoid coord match");
        }
    }

    /* Predict with loaded model */
    double X_new[] = {0.5, 0.5};
    int32_t pred1[1], pred2[1];
    cl_predict(r, X_new, 1, D_3C, pred1);
    cl_predict(r2, X_new, 1, D_3C, pred2);
    ASSERT(pred1[0] == pred2[0], "predict matches after load");

    cl_free(r);
    cl_free(r2);
    cl_free_buffer(buf);
    return 1;
}

/* ========== Main ========== */

int main(void) {
    printf("K-Means:\n");
    RUN_TEST(test_kmeans_basic);
    RUN_TEST(test_kmeans_inertia_decreases);
    RUN_TEST(test_kmeans_predict);
    RUN_TEST(test_kmeans_n_init);
    RUN_TEST(test_kmeans_invalid);

    printf("\nMini-batch K-Means:\n");
    RUN_TEST(test_minibatch_basic);
    RUN_TEST(test_minibatch_predict);

    printf("\nDBSCAN:\n");
    RUN_TEST(test_dbscan_basic);
    RUN_TEST(test_dbscan_noise);
    RUN_TEST(test_dbscan_predict_rejected);
    RUN_TEST(test_dbscan_cosine);

    printf("\nFastPAM:\n");
    RUN_TEST(test_fastpam_basic);
    RUN_TEST(test_fastpam_predict);
    RUN_TEST(test_fastpam_td_never_increases);
    RUN_TEST(test_fastpam_manhattan);

    printf("\nHierarchical:\n");
    RUN_TEST(test_hierarchical_ward);
    RUN_TEST(test_hierarchical_single);
    RUN_TEST(test_hierarchical_complete);
    RUN_TEST(test_hierarchical_dendrogram_monotonic);
    RUN_TEST(test_hierarchical_ward_rejects_cosine);
    RUN_TEST(test_hierarchical_cut_k);

    printf("\nValidation:\n");
    RUN_TEST(test_silhouette_perfect);
    RUN_TEST(test_silhouette_samples);
    RUN_TEST(test_silhouette_with_noise);
    RUN_TEST(test_calinski_harabasz);
    RUN_TEST(test_davies_bouldin);
    RUN_TEST(test_adjusted_rand_perfect);
    RUN_TEST(test_adjusted_rand_symmetric);
    RUN_TEST(test_adjusted_rand_random);

    printf("\nSerialization:\n");
    RUN_TEST(test_save_load_kmeans);
    RUN_TEST(test_save_load_dbscan);
    RUN_TEST(test_save_load_hierarchical);
    RUN_TEST(test_save_load_fastpam);

    printf("\n%d/%d tests passed (%d assertions)\n", tests_passed, tests_run, assertions);
    return tests_passed == tests_run ? 0 : 1;
}
