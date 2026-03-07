"""
test_python.py -- Python ctypes tests for cluster library

Tests: K-Means, DBSCAN, FastPAM, Hierarchical, validation indices, serialization.
Parity tests vs sklearn where applicable.
"""

import ctypes
import os
import sys
import numpy as np
from pathlib import Path

# Load shared library
LIB_DIR = Path(__file__).parent.parent / 'build'
lib_path = LIB_DIR / 'libcluster.so'
if not lib_path.exists():
    print(f'ERROR: {lib_path} not found. Build first: cd build && cmake .. -DBUILD_TESTING=ON && make')
    sys.exit(1)

lib = ctypes.CDLL(str(lib_path))

# Constants
CL_DIST_EUCLIDEAN = 0
CL_DIST_MANHATTAN = 1
CL_DIST_COSINE = 2
CL_INIT_KMPP = 0
CL_LINK_SINGLE = 0
CL_LINK_COMPLETE = 1
CL_LINK_AVERAGE = 2
CL_LINK_WARD = 3

# Param structs
class KMeansParams(ctypes.Structure):
    _fields_ = [
        ('k', ctypes.c_int32),
        ('max_iter', ctypes.c_int32),
        ('n_init', ctypes.c_int32),
        ('init', ctypes.c_int32),
        ('tol', ctypes.c_double),
        ('seed', ctypes.c_uint32),
    ]

class DBSCANParams(ctypes.Structure):
    _fields_ = [
        ('eps', ctypes.c_double),
        ('min_pts', ctypes.c_int32),
        ('metric', ctypes.c_int32),
    ]

class HierarchicalParams(ctypes.Structure):
    _fields_ = [
        ('n_clusters', ctypes.c_int32),
        ('distance_threshold', ctypes.c_double),
        ('linkage', ctypes.c_int32),
        ('metric', ctypes.c_int32),
    ]

class FastPAMParams(ctypes.Structure):
    _fields_ = [
        ('k', ctypes.c_int32),
        ('max_iter', ctypes.c_int32),
        ('init', ctypes.c_int32),
        ('metric', ctypes.c_int32),
        ('seed', ctypes.c_uint32),
    ]

# Function signatures
lib.cl_kmeans_params_init.argtypes = [ctypes.POINTER(KMeansParams)]
lib.cl_kmeans_params_init.restype = None

lib.cl_kmeans_fit.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(KMeansParams)
]
lib.cl_kmeans_fit.restype = ctypes.c_void_p

lib.cl_dbscan_params_init.argtypes = [ctypes.POINTER(DBSCANParams)]
lib.cl_dbscan_params_init.restype = None

lib.cl_dbscan_fit.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(DBSCANParams)
]
lib.cl_dbscan_fit.restype = ctypes.c_void_p

lib.cl_hierarchical_params_init.argtypes = [ctypes.POINTER(HierarchicalParams)]
lib.cl_hierarchical_params_init.restype = None

lib.cl_hierarchical_fit.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(HierarchicalParams)
]
lib.cl_hierarchical_fit.restype = ctypes.c_void_p

lib.cl_fastpam_params_init.argtypes = [ctypes.POINTER(FastPAMParams)]
lib.cl_fastpam_params_init.restype = None

lib.cl_fastpam_fit.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(FastPAMParams)
]
lib.cl_fastpam_fit.restype = ctypes.c_void_p

lib.cl_predict.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
    ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)
]
lib.cl_predict.restype = ctypes.c_int

lib.cl_silhouette.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_int32
]
lib.cl_silhouette.restype = ctypes.c_double

lib.cl_calinski_harabasz.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32)
]
lib.cl_calinski_harabasz.restype = ctypes.c_double

lib.cl_davies_bouldin.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32)
]
lib.cl_davies_bouldin.restype = ctypes.c_double

lib.cl_adjusted_rand.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32
]
lib.cl_adjusted_rand.restype = ctypes.c_double

lib.cl_save.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int32)
]
lib.cl_save.restype = ctypes.c_int

lib.cl_load.argtypes = [ctypes.c_char_p, ctypes.c_int32]
lib.cl_load.restype = ctypes.c_void_p

lib.cl_free.argtypes = [ctypes.c_void_p]
lib.cl_free.restype = None

lib.cl_free_buffer.argtypes = [ctypes.c_void_p]
lib.cl_free_buffer.restype = None

# Result field accessors via ctypes struct
class ClResult(ctypes.Structure):
    _fields_ = [
        ('labels', ctypes.POINTER(ctypes.c_int32)),
        ('n_samples', ctypes.c_int32),
        ('n_features', ctypes.c_int32),
        ('n_clusters', ctypes.c_int32),
        ('algorithm', ctypes.c_int32),
        ('metric', ctypes.c_int32),
        ('centers', ctypes.POINTER(ctypes.c_double)),
        ('inertia', ctypes.c_double),
        ('n_iter', ctypes.c_int32),
        ('medoid_indices', ctypes.POINTER(ctypes.c_int32)),
        ('medoid_coords', ctypes.POINTER(ctypes.c_double)),
        ('dendrogram', ctypes.c_void_p),
        ('core_mask', ctypes.POINTER(ctypes.c_uint8)),
        ('n_noise', ctypes.c_int32),
        ('seed', ctypes.c_uint32),
    ]

def get_result(ptr):
    return ctypes.cast(ptr, ctypes.POINTER(ClResult)).contents

# Test data
X_3c = np.array([
    [0.1, 0.2], [0.3, -0.1], [-0.2, 0.3], [0.0, 0.0], [0.4, 0.1],
    [10.0, 10.1], [10.2, 9.8], [9.9, 10.3], [10.1, 10.0], [10.3, 9.9],
    [20.0, 0.1], [19.8, -0.2], [20.1, 0.3], [20.3, 0.0], [19.9, 0.2],
], dtype=np.float64)
true_labels = np.array([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2], dtype=np.int32)


def to_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

def to_int_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# ========== Tests ==========

def test_kmeans_basic():
    p = KMeansParams()
    lib.cl_kmeans_params_init(ctypes.byref(p))
    p.k = 3
    p.seed = 42

    r_ptr = lib.cl_kmeans_fit(to_ptr(X_3c), 15, 2, ctypes.byref(p))
    assert r_ptr, "kmeans fit failed"
    r = get_result(r_ptr)
    assert r.n_clusters == 3
    labels = [r.labels[i] for i in range(15)]
    # Same-group consistency
    assert all(labels[i] == labels[0] for i in range(5))
    assert all(labels[i] == labels[5] for i in range(5, 10))
    assert all(labels[i] == labels[10] for i in range(10, 15))
    # Distinct clusters
    assert labels[0] != labels[5]
    assert labels[0] != labels[10]
    lib.cl_free(r_ptr)


def test_kmeans_vs_sklearn():
    """Compare inertia and cluster recovery with sklearn."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        print("  SKIP: sklearn not installed")
        return

    p = KMeansParams()
    lib.cl_kmeans_params_init(ctypes.byref(p))
    p.k = 3
    p.seed = 42
    p.n_init = 10

    r_ptr = lib.cl_kmeans_fit(to_ptr(X_3c), 15, 2, ctypes.byref(p))
    r = get_result(r_ptr)
    our_inertia = r.inertia
    our_labels = [r.labels[i] for i in range(15)]
    lib.cl_free(r_ptr)

    sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_3c)
    # Both should find the same clusters (inertia should be similar)
    assert abs(our_inertia - sk.inertia_) < 1.0, f"inertia diff: {our_inertia} vs {sk.inertia_}"


def test_dbscan_basic():
    p = DBSCANParams()
    lib.cl_dbscan_params_init(ctypes.byref(p))
    p.eps = 2.0
    p.min_pts = 3

    r_ptr = lib.cl_dbscan_fit(to_ptr(X_3c), 15, 2, ctypes.byref(p))
    r = get_result(r_ptr)
    assert r.n_clusters == 3
    assert r.n_noise == 0
    lib.cl_free(r_ptr)


def test_dbscan_noise():
    X = np.array([
        [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
        [10.0, 10.0], [10.1, 10.1], [10.2, 10.0],
        [50.0, 50.0],
    ], dtype=np.float64)

    p = DBSCANParams()
    lib.cl_dbscan_params_init(ctypes.byref(p))
    p.eps = 1.0
    p.min_pts = 3

    r_ptr = lib.cl_dbscan_fit(to_ptr(X), 7, 2, ctypes.byref(p))
    r = get_result(r_ptr)
    assert r.n_clusters == 2
    assert r.n_noise == 1
    assert r.labels[6] == -1
    lib.cl_free(r_ptr)


def test_fastpam_basic():
    p = FastPAMParams()
    lib.cl_fastpam_params_init(ctypes.byref(p))
    p.k = 3
    p.seed = 42

    r_ptr = lib.cl_fastpam_fit(to_ptr(X_3c), 15, 2, ctypes.byref(p))
    r = get_result(r_ptr)
    assert r.n_clusters == 3
    assert r.medoid_indices
    labels = [r.labels[i] for i in range(15)]
    assert all(labels[i] == labels[0] for i in range(5))
    # Medoids are actual data points
    for m in range(3):
        idx = r.medoid_indices[m]
        assert 0 <= idx < 15
    lib.cl_free(r_ptr)


def test_hierarchical_ward():
    p = HierarchicalParams()
    lib.cl_hierarchical_params_init(ctypes.byref(p))
    p.n_clusters = 3
    p.linkage = CL_LINK_WARD

    r_ptr = lib.cl_hierarchical_fit(to_ptr(X_3c), 15, 2, ctypes.byref(p))
    r = get_result(r_ptr)
    assert r.n_clusters == 3
    labels = [r.labels[i] for i in range(15)]
    assert all(labels[i] == labels[0] for i in range(5))
    assert all(labels[i] == labels[5] for i in range(5, 10))
    assert all(labels[i] == labels[10] for i in range(10, 15))
    lib.cl_free(r_ptr)


def test_silhouette():
    s = lib.cl_silhouette(to_ptr(X_3c), 15, 2, to_int_ptr(true_labels), CL_DIST_EUCLIDEAN, 0)
    assert s > 0.8, f"silhouette too low: {s}"
    assert s <= 1.0


def test_silhouette_vs_sklearn():
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("  SKIP: sklearn not installed")
        return

    our = lib.cl_silhouette(to_ptr(X_3c), 15, 2, to_int_ptr(true_labels), CL_DIST_EUCLIDEAN, 0)
    sk = silhouette_score(X_3c, true_labels)
    assert abs(our - sk) < 0.01, f"silhouette mismatch: {our} vs {sk}"


def test_calinski_harabasz():
    ch = lib.cl_calinski_harabasz(to_ptr(X_3c), 15, 2, to_int_ptr(true_labels))
    assert ch > 100


def test_calinski_harabasz_vs_sklearn():
    try:
        from sklearn.metrics import calinski_harabasz_score
    except ImportError:
        print("  SKIP: sklearn not installed")
        return

    our = lib.cl_calinski_harabasz(to_ptr(X_3c), 15, 2, to_int_ptr(true_labels))
    sk = calinski_harabasz_score(X_3c, true_labels)
    assert abs(our - sk) / sk < 0.01, f"CH mismatch: {our} vs {sk}"


def test_davies_bouldin():
    db = lib.cl_davies_bouldin(to_ptr(X_3c), 15, 2, to_int_ptr(true_labels))
    assert db >= 0
    assert db < 0.5


def test_davies_bouldin_vs_sklearn():
    try:
        from sklearn.metrics import davies_bouldin_score
    except ImportError:
        print("  SKIP: sklearn not installed")
        return

    our = lib.cl_davies_bouldin(to_ptr(X_3c), 15, 2, to_int_ptr(true_labels))
    sk = davies_bouldin_score(X_3c, true_labels)
    assert abs(our - sk) < 0.01, f"DB mismatch: {our} vs {sk}"


def test_adjusted_rand():
    ari = lib.cl_adjusted_rand(to_int_ptr(true_labels), to_int_ptr(true_labels), 15)
    assert abs(ari - 1.0) < 1e-10


def test_adjusted_rand_vs_sklearn():
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        print("  SKIP: sklearn not installed")
        return

    pred = np.array([1,1,1,1,1, 2,2,2,2,2, 0,0,0,0,0], dtype=np.int32)
    our = lib.cl_adjusted_rand(to_int_ptr(true_labels), to_int_ptr(pred), 15)
    sk = adjusted_rand_score(true_labels, pred)
    assert abs(our - sk) < 1e-10, f"ARI mismatch: {our} vs {sk}"


def test_save_load_roundtrip():
    p = KMeansParams()
    lib.cl_kmeans_params_init(ctypes.byref(p))
    p.k = 3
    p.seed = 42

    r_ptr = lib.cl_kmeans_fit(to_ptr(X_3c), 15, 2, ctypes.byref(p))
    r = get_result(r_ptr)
    orig_labels = [r.labels[i] for i in range(15)]
    orig_inertia = r.inertia

    buf = ctypes.c_char_p()
    length = ctypes.c_int32()
    ret = lib.cl_save(r_ptr, ctypes.byref(buf), ctypes.byref(length))
    assert ret == 0

    # Load from buffer
    r2_ptr = lib.cl_load(buf, length.value)
    assert r2_ptr
    r2 = get_result(r2_ptr)
    assert r2.n_clusters == 3
    assert abs(r2.inertia - orig_inertia) < 1e-10
    for i in range(15):
        assert r2.labels[i] == orig_labels[i]

    lib.cl_free(r_ptr)
    lib.cl_free(r2_ptr)
    lib.cl_free_buffer(buf)


# ========== Runner ==========

tests = [
    test_kmeans_basic,
    test_kmeans_vs_sklearn,
    test_dbscan_basic,
    test_dbscan_noise,
    test_fastpam_basic,
    test_hierarchical_ward,
    test_silhouette,
    test_silhouette_vs_sklearn,
    test_calinski_harabasz,
    test_calinski_harabasz_vs_sklearn,
    test_davies_bouldin,
    test_davies_bouldin_vs_sklearn,
    test_adjusted_rand,
    test_adjusted_rand_vs_sklearn,
    test_save_load_roundtrip,
]

if __name__ == '__main__':
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f'  OK: {t.__name__}')
        except Exception as e:
            failed += 1
            print(f'  FAIL: {t.__name__}: {e}')

    print(f'\n{passed}/{passed + failed} tests passed')
    sys.exit(1 if failed else 0)
