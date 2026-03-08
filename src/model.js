const { getWasm, loadCluster } = require('./wasm.js')

// Constants matching cluster.h
const METRIC = { euclidean: 0, manhattan: 1, cosine: 2 }
const INIT = { kmpp: 0, random: 1 }
const LINKAGE = { single: 0, complete: 1, average: 2, ward: 3 }
const ALGO = { kmeans: 0, minibatch: 1, dbscan: 2, hierarchical: 3, fastpam: 4 }

function resolveEnum(map, value, fallback) {
  if (typeof value === 'number') return value
  if (typeof value === 'string') {
    const v = map[value.toLowerCase()]
    if (v !== undefined) return v
  }
  return fallback
}

function getLastError() {
  const wasm = getWasm()
  return wasm.ccall('wl_cluster_get_last_error', 'string', [], [])
}

class ClusterModel {
  #handle = null
  #params = {}
  #fitted = false
  #disposed = false

  constructor() {
    // Use ClusterModel.create() instead
  }

  static async create(params = {}) {
    await loadCluster()
    const model = new ClusterModel()
    model.#params = { ...params }
    return model
  }

  get method() { return this.#params.method || 'kmeans' }
  get isFitted() { return this.#fitted }

  fit(X) {
    if (this.#disposed) throw new Error('Model disposed')
    this.dispose()

    const wasm = getWasm()
    const method = (this.#params.method || 'kmeans').toLowerCase()

    // Normalize X
    let data, nrow, ncol
    if (X instanceof Float64Array) {
      ncol = this.#params.nFeatures || this.#params.n_features
      if (!ncol) throw new Error('nFeatures required when X is a flat Float64Array')
      nrow = X.length / ncol
      data = X
    } else if (Array.isArray(X)) {
      nrow = X.length
      ncol = Array.isArray(X[0]) ? X[0].length : 1
      data = new Float64Array(nrow * ncol)
      for (let i = 0; i < nrow; i++) {
        const row = Array.isArray(X[i]) ? X[i] : [X[i]]
        for (let j = 0; j < ncol; j++) data[i * ncol + j] = row[j]
      }
    } else {
      throw new Error('X must be Float64Array or Array')
    }

    // Allocate WASM memory for X
    const xPtr = wasm._malloc(nrow * ncol * 8)
    wasm.HEAPF64.set(data, xPtr / 8)

    let handle = null

    if (method === 'kmeans') {
      const k = this.#params.k || 3
      const maxIter = this.#params.maxIter || this.#params.max_iter || 300
      const nInit = this.#params.nInit || this.#params.n_init || 10
      const init = resolveEnum(INIT, this.#params.init, 0)
      const tol = this.#params.tol || 1e-4
      const seed = this.#params.seed || 42
      handle = wasm._wl_cluster_fit_kmeans(xPtr, nrow, ncol, k, maxIter, nInit, init, tol, seed)
    } else if (method === 'minibatch') {
      const k = this.#params.k || 3
      const maxIter = this.#params.maxIter || this.#params.max_iter || 100
      const nInit = this.#params.nInit || this.#params.n_init || 3
      const init = resolveEnum(INIT, this.#params.init, 0)
      const batchSize = this.#params.batchSize || this.#params.batch_size || 1024
      const tol = this.#params.tol || 0.0
      const seed = this.#params.seed || 42
      handle = wasm._wl_cluster_fit_minibatch(xPtr, nrow, ncol, k, maxIter, nInit, init, batchSize, tol, seed)
    } else if (method === 'dbscan') {
      const eps = this.#params.eps || 0.5
      const minPts = this.#params.minPts || this.#params.min_pts || 5
      const metric = resolveEnum(METRIC, this.#params.metric, 0)
      handle = wasm._wl_cluster_fit_dbscan(xPtr, nrow, ncol, eps, minPts, metric)
    } else if (method === 'hierarchical') {
      const nClusters = this.#params.nClusters || this.#params.n_clusters || this.#params.k || 2
      const distThresh = this.#params.distanceThreshold || this.#params.distance_threshold || 0.0
      const linkage = resolveEnum(LINKAGE, this.#params.linkage, 3)
      const metric = resolveEnum(METRIC, this.#params.metric, 0)
      handle = wasm._wl_cluster_fit_hierarchical(xPtr, nrow, ncol, nClusters, distThresh, linkage, metric)
    } else if (method === 'fastpam') {
      const k = this.#params.k || 3
      const maxIter = this.#params.maxIter || this.#params.max_iter || 100
      const init = resolveEnum(INIT, this.#params.init, 0)
      const metric = resolveEnum(METRIC, this.#params.metric, 0)
      const seed = this.#params.seed || 42
      handle = wasm._wl_cluster_fit_fastpam(xPtr, nrow, ncol, k, maxIter, init, metric, seed)
    } else {
      wasm._free(xPtr)
      throw new Error(`Unknown method: ${method}`)
    }

    wasm._free(xPtr)

    if (!handle) {
      throw new Error(`Cluster fit failed: ${getLastError()}`)
    }

    this.#handle = handle
    this.#fitted = true
    this.#params.nFeatures = ncol
    return this
  }

  get labels() {
    this.#checkFitted()
    const wasm = getWasm()
    const n = wasm._wl_cluster_get_n_samples(this.#handle)
    const ptr = wasm._wl_cluster_get_labels(this.#handle)
    return new Int32Array(wasm.HEAP32.buffer, ptr, n).slice()
  }

  get nClusters() {
    this.#checkFitted()
    return getWasm()._wl_cluster_get_n_clusters(this.#handle)
  }

  get inertia() {
    this.#checkFitted()
    return getWasm()._wl_cluster_get_inertia(this.#handle)
  }

  get nIter() {
    this.#checkFitted()
    return getWasm()._wl_cluster_get_n_iter(this.#handle)
  }

  get nNoise() {
    this.#checkFitted()
    return getWasm()._wl_cluster_get_n_noise(this.#handle)
  }

  get centers() {
    this.#checkFitted()
    const wasm = getWasm()
    const ptr = wasm._wl_cluster_get_centers(this.#handle)
    if (!ptr) return null
    const k = wasm._wl_cluster_get_n_clusters(this.#handle)
    const d = wasm._wl_cluster_get_n_features(this.#handle)
    return new Float64Array(wasm.HEAPF64.buffer, ptr, k * d).slice()
  }

  get medoidIndices() {
    this.#checkFitted()
    const wasm = getWasm()
    const ptr = wasm._wl_cluster_get_medoid_indices(this.#handle)
    if (!ptr) return null
    const k = wasm._wl_cluster_get_n_clusters(this.#handle)
    return new Int32Array(wasm.HEAP32.buffer, ptr, k).slice()
  }

  get medoidCoords() {
    this.#checkFitted()
    const wasm = getWasm()
    const ptr = wasm._wl_cluster_get_medoid_coords(this.#handle)
    if (!ptr) return null
    const k = wasm._wl_cluster_get_n_clusters(this.#handle)
    const d = wasm._wl_cluster_get_n_features(this.#handle)
    return new Float64Array(wasm.HEAPF64.buffer, ptr, k * d).slice()
  }

  get coreMask() {
    this.#checkFitted()
    const wasm = getWasm()
    const ptr = wasm._wl_cluster_get_core_mask(this.#handle)
    if (!ptr) return null
    const n = wasm._wl_cluster_get_n_samples(this.#handle)
    return new Uint8Array(wasm.HEAPU8.buffer, ptr, n).slice()
  }

  get dendrogram() {
    this.#checkFitted()
    const wasm = getWasm()
    const nMerges = wasm._wl_cluster_get_dendro_n_merges(this.#handle)
    if (nMerges === 0) return null
    const merges = []
    for (let i = 0; i < nMerges; i++) {
      merges.push({
        left: wasm._wl_cluster_get_dendro_left(this.#handle, i),
        right: wasm._wl_cluster_get_dendro_right(this.#handle, i),
        distance: wasm._wl_cluster_get_dendro_distance(this.#handle, i),
        size: wasm._wl_cluster_get_dendro_size(this.#handle, i)
      })
    }
    return merges
  }

  predict(X) {
    this.#checkFitted()
    const wasm = getWasm()

    let data, nrow
    const ncol = this.#params.nFeatures
    if (X instanceof Float64Array) {
      nrow = X.length / ncol
      data = X
    } else if (Array.isArray(X)) {
      nrow = X.length
      data = new Float64Array(nrow * ncol)
      for (let i = 0; i < nrow; i++) {
        const row = Array.isArray(X[i]) ? X[i] : [X[i]]
        for (let j = 0; j < ncol; j++) data[i * ncol + j] = row[j]
      }
    } else {
      throw new Error('X must be Float64Array or Array')
    }

    const xPtr = wasm._malloc(nrow * ncol * 8)
    wasm.HEAPF64.set(data, xPtr / 8)
    const outPtr = wasm._malloc(nrow * 4)

    const ret = wasm._wl_cluster_predict(this.#handle, xPtr, nrow, ncol, outPtr)
    if (ret !== 0) {
      wasm._free(xPtr)
      wasm._free(outPtr)
      throw new Error(`Predict failed: ${getLastError()}`)
    }

    const result = new Int32Array(wasm.HEAP32.buffer, outPtr, nrow).slice()
    wasm._free(xPtr)
    wasm._free(outPtr)
    return result
  }

  score(X) {
    this.#checkFitted()
    return silhouette(X, this.labels, {
      metric: this.#params.metric,
      nFeatures: this.#params.nFeatures
    })
  }

  save() {
    this.#checkFitted()
    const wasm = getWasm()
    const bufPtrPtr = wasm._malloc(4)
    const lenPtr = wasm._malloc(4)

    const ret = wasm._wl_cluster_save(this.#handle, bufPtrPtr, lenPtr)
    if (ret !== 0) {
      wasm._free(bufPtrPtr)
      wasm._free(lenPtr)
      throw new Error(`Save failed: ${getLastError()}`)
    }

    const bufPtr = wasm.HEAP32[bufPtrPtr / 4]
    const len = wasm.HEAP32[lenPtr / 4]
    const bytes = new Uint8Array(wasm.HEAPU8.buffer, bufPtr, len).slice()

    wasm._wl_cluster_free_buffer(bufPtr)
    wasm._free(bufPtrPtr)
    wasm._free(lenPtr)
    return bytes
  }

  static async load(bytes) {
    await loadCluster()
    const wasm = getWasm()

    const ptr = wasm._malloc(bytes.length)
    wasm.HEAPU8.set(bytes, ptr)
    const handle = wasm._wl_cluster_load(ptr, bytes.length)
    wasm._free(ptr)

    if (!handle) throw new Error(`Load failed: ${getLastError()}`)

    const model = new ClusterModel()
    model.#handle = handle
    model.#fitted = true
    model.#params.nFeatures = wasm._wl_cluster_get_n_features(handle)

    const algo = wasm._wl_cluster_get_algorithm(handle)
    const methods = ['kmeans', 'minibatch', 'dbscan', 'hierarchical', 'fastpam']
    model.#params.method = methods[algo] || 'kmeans'

    return model
  }

  dispose() {
    if (this.#handle && !this.#disposed) {
      getWasm()._wl_cluster_free(this.#handle)
      this.#handle = null
      this.#fitted = false
    }
  }

  #checkFitted() {
    if (this.#disposed) throw new Error('Model disposed')
    if (!this.#fitted) throw new Error('Model not fitted -- call fit() first')
  }
}

// Standalone validation functions
function silhouette(X, labels, opts = {}) {
  const wasm = getWasm()
  const metric = resolveEnum(METRIC, opts.metric, 0)
  const ignoreNoise = opts.ignoreNoise ? 1 : 0

  let data, nrow, ncol
  if (X instanceof Float64Array) {
    ncol = opts.nFeatures || opts.n_features
    if (!ncol) throw new Error('nFeatures required for flat Float64Array')
    nrow = X.length / ncol
    data = X
  } else if (Array.isArray(X)) {
    nrow = X.length
    ncol = Array.isArray(X[0]) ? X[0].length : 1
    data = new Float64Array(nrow * ncol)
    for (let i = 0; i < nrow; i++) {
      const row = Array.isArray(X[i]) ? X[i] : [X[i]]
      for (let j = 0; j < ncol; j++) data[i * ncol + j] = row[j]
    }
  } else {
    throw new Error('X must be Float64Array or Array')
  }

  const labelsArr = labels instanceof Int32Array ? labels : new Int32Array(labels)
  const xPtr = wasm._malloc(nrow * ncol * 8)
  const lPtr = wasm._malloc(nrow * 4)
  wasm.HEAPF64.set(data, xPtr / 8)
  wasm.HEAP32.set(labelsArr, lPtr / 4)

  const result = wasm._wl_cluster_silhouette(xPtr, nrow, ncol, lPtr, metric, ignoreNoise)

  wasm._free(xPtr)
  wasm._free(lPtr)
  return result
}

function calinskiHarabasz(X, labels, opts = {}) {
  const wasm = getWasm()

  let data, nrow, ncol
  if (X instanceof Float64Array) {
    ncol = opts.nFeatures || opts.n_features
    nrow = X.length / ncol
    data = X
  } else if (Array.isArray(X)) {
    nrow = X.length
    ncol = Array.isArray(X[0]) ? X[0].length : 1
    data = new Float64Array(nrow * ncol)
    for (let i = 0; i < nrow; i++) {
      const row = Array.isArray(X[i]) ? X[i] : [X[i]]
      for (let j = 0; j < ncol; j++) data[i * ncol + j] = row[j]
    }
  }

  const labelsArr = labels instanceof Int32Array ? labels : new Int32Array(labels)
  const xPtr = wasm._malloc(nrow * ncol * 8)
  const lPtr = wasm._malloc(nrow * 4)
  wasm.HEAPF64.set(data, xPtr / 8)
  wasm.HEAP32.set(labelsArr, lPtr / 4)

  const result = wasm._wl_cluster_calinski_harabasz(xPtr, nrow, ncol, lPtr)

  wasm._free(xPtr)
  wasm._free(lPtr)
  return result
}

function daviesBouldin(X, labels, opts = {}) {
  const wasm = getWasm()

  let data, nrow, ncol
  if (X instanceof Float64Array) {
    ncol = opts.nFeatures || opts.n_features
    nrow = X.length / ncol
    data = X
  } else if (Array.isArray(X)) {
    nrow = X.length
    ncol = Array.isArray(X[0]) ? X[0].length : 1
    data = new Float64Array(nrow * ncol)
    for (let i = 0; i < nrow; i++) {
      const row = Array.isArray(X[i]) ? X[i] : [X[i]]
      for (let j = 0; j < ncol; j++) data[i * ncol + j] = row[j]
    }
  }

  const labelsArr = labels instanceof Int32Array ? labels : new Int32Array(labels)
  const xPtr = wasm._malloc(nrow * ncol * 8)
  const lPtr = wasm._malloc(nrow * 4)
  wasm.HEAPF64.set(data, xPtr / 8)
  wasm.HEAP32.set(labelsArr, lPtr / 4)

  const result = wasm._wl_cluster_davies_bouldin(xPtr, nrow, ncol, lPtr)

  wasm._free(xPtr)
  wasm._free(lPtr)
  return result
}

function adjustedRand(labelsTrue, labelsPred) {
  const wasm = getWasm()
  const n = labelsTrue.length
  const trueArr = labelsTrue instanceof Int32Array ? labelsTrue : new Int32Array(labelsTrue)
  const predArr = labelsPred instanceof Int32Array ? labelsPred : new Int32Array(labelsPred)

  const tPtr = wasm._malloc(n * 4)
  const pPtr = wasm._malloc(n * 4)
  wasm.HEAP32.set(trueArr, tPtr / 4)
  wasm.HEAP32.set(predArr, pPtr / 4)

  const result = wasm._wl_cluster_adjusted_rand(tPtr, pPtr, n)

  wasm._free(tPtr)
  wasm._free(pPtr)
  return result
}

module.exports = { ClusterModel, silhouette, calinskiHarabasz, daviesBouldin, adjustedRand }
