import { ClusterModel, silhouette, calinskiHarabasz, daviesBouldin, adjustedRand, loadCluster } from '../src/index.js'
import assert from 'node:assert/strict'

let tests = 0, passed = 0

async function run(name, fn) {
  tests++
  try {
    await fn()
    passed++
    console.log(`  OK: ${name}`)
  } catch (e) {
    console.log(`  FAIL: ${name}: ${e.message}`)
  }
}

// 3 well-separated clusters in 2D
const X_3c = [
  [0.1, 0.2], [0.3, -0.1], [-0.2, 0.3], [0.0, 0.0], [0.4, 0.1],
  [10.0, 10.1], [10.2, 9.8], [9.9, 10.3], [10.1, 10.0], [10.3, 9.9],
  [20.0, 0.1], [19.8, -0.2], [20.1, 0.3], [20.3, 0.0], [19.9, 0.2],
]
const trueLabels = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]

function checkClusters(labels) {
  // Points in same original group should have same label
  for (let i = 1; i < 5; i++) assert.equal(labels[i], labels[0])
  for (let i = 6; i < 10; i++) assert.equal(labels[i], labels[5])
  for (let i = 11; i < 15; i++) assert.equal(labels[i], labels[10])
  // Three distinct labels
  assert.notEqual(labels[0], labels[5])
  assert.notEqual(labels[0], labels[10])
  assert.notEqual(labels[5], labels[10])
}

await loadCluster()

// ========== K-Means ==========
console.log('K-Means:')

await run('basic fit', async () => {
  const km = await ClusterModel.create({ method: 'kmeans', k: 3, seed: 42 })
  km.fit(X_3c)
  assert.equal(km.nClusters, 3)
  assert.equal(km.labels.length, 15)
  checkClusters(km.labels)
  assert.ok(km.inertia < 5.0)
  assert.ok(km.centers !== null)
  assert.equal(km.centers.length, 6) // 3 * 2
  km.dispose()
})

await run('predict', async () => {
  const km = await ClusterModel.create({ method: 'kmeans', k: 3, seed: 42 })
  km.fit(X_3c)
  const labels = km.labels
  const pred = km.predict([[0.5, 0.5], [10.0, 10.0], [20.0, 0.0]])
  assert.equal(pred[0], labels[0])
  assert.equal(pred[1], labels[5])
  assert.equal(pred[2], labels[10])
  km.dispose()
})

await run('score (silhouette)', async () => {
  const km = await ClusterModel.create({ method: 'kmeans', k: 3, seed: 42 })
  km.fit(X_3c)
  const s = km.score(X_3c)
  assert.ok(s > 0.8)
  assert.ok(s <= 1.0)
  km.dispose()
})

await run('save/load roundtrip', async () => {
  const km = await ClusterModel.create({ method: 'kmeans', k: 3, seed: 42 })
  km.fit(X_3c)
  const bytes = km.save()
  assert.ok(bytes instanceof Uint8Array)
  assert.ok(bytes.length > 0)

  const km2 = await ClusterModel.load(bytes)
  assert.equal(km2.nClusters, 3)
  assert.equal(km2.method, 'kmeans')
  const l1 = km.labels, l2 = km2.labels
  for (let i = 0; i < 15; i++) assert.equal(l1[i], l2[i])

  const pred1 = km.predict([[5.0, 5.0]])
  const pred2 = km2.predict([[5.0, 5.0]])
  assert.equal(pred1[0], pred2[0])

  km.dispose()
  km2.dispose()
})

await run('flat Float64Array input', async () => {
  const flat = new Float64Array(X_3c.flat())
  const km = await ClusterModel.create({ method: 'kmeans', k: 3, seed: 42, nFeatures: 2 })
  km.fit(flat)
  assert.equal(km.nClusters, 3)
  checkClusters(km.labels)
  km.dispose()
})

// ========== Mini-batch K-Means ==========
console.log('Mini-batch K-Means:')

await run('basic fit', async () => {
  const mb = await ClusterModel.create({ method: 'minibatch', k: 3, batchSize: 5, seed: 42 })
  mb.fit(X_3c)
  assert.equal(mb.nClusters, 3)
  checkClusters(mb.labels)
  mb.dispose()
})

await run('predict', async () => {
  const mb = await ClusterModel.create({ method: 'minibatch', k: 3, seed: 42 })
  mb.fit(X_3c)
  const labels = mb.labels
  const pred = mb.predict([[0.5, 0.5], [10.0, 10.0], [20.0, 0.0]])
  assert.equal(pred[0], labels[0])
  assert.equal(pred[1], labels[5])
  assert.equal(pred[2], labels[10])
  mb.dispose()
})

// ========== DBSCAN ==========
console.log('DBSCAN:')

await run('basic fit', async () => {
  const db = await ClusterModel.create({ method: 'dbscan', eps: 2.0, minPts: 3 })
  db.fit(X_3c)
  assert.equal(db.nClusters, 3)
  assert.equal(db.nNoise, 0)
  assert.ok(db.coreMask !== null)
  checkClusters(db.labels)
  db.dispose()
})

await run('noise detection', async () => {
  const X = [
    [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
    [10.0, 10.0], [10.1, 10.1], [10.2, 10.0],
    [50.0, 50.0],
  ]
  const db = await ClusterModel.create({ method: 'dbscan', eps: 1.0, minPts: 3 })
  db.fit(X)
  assert.equal(db.nClusters, 2)
  assert.equal(db.nNoise, 1)
  assert.equal(db.labels[6], -1)
  db.dispose()
})

await run('predict throws', async () => {
  const db = await ClusterModel.create({ method: 'dbscan', eps: 2.0, minPts: 3 })
  db.fit(X_3c)
  assert.throws(() => db.predict([[0, 0]]), /predict/i)
  db.dispose()
})

// ========== FastPAM ==========
console.log('FastPAM:')

await run('basic fit', async () => {
  const fp = await ClusterModel.create({ method: 'fastpam', k: 3, seed: 42 })
  fp.fit(X_3c)
  assert.equal(fp.nClusters, 3)
  assert.ok(fp.medoidIndices !== null)
  assert.ok(fp.medoidCoords !== null)
  assert.equal(fp.centers, null) // no centroids for PAM
  checkClusters(fp.labels)

  // Medoid indices should be valid
  for (const idx of fp.medoidIndices) {
    assert.ok(idx >= 0 && idx < 15)
  }
  fp.dispose()
})

await run('predict', async () => {
  const fp = await ClusterModel.create({ method: 'fastpam', k: 3, seed: 42 })
  fp.fit(X_3c)
  const labels = fp.labels
  const pred = fp.predict([[0.5, 0.5], [10.0, 10.0], [20.0, 0.0]])
  assert.equal(pred[0], labels[0])
  assert.equal(pred[1], labels[5])
  assert.equal(pred[2], labels[10])
  fp.dispose()
})

await run('manhattan metric', async () => {
  const fp = await ClusterModel.create({ method: 'fastpam', k: 3, metric: 'manhattan', seed: 42 })
  fp.fit(X_3c)
  assert.equal(fp.nClusters, 3)
  fp.dispose()
})

await run('save/load roundtrip', async () => {
  const fp = await ClusterModel.create({ method: 'fastpam', k: 3, seed: 42 })
  fp.fit(X_3c)
  const bytes = fp.save()

  const fp2 = await ClusterModel.load(bytes)
  assert.equal(fp2.method, 'fastpam')
  assert.ok(fp2.medoidIndices !== null)

  const pred1 = fp.predict([[5.0, 5.0]])
  const pred2 = fp2.predict([[5.0, 5.0]])
  assert.equal(pred1[0], pred2[0])

  fp.dispose()
  fp2.dispose()
})

// ========== Hierarchical ==========
console.log('Hierarchical:')

await run('ward linkage', async () => {
  const hc = await ClusterModel.create({ method: 'hierarchical', k: 3, linkage: 'ward' })
  hc.fit(X_3c)
  assert.equal(hc.nClusters, 3)
  checkClusters(hc.labels)
  hc.dispose()
})

await run('single linkage', async () => {
  const hc = await ClusterModel.create({ method: 'hierarchical', k: 3, linkage: 'single' })
  hc.fit(X_3c)
  assert.equal(hc.nClusters, 3)
  hc.dispose()
})

await run('dendrogram', async () => {
  const hc = await ClusterModel.create({ method: 'hierarchical', k: 1, linkage: 'ward' })
  hc.fit(X_3c)
  const dendro = hc.dendrogram
  assert.ok(dendro !== null)
  assert.equal(dendro.length, 14) // n-1 merges
  // Monotonic merge distances
  for (let i = 1; i < dendro.length; i++) {
    assert.ok(dendro[i].distance >= dendro[i-1].distance - 1e-10)
  }
  // Last merge includes all points
  assert.equal(dendro[dendro.length - 1].size, 15)
  hc.dispose()
})

await run('ward rejects cosine', async () => {
  const hc = await ClusterModel.create({ method: 'hierarchical', k: 3, linkage: 'ward', metric: 'cosine' })
  assert.throws(() => hc.fit(X_3c), /Ward|Euclidean/i)
  hc.dispose()
})

await run('save/load roundtrip', async () => {
  const hc = await ClusterModel.create({ method: 'hierarchical', k: 3, linkage: 'ward' })
  hc.fit(X_3c)
  const bytes = hc.save()

  const hc2 = await ClusterModel.load(bytes)
  assert.equal(hc2.method, 'hierarchical')
  assert.ok(hc2.dendrogram !== null)
  const d1 = hc.dendrogram, d2 = hc2.dendrogram
  assert.equal(d1.length, d2.length)
  for (let i = 0; i < d1.length; i++) {
    assert.equal(d1[i].left, d2[i].left)
    assert.equal(d1[i].right, d2[i].right)
    assert.ok(Math.abs(d1[i].distance - d2[i].distance) < 1e-10)
  }

  hc.dispose()
  hc2.dispose()
})

// ========== Validation ==========
console.log('Validation:')

await run('silhouette', async () => {
  const s = silhouette(X_3c, trueLabels)
  assert.ok(s > 0.8)
  assert.ok(s <= 1.0)
})

await run('calinski-harabasz', async () => {
  const ch = calinskiHarabasz(X_3c, trueLabels)
  assert.ok(ch > 100)
})

await run('davies-bouldin', async () => {
  const db = daviesBouldin(X_3c, trueLabels)
  assert.ok(db >= 0)
  assert.ok(db < 0.5)
})

await run('adjusted rand - perfect', async () => {
  const ari = adjustedRand(trueLabels, trueLabels)
  assert.ok(Math.abs(ari - 1.0) < 1e-10)
})

await run('adjusted rand - permuted', async () => {
  const permuted = [1,1,1,1,1, 2,2,2,2,2, 0,0,0,0,0]
  const ari = adjustedRand(trueLabels, permuted)
  assert.ok(Math.abs(ari - 1.0) < 1e-10)
})

await run('adjusted rand - symmetric', async () => {
  const pred = [1,1,1,1,1, 2,2,2,2,2, 0,0,0,0,0]
  const ari1 = adjustedRand(trueLabels, pred)
  const ari2 = adjustedRand(pred, trueLabels)
  assert.ok(Math.abs(ari1 - ari2) < 1e-10)
})

// ========== Error handling ==========
console.log('Error handling:')

await run('not fitted throws', async () => {
  const km = await ClusterModel.create({ method: 'kmeans', k: 3 })
  assert.throws(() => km.labels, /not fitted/i)
  assert.throws(() => km.predict([[0, 0]]), /not fitted/i)
  assert.throws(() => km.save(), /not fitted/i)
})

await run('dispose cleans up', async () => {
  const km = await ClusterModel.create({ method: 'kmeans', k: 3, seed: 42 })
  km.fit(X_3c)
  km.dispose()
  // Re-fit should work after dispose
  km.fit(X_3c)
  assert.equal(km.nClusters, 3)
  km.dispose()
})

await run('unknown method throws', async () => {
  const m = await ClusterModel.create({ method: 'bogus' })
  assert.throws(() => m.fit(X_3c), /Unknown method/i)
})

// ========== Summary ==========
console.log(`\n${passed}/${tests} tests passed`)
process.exit(passed === tests ? 0 : 1)
