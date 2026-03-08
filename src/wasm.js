// WASM loader -- loads the Cluster WASM module (singleton, lazy init)

let wasmModule = null
let loading = null

async function loadCluster(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const createCluster = require('../wasm/cluster.js')
    wasmModule = await createCluster(options)
    return wasmModule
  })()

  return loading
}

function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadCluster() first')
  return wasmModule
}

module.exports = { loadCluster, getWasm }
