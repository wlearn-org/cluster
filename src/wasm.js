// WASM loader -- loads the Cluster WASM module (singleton, lazy init)

import { createRequire } from 'module'

let wasmModule = null
let loading = null

export async function loadCluster(options = {}) {
  if (wasmModule) return wasmModule
  if (loading) return loading

  loading = (async () => {
    const require = createRequire(import.meta.url)
    const createCluster = require('../wasm/cluster.cjs')
    wasmModule = await createCluster(options)
    return wasmModule
  })()

  return loading
}

export function getWasm() {
  if (!wasmModule) throw new Error('WASM not loaded -- call loadCluster() first')
  return wasmModule
}
