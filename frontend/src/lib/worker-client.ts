import type { WorkerRequest, WorkerResponse } from './types'

export class WorkerClient {
  private worker: Worker

  constructor(onMessage: (msg: WorkerResponse) => void, onError: (e: ErrorEvent) => void) {
    // Use string URL to avoid Vite's worker import analysis — worker.js
    // lives in public/ and imports from /pkg/ which only exists at runtime.
    const workerUrl = new URL('/worker.js', window.location.origin)
    this.worker = new Worker(workerUrl, { type: 'module' })
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => onMessage(e.data)
    this.worker.onerror = onError
  }

  send(msg: WorkerRequest, transfer?: Transferable[]) {
    this.worker.postMessage(msg, transfer ?? [])
  }

  terminate() {
    this.worker.terminate()
  }
}
