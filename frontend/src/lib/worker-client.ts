import type { ProgramName, WorkerRequest, WorkerResponse } from './types'

export class WorkerClient {
  private worker: Worker
  // patches/0010: a rayon pool worker that traps leaves worker.js blocked in
  // the prove call, so its error arrives on this channel instead.
  private pool = new BroadcastChannel('wasm_bindgen_rayon')
  private program: ProgramName | undefined

  constructor(onMessage: (msg: WorkerResponse) => void, onError: (e: ErrorEvent) => void) {
    // Use string URL to avoid Vite's worker import analysis — worker.js
    // lives in public/ and imports from /pkg/ which only exists at runtime.
    const workerUrl = new URL('/worker.js', window.location.origin)
    // Pool workers inherit this name; the channel is origin-wide.
    const name = crypto.randomUUID()
    this.worker = new Worker(workerUrl, { type: 'module', name })
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => { onMessage(e.data) }
    this.worker.onerror = onError
    this.pool.onmessage = (e: MessageEvent<{ name: string; error: string }>) => {
      if (e.data.name !== name) return
      // Every pool worker traps on the same failure; the first report kills the session.
      this.pool.close()
      onMessage({ type: 'error', error: e.data.error, ...(this.program && { trapped: this.program }) })
    }
  }

  send(msg: WorkerRequest, transfer?: Transferable[]) {
    if (msg.type === 'prove' || msg.type === 'verify') this.program = msg.data.program
    this.worker.postMessage(msg, transfer ?? [])
  }

  terminate() {
    this.worker.terminate()
    this.pool.close()
  }
}
