import type { WorkerRequest, WorkerResponse, GpuInfo } from '../src/lib/types'

export type RuntimeRequest = Exclude<WorkerRequest, {type: 'init' | 'load-program' | 'verify'}>
  | {type: 'init'; data: {numThreads: number; cacheBust?: string; gpu?: boolean | 'w1' | 'w2' | 's2off'; gpuTimeoutMs?: number; parityRounds?: number}}
  | {type: 'load-program'; data: {program: string; programPreprocessing: ArrayBuffer; elfBytes: ArrayBuffer}}
  | {type: 'prove'; data: {program: 'sha2-chain'; input: number[]; numIters: number}}
  | {type: 'verify'; data: {program: string; proof: Uint8Array; programIo: Uint8Array; verifierPreprocessing: Uint8Array}}
  | {type: 'gpu-trip-probe'; data: {n: number}}
  | {type: 'hang-gpu-proxy'}

export type ProxyRequest = {type: 'hang'} | {type: 'init'; memory: WebAssembly.Memory; mailboxPtr: number}
export type ProxyReport = {type: 'unavailable'; reason: string} | {type: 'ready'; adapter: NonNullable<GpuInfo['adapter']>; features: string[]; limits: Record<string, number>}
export type BenchReply = WorkerResponse

declare global {
  interface Window {
    __bench: {
      worker: Worker
      pending: Map<string, (message: BenchReply) => void>
      wait: (type: string) => Promise<BenchReply>
    }
  }
}
