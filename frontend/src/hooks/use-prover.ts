import { useState, useRef, useCallback, useEffect } from 'react'
import type {
  ProgramName,
  ProgramState,
  AppStatus,
  GpuStatus,
  WorkerResponse,
} from '@/lib/types'
import { PROGRAMS, PROGRAM_FILES, CACHE_BUST, SHA2_MAX_BYTES } from '@/lib/constants'
import { WorkerClient } from '@/lib/worker-client'

// Parsed once at app init; the arm can't be toggled after the thread pool
// and GPU worker come up, so a reload is the only switch anyway.
const WEBGPU_REQUESTED =
  new URLSearchParams(window.location.search).get('webgpu') !== '0'

// Recycle the prover worker after any GPU-arm prove this large: warm
// reruns in one worker accumulate wasm-allocator churn and device state
// until they slow down (+66% @2^23) or lose the device outright (@2^22,
// run 4-6 on the wave-3 lineage). Fresh workers are immune; below this
// scale the accumulation has never been observed. `?recycle=0` keeps the
// old keep-warm behavior for A/B.
const RECYCLE_MIN_PADDED = 4_194_304
const RECYCLE_ENABLED =
  new URLSearchParams(window.location.search).get('recycle') !== '0'

interface ProverState {
  status: AppStatus
  statusText: string
  wasmReady: boolean
  gpu: GpuStatus
  programStates: Record<ProgramName, ProgramState>
  outputLogs: Record<ProgramName, string>
}

function initialProgramStates(): Record<ProgramName, ProgramState> {
  const states = {} as Record<ProgramName, ProgramState>
  for (const p of PROGRAMS) {
    states[p] = { loadState: 'idle', proofBytes: null, programIoBytes: null, verifyResult: null }
  }
  return states
}

async function sha256Digest(bytes: Uint8Array): Promise<Uint8Array> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', bytes.buffer as ArrayBuffer)
  return new Uint8Array(hashBuffer)
}

export function useProver() {
  const [state, setState] = useState<ProverState>({
    status: 'loading',
    statusText: 'Initializing WASM...',
    wasmReady: false,
    gpu: WEBGPU_REQUESTED ? 'pending' : 'off',
    programStates: initialProgramStates(),
    outputLogs: { sha2: '', keccak: '' },
  })

  const clientRef = useRef<WorkerClient | null>(null)
  const programLoadResolvers = useRef<Record<string, () => void>>({})
  const programStatesRef = useRef(state.programStates)
  programStatesRef.current = state.programStates
  const gpuRef = useRef(state.gpu)
  gpuRef.current = state.gpu
  // Breaks the handleMessage → recycle → boot → handleMessage cycle.
  const recycleRef = useRef<() => void>(() => {})

  const log = useCallback((program: ProgramName, msg: string) => {
    setState((prev) => ({
      ...prev,
      outputLogs: {
        ...prev.outputLogs,
        [program]: prev.outputLogs[program] + msg + '\n',
      },
    }))
    console.log(`[${program}] ${msg}`)
  }, [])

  const setStatus = useCallback((statusText: string, status: AppStatus) => {
    setState((prev) => ({ ...prev, status, statusText }))
  }, [])

  const handleMessage = useCallback(
    (msg: WorkerResponse) => {
      if (msg.type === 'error') {
        setStatus('Error: ' + msg.error, 'error')
        return
      }

      if (msg.type === 'init-done') {
        const gpu: GpuStatus = !WEBGPU_REQUESTED
          ? 'off'
          : msg.webgpu
            ? 'on'
            : 'unavailable'
        if (gpu === 'unavailable')
          console.warn('[app] WebGPU unavailable — proving on CPU')
        setState((prev) => {
          // A recycled worker has no programs loaded, but proofs and verify
          // results live page-side — keep them.
          const programStates = {} as Record<ProgramName, ProgramState>
          for (const p of PROGRAMS) {
            programStates[p] = { ...prev.programStates[p], loadState: 'idle' }
          }
          return {
            ...prev,
            wasmReady: true,
            status: 'ready',
            statusText: 'Ready',
            gpu,
            programStates,
          }
        })
        return
      }

      if (msg.type === 'program-loaded') {
        const p = msg.program
        setState((prev) => ({
          ...prev,
          programStates: {
            ...prev.programStates,
            [p]: { ...prev.programStates[p], loadState: 'ready' },
          },
        }))
        log(p, 'Ready')
        programLoadResolvers.current[p]?.()
        delete programLoadResolvers.current[p]
        return
      }

      if (msg.type === 'trace') {
        const blob = new Blob([msg.trace], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `jolt-trace-${Date.now()}.json`
        a.click()
        URL.revokeObjectURL(url)
        return
      }

      if (msg.type === 'prove-done') {
        const p = msg.program
        setState((prev) => ({
          ...prev,
          status: 'ready',
          statusText: 'Proof generated!',
          programStates: {
            ...prev.programStates,
            [p]: {
              ...prev.programStates[p],
              proofBytes: msg.proof,
              programIoBytes: msg.programIo,
              verifyResult: null,
            },
          },
        }))
        const arm = gpuRef.current === 'on' ? 'GPU' : 'CPU'
        log(p, `Proof generated in ${(msg.elapsed / 1000).toFixed(2)}s (${arm})`)
        if (msg.numCycles != null)
          log(p, `RISC-V cycles: ${msg.numCycles.toLocaleString()}`)
        log(p, `Proof size: ${(msg.proofSize / 1024).toFixed(2)} KB`)
        if (msg.peakMemory != null)
          log(
            p,
            `Peak WASM memory: ${(msg.peakMemory / 1024 / 1024).toFixed(0)} MB`,
          )
        if (
          RECYCLE_ENABLED &&
          gpuRef.current === 'on' &&
          (msg.paddedCycles ?? 0) >= RECYCLE_MIN_PADDED
        ) {
          recycleRef.current()
        }
        return
      }

      if (msg.type === 'verify-done') {
        const p = msg.program
        setState((prev) => ({
          ...prev,
          status: msg.valid ? 'ready' : 'error',
          statusText: msg.valid ? 'Verification passed!' : 'Verification failed!',
          programStates: {
            ...prev.programStates,
            [p]: {
              ...prev.programStates[p],
              verifyResult: { valid: msg.valid, elapsed: msg.elapsed },
            },
          },
        }))
        log(p, `Verification completed in ${(msg.elapsed / 1000).toFixed(2)}s`)
        log(p, `Result: ${msg.valid ? 'VALID' : 'INVALID'}`)
        return
      }
    },
    [log, setStatus],
  )

  const bootWorker = useCallback(
    (statusText: string) => {
      const client = new WorkerClient(handleMessage, (e) => {
        const msg = e.message || 'Failed to load WASM module. Run: wasm-pack build --release --target web'
        setStatus(msg, 'error')
        console.error(e)
      })
      clientRef.current = client

      const numThreads = Math.min(navigator.hardwareConcurrency || 6, 8)
      setStatus(statusText, 'loading')
      client.send({
        type: 'init',
        data: { numThreads, webgpu: WEBGPU_REQUESTED ? {} : null },
      })
    },
    [handleMessage, setStatus],
  )

  const recycleWorker = useCallback(() => {
    console.log('[app] recycling prover worker after large prove')
    clientRef.current?.terminate()
    setState((prev) => ({ ...prev, wasmReady: false }))
    bootWorker('Recycling prover worker...')
  }, [bootWorker])
  recycleRef.current = recycleWorker

  useEffect(() => {
    if (!crossOriginIsolated) {
      setStatus(
        'This page requires SharedArrayBuffer support. Please open in Chrome or Safari.',
        'error',
      )
      return
    }

    const numThreads = Math.min(navigator.hardwareConcurrency || 6, 8)
    bootWorker(
      `Initializing WASM (${numThreads} threads${WEBGPU_REQUESTED ? ' + WebGPU' : ''})...`,
    )

    // clientRef, not a captured client: a recycle may have swapped the
    // worker since mount.
    return () => clientRef.current?.terminate()
  }, [bootWorker, setStatus])

  const loadProgram = useCallback(
    async (name: ProgramName) => {
      const client = clientRef.current
      if (!client) return

      setState((prev) => ({
        ...prev,
        programStates: {
          ...prev.programStates,
          [name]: { ...prev.programStates[name], loadState: 'loading' },
        },
      }))

      log(name, 'Loading preprocessing...')
      setStatus(`Loading ${name} preprocessing...`, 'loading')

      const files = PROGRAM_FILES[name]
      const [prover, verifier, elf] = await Promise.all([
        fetch(`./${files.prover}?${CACHE_BUST}`).then((r) => {
          if (!r.ok) throw new Error(`Failed to load ${files.prover}`)
          return r.arrayBuffer()
        }),
        fetch(`./${files.verifier}?${CACHE_BUST}`).then((r) => {
          if (!r.ok) throw new Error(`Failed to load ${files.verifier}`)
          return r.arrayBuffer()
        }),
        fetch(`./${files.elf}?${CACHE_BUST}`).then((r) => {
          if (!r.ok) throw new Error(`Failed to load ${files.elf}`)
          return r.arrayBuffer()
        }),
      ])

      log(
        name,
        `Prover preprocessing: ${(prover.byteLength / 1024 / 1024).toFixed(2)} MB`,
      )
      log(
        name,
        `Verifier preprocessing: ${(verifier.byteLength / 1024 / 1024).toFixed(2)} MB`,
      )
      log(name, `Guest ELF: ${(elf.byteLength / 1024).toFixed(2)} KB`)
      log(name, 'Initializing prover & verifier...')

      client.send(
        {
          type: 'load-program',
          data: {
            program: name,
            proverPreprocessing: prover,
            verifierPreprocessing: verifier,
            elfBytes: elf,
          },
        },
        [prover, verifier, elf],
      )
    },
    [log, setStatus],
  )

  const ensureProgramLoaded = useCallback(
    async (name: ProgramName): Promise<boolean> => {
      const currentState = programStatesRef.current[name]
      if (!currentState) return false

      if (currentState.loadState === 'ready') return true
      // Treat 'loading' same as 'idle' — loadState can be stale after HMR/StrictMode
      // re-mount where the worker was recreated but state was preserved.
      try {
        await loadProgram(name)
      } catch (e) {
        setState((prev) => ({
          ...prev,
          programStates: {
            ...prev.programStates,
            [name]: { ...prev.programStates[name], loadState: 'idle' },
          },
        }))
        const msg = e instanceof Error ? e.message : String(e)
        setStatus(`Error: ${msg}`, 'error')
        log(name, `Error: ${msg}`)
        return false
      }

      return new Promise<boolean>((resolve) => {
        programLoadResolvers.current[name] = () => resolve(true)
      })
    },
    [loadProgram, log, setStatus],
  )

  const proveSha2 = useCallback(
    async (message: string) => {
      const input = new TextEncoder().encode(message)
      if (input.length > SHA2_MAX_BYTES) {
        log('sha2', `Message too large: ${input.length} bytes (max ${SHA2_MAX_BYTES})`)
        return
      }
      setStatus('Loading...', 'loading')
      if (!(await ensureProgramLoaded('sha2'))) return
      setStatus('Proving...', 'proving')
      log('sha2', `\nProving SHA-256 [${input.length} bytes]`)
      clientRef.current?.send({
        type: 'prove',
        data: { program: 'sha2', input: Array.from(input) },
      })
    },
    [ensureProgramLoaded, log, setStatus],
  )

  const proveKeccak = useCallback(
    async (message: string, numIters: number) => {
      if (numIters < 1 || numIters > 100) {
        log('keccak', 'Iterations must be between 1 and 100')
        return
      }
      setStatus('Loading...', 'loading')
      const messageBytes = new TextEncoder().encode(message)
      const input = await sha256Digest(messageBytes)
      if (!(await ensureProgramLoaded('keccak'))) return
      setStatus('Proving...', 'proving')
      log('keccak', `\nProving Keccak chain("${message}", ${numIters} iters)`)
      clientRef.current?.send({
        type: 'prove',
        data: { program: 'keccak', input: Array.from(input), numIters },
      })
    },
    [ensureProgramLoaded, log, setStatus],
  )

  const verify = useCallback(
    async (program: ProgramName) => {
      const ps = state.programStates[program]
      if (!ps?.proofBytes || !ps?.programIoBytes) {
        log(program, 'No proof to verify. Generate a proof first.')
        return
      }
      // A post-prove recycle leaves the fresh worker with no verifier;
      // re-load (HTTP-cached) before posting.
      if (!(await ensureProgramLoaded(program))) return
      setStatus('Verifying...', 'proving')
      log(program, '\nStarting verification...')
      clientRef.current?.send({
        type: 'verify',
        data: {
          program,
          proof: ps.proofBytes,
          programIo: ps.programIoBytes,
        },
      })
    },
    [state.programStates, ensureProgramLoaded, log, setStatus],
  )

  const downloadTrace = useCallback(() => {
    clientRef.current?.send({ type: 'get-trace' })
  }, [])

  return {
    status: state.status,
    statusText: state.statusText,
    wasmReady: state.wasmReady,
    gpu: state.gpu,
    programStates: state.programStates,
    outputLogs: state.outputLogs,
    proveSha2,
    proveKeccak,
    verify,
    downloadTrace,
  }
}
