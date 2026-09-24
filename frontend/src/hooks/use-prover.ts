import { useState, useRef, useCallback, useEffect } from 'react'
import type {
  ProgramName,
  ProgramState,
  AppStatus,
  WorkerResponse,
  GpuInfo,
  ProveMode,
} from '@/lib/types'
import {
  PROGRAMS,
  PROGRAM_FILES,
  CACHE_BUST,
  SHA2_MAX_BYTES,
  ECDSA_TEST_VECTOR,
  MODE_STORAGE_KEY,
} from '@/lib/constants'
import { WorkerClient } from '@/lib/worker-client'

interface ProverState {
  status: AppStatus
  statusText: string
  wasmReady: boolean
  programStates: Record<ProgramName, ProgramState>
  outputLogs: Record<ProgramName, string>
  gpu: GpuInfo | null
  mode: ProveMode
  // Why the GPU is not selectable (null when it is).
  modeReason: string | null
}

function initialProgramStates(): Record<ProgramName, ProgramState> {
  const states = {} as Record<ProgramName, ProgramState>
  for (const p of PROGRAMS) {
    states[p] = {
      loadState: 'idle',
      proofBytes: null,
      programIoBytes: null,
      verifierPreprocessingBytes: null,
      verifyResult: null,
      lastProof: null,
    }
  }
  return states
}

async function sha256Digest(bytes: Uint8Array): Promise<Uint8Array> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', bytes.buffer as ArrayBuffer)
  return new Uint8Array(hashBuffer)
}

const hex = (bytes: Uint8Array) =>
  Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')

function storedMode(): ProveMode | null {
  const v = localStorage.getItem(MODE_STORAGE_KEY)
  return v === 'gpu' || v === 'cpu' ? v : null
}

function gpuState(gpu: GpuInfo): Pick<ProverState, 'gpu' | 'mode' | 'modeReason'> {
  const ok = gpu.status === 'ok'
  const chosenOff = gpu.status === 'disabled'
  return {
    gpu,
    mode: ok ? 'gpu' : 'cpu',
    modeReason: ok || chosenOff ? null : (gpu.reason ?? gpu.status),
  }
}

const ms = (v: number) => `${Math.round(v)} ms`

export function useProver() {
  const [state, setState] = useState<ProverState>({
    status: 'loading',
    statusText: 'Initializing WASM...',
    wasmReady: false,
    programStates: initialProgramStates(),
    outputLogs: { sha2: '', ecdsa: '', keccak: '' },
    gpu: null,
    mode: 'cpu',
    modeReason: null,
  })

  const clientRef = useRef<WorkerClient | null>(null)
  const programLoadResolvers = useRef<Record<string, () => void>>({})
  const programStatesRef = useRef(state.programStates)
  programStatesRef.current = state.programStates

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
        const gpu = msg.gpu
        setState((prev) => ({
          ...prev,
          status: 'error',
          statusText: 'Error: ' + msg.error,
          ...(gpu ? gpuState(gpu) : {}),
        }))
        return
      }

      if (msg.type === 'init-done') {
        setState((prev) => ({
          ...prev,
          wasmReady: true,
          status: 'ready',
          statusText: 'Ready',
          programStates: initialProgramStates(),
          ...gpuState(msg.gpu),
        }))
        return
      }

      if (msg.type === 'gpu-status') {
        setState((prev) => ({
          ...prev,
          status: prev.status === 'loading' ? 'ready' : prev.status,
          statusText: prev.status === 'loading' ? 'Ready' : prev.statusText,
          ...gpuState(msg.gpu),
        }))
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
        const mode: ProveMode = msg.gpuStatus === 'ok' ? 'gpu' : 'cpu'
        const stage = (json: string): number | undefined =>
          json ? (JSON.parse(json) as { total_ms: number }).total_ms : undefined
        const lastProof = {
          mode,
          proveMs: msg.proveMs,
          totalMs: msg.elapsed,
          commitMs: stage(msg.gpuCommit),
          digitRangeMs: stage(msg.gpuDigitRange),
          stage2Ms: stage(msg.gpuStage2),
        }
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
              verifierPreprocessingBytes: msg.verifierPreprocessing,
              verifyResult: null,
              lastProof,
            },
          },
        }))
        log(p, `Proof generated in ${(msg.elapsed / 1000).toFixed(2)}s`)
        log(
          p,
          `  trace ${(msg.traceMs / 1000).toFixed(2)}s · Akita setup ${(msg.setupMs / 1000).toFixed(2)}s · prove ${(msg.proveMs / 1000).toFixed(2)}s`,
        )
        log(
          p,
          `  mode ${mode.toUpperCase()}` +
            (lastProof.commitMs != null ? ` · commit ${ms(lastProof.commitMs)}` : '') +
            (lastProof.digitRangeMs != null ? ` · digit range ${ms(lastProof.digitRangeMs)}` : '') +
            (lastProof.stage2Ms != null ? ` · stage 2 ${ms(lastProof.stage2Ms)}` : ''),
        )
        sha256Digest(msg.proof).then((d) => log(p, `Proof SHA-256: ${hex(d)}`))
        if (msg.numCycles != null)
          log(p, `RISC-V cycles: ${msg.numCycles.toLocaleString()}`)
        log(p, `Proof size: ${(msg.proofSize / 1024).toFixed(2)} KB`)
        if (msg.paddedCycles != null)
          log(p, `Padded trace length: 2^${Math.log2(msg.paddedCycles)}`)
        if (msg.peakMemory != null)
          log(
            p,
            `Peak WASM memory: ${(msg.peakMemory / 1024 / 1024).toFixed(0)} MB`,
          )
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

  useEffect(() => {
    if (!crossOriginIsolated) {
      setStatus(
        'This page requires SharedArrayBuffer support. Please open in Chrome or Safari.',
        'error',
      )
      return
    }

    const client = new WorkerClient(handleMessage, (e) => {
      const msg = e.message || 'Failed to load WASM module. Run: wasm-pack build --release --target web'
      setStatus(msg, 'error')
      console.error(e)
    })
    clientRef.current = client

    const numThreads = Math.min(navigator.hardwareConcurrency || 6, 8)
    setStatus(`Initializing WASM (${numThreads} threads)...`, 'loading')
    // GPU unless the user chose CPU; the worker reports why when it cannot.
    const gpu = storedMode() !== 'cpu'
    client.send({ type: 'init', data: { numThreads, cacheBust: CACHE_BUST, gpu } })

    return () => client.terminate()
  }, [handleMessage, setStatus])

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
      const [program, elf] = await Promise.all([
        fetch(`./${files.program}?${CACHE_BUST}`).then((r) => {
          if (!r.ok) throw new Error(`Failed to load ${files.program}`)
          return r.arrayBuffer()
        }),
        fetch(`./${files.elf}?${CACHE_BUST}`).then((r) => {
          if (!r.ok) throw new Error(`Failed to load ${files.elf}`)
          return r.arrayBuffer()
        }),
      ])
      log(
        name,
        `Program preprocessing: ${(program.byteLength / 1024 / 1024).toFixed(2)} MB`,
      )
      log(name, `Guest ELF: ${(elf.byteLength / 1024).toFixed(2)} KB`)
      log(name, 'Initializing prover...')
      client.send(
        {
          type: 'load-program',
          data: {
            program: name,
            programPreprocessing: program,
            elfBytes: elf,
          },
        },
        [program, elf],
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

  const proveEcdsa = useCallback(async () => {
    setStatus('Loading...', 'loading')
    if (!(await ensureProgramLoaded('ecdsa'))) return
    setStatus('Proving...', 'proving')
    log(
      'ecdsa',
      `\nProving secp256k1 ECDSA verify ("${ECDSA_TEST_VECTOR.message}", 1 signature, inline)`,
    )
    const { z, r, s, q } = ECDSA_TEST_VECTOR
    clientRef.current?.send({ type: 'prove', data: { program: 'ecdsa', z, r, s, q } })
  }, [ensureProgramLoaded, log, setStatus])

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
    (program: ProgramName) => {
      const ps = state.programStates[program]
      if (!ps?.proofBytes || !ps?.programIoBytes || !ps?.verifierPreprocessingBytes) {
        log(program, 'No proof to verify. Generate a proof first.')
        return
      }
      setStatus('Verifying...', 'proving')
      log(program, '\nStarting verification...')
      clientRef.current?.send({
        type: 'verify',
        data: {
          program,
          proof: ps.proofBytes,
          programIo: ps.programIoBytes,
          verifierPreprocessing: ps.verifierPreprocessingBytes,
        },
      })
    },
    [state.programStates, log, setStatus],
  )

  const downloadTrace = useCallback(() => {
    clientRef.current?.send({ type: 'get-trace' })
  }, [])

  const setMode = useCallback(
    (mode: ProveMode) => {
      localStorage.setItem(MODE_STORAGE_KEY, mode)
      if (mode === 'gpu') setStatus('Enabling GPU...', 'loading')
      clientRef.current?.send({ type: 'set-gpu', data: { enabled: mode === 'gpu' } })
    },
    [setStatus],
  )

  return {
    status: state.status,
    statusText: state.statusText,
    wasmReady: state.wasmReady,
    programStates: state.programStates,
    outputLogs: state.outputLogs,
    gpu: state.gpu,
    mode: state.mode,
    modeReason: state.modeReason,
    setMode,
    proveSha2,
    proveEcdsa,
    proveKeccak,
    verify,
    downloadTrace,
  }
}
