import { useState, useRef, useCallback, useEffect } from 'react'
import type {
  ProgramName,
  ProgramState,
  AppStatus,
  WorkerResponse,
} from '@/lib/types'
import { PROGRAMS, PROGRAM_FILES, CACHE_BUST, SHA2_MAX_BYTES } from '@/lib/constants'
import { WorkerClient } from '@/lib/worker-client'

interface ProverState {
  status: AppStatus
  statusText: string
  wasmReady: boolean
  programStates: Record<ProgramName, ProgramState>
  outputLogs: Record<ProgramName, string>
}

function initialProgramStates(): Record<ProgramName, ProgramState> {
  const states = {} as Record<ProgramName, ProgramState>
  for (const p of PROGRAMS) {
    states[p] = { loadState: 'idle', proofBytes: null, programIoBytes: null }
  }
  return states
}

async function sha256Digest(bytes: Uint8Array): Promise<Uint8Array> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', bytes)
  return new Uint8Array(hashBuffer)
}

export function useProver() {
  const [state, setState] = useState<ProverState>({
    status: 'loading',
    statusText: 'Initializing WASM...',
    wasmReady: false,
    programStates: initialProgramStates(),
    outputLogs: { sha2: '', keccak: '' },
  })

  const clientRef = useRef<WorkerClient | null>(null)
  const programLoadResolvers = useRef<Record<string, () => void>>({})

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
        setState((prev) => ({
          ...prev,
          wasmReady: true,
          status: 'ready',
          statusText: 'Ready',
          programStates: initialProgramStates(),
        }))
        return
      }

      if (msg.type === 'program-loaded') {
        const p = msg.program
        setState((prev) => ({
          ...prev,
          status: 'ready',
          statusText: 'Ready',
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
            },
          },
        }))
        log(p, `Proof generated in ${(msg.elapsed / 1000).toFixed(2)}s`)
        if (msg.numCycles != null)
          log(p, `RISC-V cycles: ${msg.numCycles.toLocaleString()}`)
        log(p, `Proof size: ${(msg.proofSize / 1024).toFixed(2)} KB`)
        log(
          p,
          `Proof size (compressed): ${(msg.compressedProofSize / 1024).toFixed(2)} KB`,
        )
        if (msg.peakMemory != null)
          log(
            p,
            `Peak WASM memory: ${(msg.peakMemory / 1024 / 1024).toFixed(0)} MB`,
          )
        return
      }

      if (msg.type === 'verify-done') {
        const p = msg.program
        log(p, `Verification completed in ${(msg.elapsed / 1000).toFixed(2)}s`)
        log(p, `Result: ${msg.valid ? 'VALID' : 'INVALID'}`)
        setStatus(
          msg.valid ? 'Verification passed!' : 'Verification failed!',
          msg.valid ? 'ready' : 'error',
        )
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
    client.send({ type: 'init', data: { numThreads } })

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
      // Need to read current state outside of setState
      let currentState: ProgramState | undefined
      setState((prev) => {
        currentState = prev.programStates[name]
        return prev
      })
      if (!currentState) return false

      if (currentState.loadState === 'ready') return true
      if (currentState.loadState === 'idle') {
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
      if (!ps?.proofBytes || !ps?.programIoBytes) {
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
        },
      })
    },
    [state.programStates, log, setStatus],
  )

  const downloadTrace = useCallback(() => {
    clientRef.current?.send({ type: 'get-trace' })
  }, [])

  return {
    status: state.status,
    statusText: state.statusText,
    wasmReady: state.wasmReady,
    programStates: state.programStates,
    outputLogs: state.outputLogs,
    proveSha2,
    proveKeccak,
    verify,
    downloadTrace,
  }
}
