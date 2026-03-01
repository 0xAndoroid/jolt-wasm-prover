import { useRef } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { StatusBadge } from '@/components/status-badge'
import { Sha2Inputs } from '@/components/sha2-inputs'
import { KeccakInputs } from '@/components/keccak-inputs'
import { ProgramPanel } from '@/components/program-panel'
import { useProver } from '@/hooks/use-prover'

export function App() {
  const {
    status,
    statusText,
    wasmReady,
    programStates,
    outputLogs,
    proveSha2,
    proveKeccak,
    verify,
    downloadTrace,
  } = useProver()

  const sha2MessageRef = useRef('Hello, world!')
  const keccakMessageRef = useRef('Hello, world!')
  const keccakItersRef = useRef(3)

  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-4xl items-center gap-3 px-4 py-3">
          <img src="/favicon.png" alt="Jolt" className="h-6 w-6" />
          <span className="text-sm font-semibold font-mono tracking-tight">
            Jolt WASM Prover
          </span>
          <StatusBadge status={status} className="ml-auto" />
        </div>
      </header>

      <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-8 md:py-12">
        <div className="mb-8">
          <h1 className="text-2xl font-bold font-mono md:text-3xl">
            Zero-Knowledge Prover
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Generate and verify ZK proofs in-browser using Jolt zkVM compiled to
            WebAssembly.
          </p>
        </div>

        {status === 'error' && (
          <div className="mb-6 rounded-md border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {statusText}
          </div>
        )}

        <Tabs defaultValue="sha2">
          <TabsList variant="folder">
            <TabsTrigger value="sha2">SHA-256</TabsTrigger>
            <TabsTrigger value="keccak">Keccak Chain</TabsTrigger>
          </TabsList>

          <TabsContent value="sha2" id="page-sha2">
            <ProgramPanel
              program="sha2"
              programState={programStates.sha2}
              appStatus={status}
              wasmReady={wasmReady}
              output={outputLogs.sha2}
              onProve={() => proveSha2(sha2MessageRef.current)}
              onVerify={() => verify('sha2')}
              onTrace={downloadTrace}
            >
              <Sha2Inputs
                onMessageChange={(msg) => {
                  sha2MessageRef.current = msg
                }}
              />
            </ProgramPanel>
          </TabsContent>

          <TabsContent value="keccak" id="page-keccak">
            <ProgramPanel
              program="keccak"
              programState={programStates.keccak}
              appStatus={status}
              wasmReady={wasmReady}
              output={outputLogs.keccak}
              onProve={() =>
                proveKeccak(keccakMessageRef.current, keccakItersRef.current)
              }
              onVerify={() => verify('keccak')}
              onTrace={downloadTrace}
            >
              <KeccakInputs
                onInputsChange={(msg, iters) => {
                  keccakMessageRef.current = msg
                  keccakItersRef.current = iters
                }}
              />
            </ProgramPanel>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
