import { Button } from '@/components/ui/button'
import { OutputLog } from './output-log'
import type { ProgramName, ProgramState, AppStatus } from '@/lib/types'

export function ProgramPanel({
  program,
  programState,
  appStatus,
  wasmReady,
  output,
  onProve,
  onVerify,
  onTrace,
  children,
}: {
  program: ProgramName
  programState: ProgramState
  appStatus: AppStatus
  wasmReady: boolean
  output: string
  onProve: () => void
  onVerify: () => void
  onTrace: () => void
  children: React.ReactNode
}) {
  const isReady = programState.loadState === 'ready' || wasmReady
  const hasProof = programState.proofBytes !== null
  const isBusy = appStatus === 'proving' || appStatus === 'loading'

  return (
    <div className="rounded-b-md rounded-tr-md border border-border bg-card">
      <div className="p-5">
        {children}
      </div>

      <div className="flex items-center gap-2 border-t border-border px-5 py-3">
        <Button
          size="sm"
          className="prove-btn"
          onClick={onProve}
          disabled={!wasmReady || isBusy}
        >
          Generate Proof
        </Button>
        <div className="flex-1" />
        <Button
          size="sm"
          variant="ghost"
          className="trace-btn text-muted-foreground"
          onClick={onTrace}
          disabled={!wasmReady}
        >
          Download Trace
        </Button>
      </div>

      <div className={`border-t border-border ${appStatus === 'proving' ? 'proving-shimmer' : ''}`}>
        <OutputLog content={output} />
      </div>

      {hasProof && (
        <div className="flex items-center border-t border-border px-5 py-3 animate-in fade-in duration-300">
          <Button
            size="sm"
            variant="outline"
            className="verify-btn"
            onClick={onVerify}
            disabled={!isReady || isBusy}
          >
            Verify Proof
          </Button>
          <div className="flex-1" />
          {programState.verifyResult && (
            <span className="flex items-center gap-2">
              <span className={`text-sm font-medium ${programState.verifyResult.valid ? 'text-emerald-600' : 'text-destructive'}`}>
                {programState.verifyResult.valid ? 'Valid' : 'Invalid'}
              </span>
              <span className="text-xs text-muted-foreground">
                {(programState.verifyResult.elapsed / 1000).toFixed(2)}s
              </span>
            </span>
          )}
        </div>
      )}
    </div>
  )
}
