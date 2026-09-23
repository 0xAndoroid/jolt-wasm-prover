import { Button } from '@/components/ui/button'
import type { ProveMode } from '@/lib/types'
import { cn } from '@/lib/utils'

const OPTIONS: { value: ProveMode; label: string }[] = [
  { value: 'gpu', label: 'GPU (GPU + CPU)' },
  { value: 'cpu', label: 'CPU only' },
]

// Neutral segmented control: the mode is a fact, not a verdict, so no accent.
export function ModeSelector({
  mode,
  gpuAvailable,
  reason,
  disabled,
  onChange,
  className,
}: {
  mode: ProveMode
  gpuAvailable: boolean
  reason: string | null
  disabled: boolean
  onChange: (mode: ProveMode) => void
  className?: string
}) {
  return (
    <div className={cn('flex min-w-0 flex-col items-start gap-1 sm:items-end', className)}>
      <div
        role="radiogroup"
        aria-label="Prover mode"
        className="inline-flex rounded-md border border-border bg-background p-0.5"
      >
        {OPTIONS.map((o) => {
          const selected = mode === o.value
          return (
            <Button
              key={o.value}
              role="radio"
              aria-checked={selected}
              size="sm"
              variant={selected ? 'outline' : 'ghost'}
              className={cn('h-7 px-2.5 text-xs', !selected && 'text-muted-foreground')}
              disabled={disabled || (o.value === 'gpu' && !gpuAvailable)}
              onClick={() => !selected && onChange(o.value)}
            >
              {o.label}
            </Button>
          )
        })}
      </div>
      {reason && (
        <span className="max-w-full truncate text-xs text-muted-foreground" title={reason}>
          GPU unavailable: {reason}
        </span>
      )}
    </div>
  )
}
