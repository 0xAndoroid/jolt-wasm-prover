import type { GpuStatus } from '@/lib/types'
import { cn } from '@/lib/utils'

const gpuConfig: Record<
  Exclude<GpuStatus, 'pending'>,
  { label: string; title: string; dot: string; badge: string }
> = {
  on: {
    label: 'GPU',
    title: 'WebGPU acceleration active',
    dot: 'bg-sky-400',
    badge: 'border-sky-500/30 bg-sky-500/10 text-sky-400',
  },
  unavailable: {
    label: 'CPU',
    title: 'WebGPU unavailable — proving on CPU',
    dot: 'bg-muted-foreground',
    badge: 'border-border bg-muted text-muted-foreground',
  },
  off: {
    label: 'CPU',
    title: 'WebGPU disabled (?webgpu=0)',
    dot: 'bg-muted-foreground',
    badge: 'border-border bg-muted text-muted-foreground',
  },
}

export function GpuBadge({ gpu, className }: { gpu: GpuStatus; className?: string }) {
  if (gpu === 'pending') return null
  const config = gpuConfig[gpu]
  return (
    <span
      id="gpu-badge"
      title={config.title}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium',
        config.badge,
        className,
      )}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full', config.dot)} />
      {config.label}
    </span>
  )
}
