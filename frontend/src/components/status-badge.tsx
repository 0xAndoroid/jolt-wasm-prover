import type { AppStatus } from '@/lib/types'
import { cn } from '@/lib/utils'

const statusConfig: Record<AppStatus, { label: string; dot: string; badge: string }> = {
  loading: {
    label: 'Loading',
    dot: 'bg-muted-foreground',
    badge: 'border-border bg-muted text-muted-foreground',
  },
  ready: {
    label: 'Ready',
    dot: 'bg-emerald-400',
    badge: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400',
  },
  proving: {
    label: 'Working',
    dot: 'bg-primary',
    badge: 'border-primary/30 bg-primary/10 text-primary',
  },
  error: {
    label: 'Error',
    dot: 'bg-destructive',
    badge: 'border-destructive/30 bg-destructive/10 text-destructive',
  },
}

export function StatusBadge({
  status,
  className,
}: {
  status: AppStatus
  className?: string
}) {
  const config = statusConfig[status]
  return (
    <span
      id="status"
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium',
        config.badge,
        status,
        className,
      )}
    >
      <span
        className={cn(
          'h-1.5 w-1.5 rounded-full',
          config.dot,
          status === 'loading' && 'animate-pulse',
          status === 'proving' && 'animate-[dot-glow_2s_ease-in-out_infinite]',
        )}
      />
      {config.label}
    </span>
  )
}
