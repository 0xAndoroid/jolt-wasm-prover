import { useRef, useEffect, useState } from 'react'
import { cn } from '@/lib/utils'

export function OutputLog({
  content,
  className,
}: {
  content: string
  className?: string
}) {
  const preRef = useRef<HTMLPreElement>(null)
  const [showPlaceholder, setShowPlaceholder] = useState(true)
  const [animatingOut, setAnimatingOut] = useState(false)

  useEffect(() => {
    if (preRef.current) {
      preRef.current.scrollTop = preRef.current.scrollHeight
    }
  }, [content])

  useEffect(() => {
    if (content && showPlaceholder && !animatingOut) {
      setAnimatingOut(true)
    }
  }, [content, showPlaceholder, animatingOut])

  return (
    <pre
      ref={preRef}
      className={cn(
        'output relative min-h-[120px] max-h-[280px] overflow-auto bg-background/50 p-4 font-mono text-xs leading-relaxed text-muted-foreground rounded-b-md',
        className,
      )}
    >
      {showPlaceholder && (
        <span
          className={cn(
            'absolute inset-0 flex items-center justify-center text-muted-foreground/50',
            animatingOut && 'animate-out fade-out duration-300',
          )}
          onAnimationEnd={() => setShowPlaceholder(false)}
        >
          Output will appear here...
        </span>
      )}
      {content}
    </pre>
  )
}
