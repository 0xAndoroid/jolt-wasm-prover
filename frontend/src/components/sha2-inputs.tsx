import { useState } from 'react'
import { Textarea } from '@/components/ui/textarea'
import { SHA2_MAX_BYTES } from '@/lib/constants'

export function Sha2Inputs({
  onMessageChange,
}: {
  onMessageChange: (message: string) => void
}) {
  const [message, setMessage] = useState('Hello, world!')
  const byteLen = new TextEncoder().encode(message).length
  const overLimit = byteLen > SHA2_MAX_BYTES

  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="sha2-message" className="text-sm font-medium">
        Message
      </label>
      <Textarea
        id="sha2-message"
        value={message}
        onChange={(e) => {
          setMessage(e.target.value)
          onMessageChange(e.target.value)
        }}
        placeholder="Enter a message to hash..."
        className="min-h-[80px] resize-none font-mono text-sm"
      />
      <p
        id="sha2-byte-counter"
        className={`text-xs ${overLimit ? 'font-medium text-destructive' : 'text-muted-foreground'}`}
      >
        {byteLen} / {SHA2_MAX_BYTES} bytes
      </p>
    </div>
  )
}
