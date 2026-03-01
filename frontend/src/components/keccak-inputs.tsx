import { useState } from 'react'
import { Input } from '@/components/ui/input'

export function KeccakInputs({
  onInputsChange,
}: {
  onInputsChange: (message: string, iterations: number) => void
}) {
  const [message, setMessage] = useState('Hello, world!')
  const [iterations, setIterations] = useState(3)

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label htmlFor="keccak-message" className="text-sm font-medium">
          Message
        </label>
        <Input
          id="keccak-message"
          value={message}
          onChange={(e) => {
            setMessage(e.target.value)
            onInputsChange(e.target.value, iterations)
          }}
          placeholder="Enter a message..."
          className="font-mono text-sm"
        />
      </div>
      <div className="flex flex-col gap-2">
        <label htmlFor="keccak-iters" className="text-sm font-medium">
          Iterations
        </label>
        <Input
          id="keccak-iters"
          type="number"
          min={1}
          max={100}
          value={iterations}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10)
            setIterations(val)
            onInputsChange(message, val)
          }}
          className="w-24 font-mono text-sm"
        />
        <p className="text-xs text-muted-foreground">
          Number of Keccak-256 hash iterations (1-100)
        </p>
      </div>
    </div>
  )
}
