import { ECDSA_TEST_VECTOR } from '@/lib/constants'

const beHex = (limbs: string[]) =>
  limbs
    .slice()
    .reverse()
    .map((l) => l.slice(2).padStart(16, '0'))
    .join('')

const { message, z, r, s, q } = ECDSA_TEST_VECTOR
const FIELDS: [string, string][] = [
  ['Message', message],
  ['Message hash z = SHA-256(message)', beHex(z)],
  ['Signature r', beHex(r)],
  ['Signature s', beHex(s)],
  ['Public key Q.x', beHex(q.slice(0, 4))],
  ['Public key Q.y', beHex(q.slice(4, 8))],
]

export function EcdsaInputs() {
  return (
    <div className="flex flex-col gap-4">
      <dl className="flex flex-col gap-4">
        {FIELDS.map(([label, value]) => (
          <div key={label} className="flex flex-col gap-1">
            <dt className="text-sm font-medium">{label}</dt>
            <dd className="break-all font-mono text-xs text-muted-foreground">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-muted-foreground">
        Fixed secp256k1 test vector; the curve arithmetic runs in the{' '}
        <code>jolt-inlines-secp256k1</code> inline. A proof that verifies means the
        signature is valid — an invalid signature spoils the proof so it cannot verify.
      </p>
    </div>
  )
}
