import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

function servePkg(): Plugin {
  const pkgDir = path.resolve(__dirname, '..', 'pkg')
  return {
    name: 'serve-pkg',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url?.split('?')[0]
        if (url === '/pkg/' || url === '/pkg') {
          res.writeHead(302, { Location: '/pkg/jolt_wasm_prover.js' })
          res.end()
          return
        }
        if (url?.startsWith('/pkg/')) {
          const fileName = url.slice('/pkg/'.length)
          const filePath = path.resolve(pkgDir, fileName)
          if (!filePath.startsWith(pkgDir + path.sep) && filePath !== pkgDir) {
            res.writeHead(403, { 'Content-Type': 'text/plain' })
            res.end('Forbidden')
            return
          }
          if (fs.existsSync(filePath)) {
            const ext = path.extname(filePath)
            const mimeTypes: Record<string, string> = {
              '.js': 'application/javascript',
              '.wasm': 'application/wasm',
              '.d.ts': 'text/plain',
            }
            res.setHeader(
              'Content-Type',
              mimeTypes[ext] || 'application/octet-stream',
            )
            res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
            res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp')
            fs.createReadStream(filePath).pipe(res)
            return
          }
          res.writeHead(404, { 'Content-Type': 'text/plain' })
          res.end('Not found. Run: wasm-pack build --release --target web')
          return
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), servePkg()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 8080,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
