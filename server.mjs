import http from 'http';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';

const DIST_ROOT = path.resolve('frontend/dist');
const PKG_ROOT = path.resolve('pkg');

const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.css': 'text/css',
  '.bin': 'application/octet-stream',
  '.elf': 'application/octet-stream',
  '.png': 'image/png',
};

const MAX_CACHE_ENTRIES = 64;
const compressCache = new Map();

function getCompressed(filePath, content) {
  if (compressCache.has(filePath)) return compressCache.get(filePath);

  if (compressCache.size >= MAX_CACHE_ENTRIES) {
    const oldest = compressCache.keys().next().value;
    compressCache.delete(oldest);
  }

  const entry = {
    br: zlib.brotliCompressSync(content, {
      params: { [zlib.constants.BROTLI_PARAM_QUALITY]: 6 },
    }),
    gzip: zlib.gzipSync(content),
    raw: content,
  };
  compressCache.set(filePath, entry);
  return entry;
}

const securityHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'Content-Security-Policy': [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "img-src 'self' data:",
  ].join('; '),
};

function setSecurityHeaders(res) {
  for (const [k, v] of Object.entries(securityHeaders)) {
    res.setHeader(k, v);
  }
}

const server = http.createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405);
    res.end();
    return;
  }

  setSecurityHeaders(res);

  let filePath = req.url.split('?')[0];

  if (filePath === '/pkg/' || filePath === '/pkg') {
    res.writeHead(302, { 'Location': '/pkg/jolt_wasm_prover.js' });
    res.end();
    return;
  }

  if (filePath === '/') filePath = '/frontend/dist/index.html';
  else if (!filePath.startsWith('/pkg')) filePath = '/frontend/dist' + filePath;

  filePath = '.' + filePath;

  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(DIST_ROOT + path.sep) && !resolved.startsWith(PKG_ROOT + path.sep)
      && resolved !== DIST_ROOT && resolved !== PKG_ROOT) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const ext = path.extname(resolved);
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  fs.readFile(resolved, (err, content) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    res.setHeader('Cache-Control', 'public, max-age=86400, immutable');

    const compressed = getCompressed(resolved, content);
    const acceptEncoding = req.headers['accept-encoding'] || '';

    if (acceptEncoding.includes('br')) {
      res.writeHead(200, { 'Content-Type': contentType, 'Content-Encoding': 'br' });
      res.end(compressed.br);
    } else if (acceptEncoding.includes('gzip')) {
      res.writeHead(200, { 'Content-Type': contentType, 'Content-Encoding': 'gzip' });
      res.end(compressed.gzip);
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(compressed.raw);
    }
  });
});

server.listen(8080, '127.0.0.1', () => console.log('http://localhost:8080'));
