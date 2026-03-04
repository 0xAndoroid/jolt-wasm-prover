import http from 'http';
import fs from 'fs';
import path from 'path';
import zlib from 'zlib';

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

// Pre-compressed file cache: filePath -> { br: Buffer, gzip: Buffer }
const compressCache = new Map();

function getCompressed(filePath, content) {
  if (compressCache.has(filePath)) return compressCache.get(filePath);

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

const server = http.createServer((req, res) => {
  let filePath = req.url.split('?')[0];

  // wasm-bindgen-rayon's workerHelpers.js does `import('../../..')` which resolves
  // to /pkg/ — redirect to the actual JS module so import.meta.url is correct
  if (filePath === '/pkg/' || filePath === '/pkg') {

    res.writeHead(302, { 'Location': '/pkg/jolt_wasm_prover.js' });
    res.end();
    return;
  }

  if (filePath === '/') filePath = '/frontend/dist/index.html';
  else if (!filePath.startsWith('/pkg')) filePath = '/frontend/dist' + filePath;

  filePath = '.' + filePath;

  const ext = path.extname(filePath);
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {

    if (err) {
      res.writeHead(404);
      res.end('Not found: ' + filePath);
      return;
    }


    res.setHeader('Cache-Control', 'public, max-age=86400, immutable');

    const compressed = getCompressed(filePath, content);
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

server.listen(8080, () => console.log('http://localhost:8080'));
