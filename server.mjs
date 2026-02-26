import http from 'http';
import fs from 'fs';
import path from 'path';

const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.wasm': 'application/wasm',
  '.json': 'application/json',
  '.css': 'text/css',
  '.bin': 'application/octet-stream',
  '.elf': 'application/octet-stream',
};

const server = http.createServer((req, res) => {
  let filePath = req.url.split('?')[0];

  // wasm-bindgen-rayon's workerHelpers.js does `import('../../..')` which resolves
  // to /pkg/ — redirect to the actual JS module so import.meta.url is correct
  if (filePath === '/pkg/' || filePath === '/pkg') {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.writeHead(302, { 'Location': '/pkg/jolt_wasm_prover.js' });
    res.end();
    return;
  }

  if (filePath === '/') filePath = '/www/index.html';
  else if (!filePath.startsWith('/pkg')) filePath = '/www' + filePath;

  filePath = '.' + filePath;

  const ext = path.extname(filePath);
  const contentType = mimeTypes[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, content) => {
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

    if (err) {
      res.writeHead(404);
      res.end('Not found: ' + filePath);
    } else {
      if (ext === '.wasm') {
        res.setHeader('Cache-Control', 'no-cache');
      } else {
        res.setHeader('Cache-Control', 'no-store');
      }
      res.setHeader('CDN-Cache-Control', 'public, max-age=86400');
      res.writeHead(200, { 'Content-Type': contentType });
      res.end(content);
    }
  });
});

server.listen(8080, () => console.log('http://localhost:8080'));
