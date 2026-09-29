#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == --staged ]]; then
    git diff --cached --name-only --diff-filter=ACMRD -- '*.ts' '*.tsx' '*.js' '*.mjs' '*.css' '*.html' '*package*.json' '*tsconfig*.json' '.oxlintrc.json' 'knip.jsonc' '.htmlvalidate.json' 'scripts/lint/web.sh' | grep -q . || exit 0
    command -v node > /dev/null 2>&1 && [[ -x frontend/node_modules/.bin/oxlint && -x frontend/node_modules/.bin/tsc && -x frontend/node_modules/.bin/knip && -x frontend/node_modules/.bin/stylelint && -x frontend/node_modules/.bin/html-validate ]] || exit 0
fi
cd frontend
npm run -s lint:types
npm run -s lint
npm run -s lint:knip
npm run -s lint:css
npm run -s lint:html
