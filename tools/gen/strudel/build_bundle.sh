#!/usr/bin/env bash
# Rebuild dist/strudel.bundle.js - Strudel's pattern engine (@strudel/core,
# mini, tonal, transpiler) as ONE browser-style script that lib/gen runs
# inside the Python process via an embedded V8 (mini-racer). The show box
# needs neither node nor npm at runtime; only whoever rebuilds this does.
#   cd tools/gen/strudel && ./build_bundle.sh
set -euo pipefail
cd "$(dirname "$0")"
npm install --silent
npm install --silent --no-save esbuild
cat > .entry.mjs <<'JS'
import * as core from '@strudel/core';
import * as mini from '@strudel/mini';
import * as tonal from '@strudel/tonal';
import { transpiler } from '@strudel/transpiler';
globalThis.__strudel = { core, mini, tonal, transpiler };
JS
npx esbuild .entry.mjs --bundle --format=iife --platform=browser --outfile=dist/strudel.bundle.js --log-level=warning
rm -f .entry.mjs
echo "wrote dist/strudel.bundle.js ($(wc -c < dist/strudel.bundle.js) bytes) from @strudel/core $(node -p "require('@strudel/core/package.json').version")"
