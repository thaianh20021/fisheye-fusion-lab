#!/usr/bin/env node
import { main } from '../src/cli.js';

const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 18 || (major === 18 && minor < 3)) {
  console.error(`fflab needs Node.js 18.3 or newer (found ${process.version})`);
  process.exit(1);
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code ?? 0),
  (err) => {
    console.error(`fflab: error: ${err.message}`);
    if (process.env.FFLAB_DEBUG) console.error(err.stack);
    process.exit(1);
  },
);
