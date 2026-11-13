// A small argv parser: --key value, --key=value, --flag, --no-flag, repeated or comma-separated lists.
// Booleans must be declared so that "--sky data" never swallows a positional.

export function parseArgs(argv, { booleans = new Set(), lists = new Set(), aliases = {} } = {}) {
  const out = { _: [], opts: {} };
  const put = (key, value) => {
    if (lists.has(key)) {
      const parts = String(value).split(',').map((s) => s.trim()).filter(Boolean);
      out.opts[key] = [...(out.opts[key] || []), ...parts];
    } else {
      out.opts[key] = value;
    }
  };
  for (let i = 0; i < argv.length; i++) {
    let a = argv[i];
    if (a === '--') {
      out._.push(...argv.slice(i + 1));
      break;
    }
    if (/^-[a-zA-Z]$/.test(a) && aliases[a.slice(1)]) a = `--${aliases[a.slice(1)]}`;
    if (!a.startsWith('--') || a === '--') {
      out._.push(a);
      continue;
    }
    const body = a.slice(2);
    const eq = body.indexOf('=');
    if (eq >= 0) {
      const key = body.slice(0, eq);
      const raw = body.slice(eq + 1);
      put(key, booleans.has(key) ? parseBool(raw, key) : raw);
      continue;
    }
    if (body.startsWith('no-') && booleans.has(body.slice(3))) {
      out.opts[body.slice(3)] = false;
      continue;
    }
    if (booleans.has(body)) {
      out.opts[body] = true;
      continue;
    }
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) throw new Error(`--${body} needs a value`);
    put(body, next);
    i++;
  }
  return out;
}

export function parseBool(raw, key = 'value') {
  const v = String(raw).toLowerCase();
  if (['1', 'true', 'yes', 'on'].includes(v)) return true;
  if (['0', 'false', 'no', 'off'].includes(v)) return false;
  throw new Error(`--${key} expects true/false, got "${raw}"`);
}
