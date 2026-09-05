// Headless Strudel for lib/gen: evaluate pattern code, query cycles into
// events. JSON lines on stdin -> JSON lines on stdout.
//
//   {"id":1,"op":"eval","code":"s(\"bd*4\")"}          -> {"id":1,"ok":true}
//   {"id":2,"op":"query","from":0,"to":4,"ctx":{...}}    -> {"id":2,"haps":[{"b":0,"e":0.25,"v":{"s":"bd"}},...]}
//   {"id":3,"op":"ping"}                                -> {"id":3,"ok":true,"version":"..."}
//
// One cycle == one bar (the Python side runs it at cpm = bpm/4). `ctx`
// values (energy, section, bar, key, bpm) are exposed as globals so a
// pattern can react to the composer's form: .gain(energy) etc.
import readline from 'node:readline';

// stdout is the protocol channel: route every library log line to stderr.
const out = (o) => process.stdout.write(JSON.stringify(o) + '\n');
console.log = (...a) => process.stderr.write(a.join(' ') + '\n');
console.warn = console.log;
const core = await import('@strudel/core');
const mini = await import('@strudel/mini');
const tonal = await import('@strudel/tonal');
const { transpiler } = await import('@strudel/transpiler');

await core.evalScope(core, mini, tonal);
// Composer context. Numeric values are exposed as SIGNALS (read at query
// time, so `.gain(energy)` follows the composer's form phrase by phrase);
// the raw values live on `ctx` for code that needs them as plain JS.
const ctx = { energy: 0.5, section: 'groove', bar: 0, key: '8A', bpm: 120, phrase: 0, chords: [] };
globalThis.ctx = ctx;
for (const k of ['energy', 'bar', 'bpm', 'phrase']) globalThis[k] = core.signal(() => Number(ctx[k]));
globalThis.section = () => ctx.section;
globalThis.key = () => ctx.key;
let pattern = null;
const num = (x) => (x && typeof x.valueOf === 'function') ? Number(x.valueOf()) : Number(x);

// Requests are handled strictly in order (eval is async; a query must
// never overtake the eval that precedes it).
let chain = Promise.resolve();
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => { chain = chain.then(() => handle(line)).catch(() => {}); });
async function handle(line) {
  let req;
  try { req = JSON.parse(line); } catch (e) { out({ error: 'bad json' }); return; }
  const id = req.id;
  try {
    if (req.op === 'ping') { out({ id, ok: true, version: '1.2.6' }); return; }
    if (req.op === 'eval') {
      const r = await core.evaluate(String(req.code || ''), transpiler);
      if (!r || !r.pattern || typeof r.pattern.queryArc !== 'function') throw new Error('code did not produce a pattern');
      r.pattern.queryArc(0, 1);           // surface runtime errors now, not mid-show
      pattern = r.pattern;
      out({ id, ok: true });
      return;
    }
    if (req.op === 'query') {
      if (!pattern) { out({ id, haps: [] }); return; }
      Object.assign(ctx, req.ctx || {});
      const haps = pattern.queryArc(Number(req.from), Number(req.to))
        .filter((h) => h.hasOnset && h.hasOnset())
        .map((h) => ({ b: num(h.whole.begin), e: num(h.whole.end), v: h.value }));
      out({ id, haps });
      return;
    }
    out({ id, error: 'unknown op ' + req.op });
  } catch (e) {
    out({ id, error: String(e && e.message ? e.message : e) });
  }
}
