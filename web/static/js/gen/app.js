/** /gen surface renderer. Fetches the declarative spec from
 *  /api/gen/surface, builds cards from the widget registry, then feeds
 *  every widget the live gen_info state at each state_update. */
import { get as widget } from './registry.js';
import { store, el } from './store.js';
import { socket, emit } from './actions.js';

const $ = (id) => document.getElementById(id);
const ctx = { emit, store };
let built = [];                  // [{card, cardEl, widgets: [{spec, el, def}]}]
let specVersion = null;

function showWhen(rule, live) {
    return !rule || rule === 'always' || (rule === 'live' && live) || (rule === 'idle' && !live);
}

function buildCard(card) {
    const cardEl = el('div', 'card');
    cardEl.dataset.id = card.id;
    if (card.kind === 'banner' || card.kind === 'transport' || card.kind === 'strip') cardEl.classList.add('span', card.kind);
    else {
        cardEl.classList.add('section-panel');
        if (card.col) cardEl.classList.add('col' + card.col);
        if (card.title) {
            const h = el('div', 'section-header');
            const h3 = el('h3', '', card.title);
            if (card.hint) h3.appendChild(el('small', '', card.hint));
            h.appendChild(h3); cardEl.appendChild(h);
        }
        if (card.foldable) {
            cardEl.classList.add('foldable');
            const k = 'gen-fold-' + card.id;
            let folded = !!card.folded;
            try { const v = localStorage.getItem(k); if (v === '1') folded = true; else if (v === '0') folded = false; } catch (e) {}
            cardEl.classList.toggle('folded', folded);
            cardEl.querySelector('.section-header').addEventListener('click', () => {
                cardEl.classList.toggle('folded');
                try { localStorage.setItem(k, cardEl.classList.contains('folded') ? '1' : '0'); } catch (e) {}
            });
        }
        if (card.advanced) cardEl.classList.add('advanced');
    }
    if (card.sticky) cardEl.classList.add('sticky');
    const widgets = [];
    for (const w of card.widgets || []) {
        const def = widget(w.type);
        if (!def) { cardEl.appendChild(el('div', 'status-line err', `unknown widget ${w.type}`)); continue; }
        const wEl = def.create(w, ctx);
        cardEl.appendChild(wEl);
        widgets.push({ spec: w, el: wEl, def });
    }
    return { card, cardEl, widgets };
}

const wide = window.matchMedia('(min-width: 900px)');

/** Narrow: cards flow in spec order. Wide: two independent column stacks
 *  (no shared grid rows, so a tall card never leaves a gap beside it). */
function layout() {
    const root = $('surface');
    const spans = built.filter((b) => b.cardEl.classList.contains('span'));
    const cols = built.filter((b) => !b.cardEl.classList.contains('span'));
    root.innerHTML = '';
    spans.forEach((b) => root.appendChild(b.cardEl));
    if (wide.matches) {
        const c1 = el('div', 'colstack'), c2 = el('div', 'colstack');
        cols.forEach((b) => ((b.card.col === 2) ? c2 : c1).appendChild(b.cardEl));
        const wrap = el('div', 'cols'); wrap.appendChild(c1); wrap.appendChild(c2); root.appendChild(wrap);
    } else {
        cols.forEach((b) => root.appendChild(b.cardEl));
    }
}
wide.addEventListener('change', layout);

function build(spec) {
    built = (spec.cards || []).map(buildCard);
    layout();
    specVersion = spec.version;
    document.title = spec.title || document.title;
}

function update() {
    const s = store.state;
    const live = store.live();
    for (const b of built) {
        const vis = !!s && showWhen(b.card.show_when, live);
        b.cardEl.classList.toggle('hidden', !vis && b.card.kind !== 'banner');
        if (!vis && b.card.kind !== 'banner') continue;
        for (const w of b.widgets) {
            try { w.def.update(w.el, s, w.spec, ctx); }
            catch (e) { console.error('widget', w.spec.type, e); }
        }
    }
}

async function loadSpec() {
    const r = await fetch('/api/gen/surface');
    const spec = await r.json();
    build(spec);
    update();
}

socket.on('state_update', (s) => {
    $('status-text').textContent = 'Connected';
    store.state = s.gen || null;
    if (document.hidden) return;
    update();
});
socket.on('disconnect', () => { $('status-text').textContent = 'Disconnected'; });
document.addEventListener('visibilitychange', () => { if (!document.hidden) update(); });

loadSpec().catch((e) => { $('surface').innerHTML = `<div class="gen-banner err">surface spec failed to load: ${e}</div>`; });
export { update, build };
