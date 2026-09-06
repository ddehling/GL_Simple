/** Operator inputs: chips, choice buttons, sliders, selects, text, toggles. */
import { register } from '../registry.js';
import { el, fmt, CAMELOT } from '../store.js';
import { dragging, throttled } from '../actions.js';

register('chips', {
    create(spec, ctx) { const c = el('div', 'chips'); c._key = ''; return c; },
    update(c, s, spec, ctx) {
        const items = s[spec.items_key] || [];
        const key = items.map((i) => i[spec.id_field || 'id']).join('|');
        if (key !== c._key) {
            c._key = key; c.innerHTML = '';
            for (const it of items) {
                const chip = el('span', 'chip', it[spec.label_field || 'label']);
                chip.onclick = () => { ctx.emit(spec.action, it[spec.id_field || 'id']); if (spec.flash) { chip.classList.add('flash'); setTimeout(() => chip.classList.remove('flash'), 900); } };
                c.appendChild(chip);
            }
        }
    },
});

register('choice', {
    create() { const r = el('div', 'style-row'); r._key = ''; return r; },
    update(r, s, spec, ctx) {
        const opts = s[spec.options_key] || [];
        const key = opts.map((o) => o[spec.id_field || 'id']).join('|');
        if (key !== r._key) {
            r._key = key; r.innerHTML = '';
            for (const o of opts) {
                const b = el('button', 'style-btn'); b.dataset.id = o[spec.id_field || 'id'];
                b.textContent = o[spec.id_field || 'id'];
                if (spec.sub_format) b.appendChild(el('small', '', fmt(spec.sub_format, o)));
                b.onclick = () => ctx.emit(spec.action, b.dataset.id);
                r.appendChild(b);
            }
        }
        const cur = s[spec.key];
        r.querySelectorAll('.style-btn').forEach((b) => b.classList.toggle('on', b.dataset.id === cur));
    },
});

register('slider', {
    create(spec, ctx) {
        const row = el('div', 'row'); row.appendChild(el('label', '', spec.label || spec.key));
        const inp = el('input'); inp.type = 'range'; inp.min = spec.min; inp.max = spec.max; inp.step = spec.step || 0.01;
        const val = el('span', 'val', '--');
        const id = 'slider-' + spec.key;
        const show = (x) => { val.textContent = (spec.signed && x >= 0 ? '+' : '') + x.toFixed(spec.decimals ?? 2); };
        const send = throttled(120);
        inp.addEventListener('pointerdown', () => dragging.add(id));
        const done = () => { dragging.delete(id); ctx.emit(spec.action, parseFloat(inp.value)); };
        inp.addEventListener('pointerup', done); inp.addEventListener('touchend', done); inp.addEventListener('change', done);
        inp.addEventListener('input', () => { const x = parseFloat(inp.value); show(x); send(() => ctx.emit(spec.action, x)); });
        row.appendChild(inp); row.appendChild(val); row._inp = inp; row._show = show; row._id = id;
        return row;
    },
    update(row, s, spec) {
        const v = s[spec.key];
        if (v === undefined || v === null || dragging.has(row._id)) return;
        if (parseFloat(row._inp.value) !== v) row._inp.value = v;
        row._show(Number(v));
    },
});

register('select', {
    create(spec, ctx) {
        const row = el('div', 'row'); row.appendChild(el('label', '', spec.label || spec.key));
        const sel = el('select');
        const opts = spec.options === 'camelot' ? CAMELOT : (spec.options || []);
        for (const o of opts) { const op = el('option', '', o.label); op.value = String(o.id); sel.appendChild(op); }
        sel.addEventListener('change', () => ctx.emit(spec.action, isNaN(Number(sel.value)) ? sel.value : Number(sel.value)));
        row.appendChild(sel);
        if (spec.trailing_key) row.appendChild(el('span', 'val'));
        row._sel = sel; return row;
    },
    update(row, s, spec) {
        let v = s[spec.key];
        if ((v === undefined || v === null) && spec.idle_key) v = s[spec.idle_key];
        if (v !== undefined && v !== null) { const sv = String(typeof v === 'number' ? Math.round(v) : v); if (row._sel.value !== sv && [...row._sel.options].some((o) => o.value === sv)) row._sel.value = sv; }
        if (spec.trailing_key) row.querySelector('.val').textContent = s[spec.trailing_key] || '';
    },
});

register('text', {
    create(spec, ctx) {
        const row = el('div', 'row'); row.appendChild(el('label', '', spec.label || spec.key));
        const inp = el('input'); inp.type = 'text'; inp.placeholder = spec.placeholder || '';
        inp.addEventListener('change', () => ctx.emit(spec.action, inp.value.trim()));
        row.appendChild(inp); row._inp = inp; return row;
    },
    update(row, s, spec) {
        if (document.activeElement === row._inp) return;
        const v = s[spec.key]; row._inp.value = Array.isArray(v) ? v.join(',') : (v || '');
    },
});

register('toggles', {
    create() { const c = el('div', 'chips'); c._key = ''; return c; },
    update(c, s, spec, ctx) {
        const items = s[spec.items_key] || [];
        const key = items.join('|');
        if (key !== c._key) {
            c._key = key; c.innerHTML = '';
            for (const it of items) {
                const chip = el('span', 'chip'); chip.dataset.item = it;
                chip.onclick = () => {
                    const next = !chip.classList.contains('muted');
                    const vf = spec.value_format;
                    let value = it;
                    if (vf && typeof vf === 'object') { value = {}; for (const [k, v] of Object.entries(vf)) value[k] = v === '$item' ? it : (v === '$next' ? next : v); }
                    ctx.emit(spec.action, value);
                };
                c.appendChild(chip);
            }
        }
        const on = new Set(s[spec.on_key] || []), off = new Set(s[spec.off_key] || []), badge = new Set(s[spec.badge_key] || []);
        for (const chip of c.children) {
            const it = chip.dataset.item;
            chip.classList.toggle('live', on.has(it) && !off.has(it)); chip.classList.toggle('muted', off.has(it));
            chip.innerHTML = it + (badge.has(it) ? `<span class="badge">${spec.badge || ''}</span>` : '');
        }
    },
});
