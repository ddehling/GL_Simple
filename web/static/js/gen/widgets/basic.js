/** Banner, button rows, headline, key line, beats, chords, countdown, meters, key/values. */
import { register } from '../registry.js';
import { el, fmt, duration } from '../store.js';

register('banner', {
    create() { return el('div', 'gen-banner', 'Connecting to the show…'); },
    update(e, s) {
        if (!s) { e.className = 'gen-banner err'; e.textContent = 'Generative subsystem not reported by the show. Is gen.enabled set in config.yaml?'; return; }
        if (s.error) { e.className = 'gen-banner err'; e.textContent = 'Error: ' + s.error; return; }
        if (s.active) {
            const mode = s.state === 'ending' ? 'ending after the outro' : (s.state === 'hold' ? 'holding this section' : 'autonomous');
            e.className = 'gen-banner on'; e.textContent = `Playing · ${s.style} · ${s.bpm} bpm · ${s.key} (${s.camelot}) · ${mode}`;
        } else { e.className = 'gen-banner'; e.textContent = 'Idle. Steering below arms the next start.'; }
    },
});

register('buttons', {
    create(spec, ctx) {
        const row = el('div', 'buttons');
        for (const it of spec.items || []) {
            const b = el('button', 'btn' + (it.style ? ' ' + it.style : ''), it.label);
            b.dataset.show = it.show_when || 'always';
            b.onclick = () => {
                if (it.confirm && !window.confirm(it.confirm)) return;
                let v = it.value;
                if (it.toggle_key) v = !(b.classList.contains('on'));
                ctx.emit(it.action, v === undefined ? null : v);
            };
            b._item = it;
            row.appendChild(b);
        }
        if (spec.trailing_key) row.appendChild(el('span', 'trailing'));
        return row;
    },
    update(row, s, spec) {
        const live = !!(s && s.active);
        for (const b of row.querySelectorAll('.btn')) {
            const rule = b.dataset.show;
            b.classList.toggle('hidden', !(rule === 'always' || (rule === 'live' && live) || (rule === 'idle' && !live)));
            const it = b._item;
            if (it.toggle_key && s) b.classList.toggle('on', s[it.toggle_key] === it.toggle_value);
        }
        if (spec.trailing_key && s) row.querySelector('.trailing').textContent = fmt(spec.trailing_format || '{0}', s[spec.trailing_key] || {});
    },
});

register('headline', {
    create() { const d = el('div', 'now-title'); d.appendChild(el('span', 'sec', '--')); d.appendChild(el('span', 'sub')); d.appendChild(el('span', 'arrow')); return d; },
    update(d, s, spec) {
        d.querySelector('.sec').textContent = s[spec.key] || '--';
        d.querySelector('.sub').textContent = (spec.sub_keys || []).map((k) => ` · ${k} ${s[k]}`).join('');
        const a = spec.arrow_key && s[spec.arrow_key];
        d.querySelector('.arrow').textContent = a ? `  → ${a}` : '';
    },
});

register('keyline', {
    create() { return el('div', 'now-sub', '--'); },
    update(d, s, spec) {
        const k = spec.keys || [];
        d.textContent = `${s[k[0]]} (${s[k[1]]}) · ${s[k[2]]} bpm · chord ${s[k[3]] || '–'}` + (s[k[4]] ? ` · motif ${s[k[4]]}` : '');
    },
});

register('beats', {
    create() { const r = el('div', 'beatrow'); for (let i = 1; i <= 4; i++) r.appendChild(el('div', 'beat' + (i === 1 ? ' down' : ''))); return r; },
    update(r, s, spec) { const b = s[spec.key]; [...r.children].forEach((c, i) => c.classList.toggle('on', b === i + 1)); },
});

register('chords', {
    create() { return el('div', 'chords'); },
    update(d, s, spec) {
        const chords = s[spec.key] || [];
        if (d.children.length !== chords.length) { d.innerHTML = ''; chords.forEach(() => d.appendChild(el('div', 'chord'))); }
        const cur = Math.floor((s[spec.phase_key] || 0) * chords.length);
        chords.forEach((c, i) => { d.children[i].textContent = c; d.children[i].classList.toggle('on', i === cur); });
    },
});

register('countdown', {
    create() { return el('div', 'countdown'); },
    update(d, s, spec) {
        const v = s[spec.key];
        if (v === null || v === undefined) { d.textContent = ''; d.classList.remove('hot'); return; }
        d.textContent = `${spec.label || ''} ${Number(v).toFixed(1)} s`; d.classList.toggle('hot', v < (spec.hot_below || 0));
    },
});

register('meter', {
    create(spec) {
        const w = el('div', 'meter-wrap');
        const lab = el('div', 'labels'); lab.appendChild(el('span', '', spec.label || '')); lab.appendChild(el('span', 'right'));
        const m = el('div', 'meter'); m.appendChild(el('div', 'fill ' + (spec.palette || 'plain')));
        w.appendChild(lab); w.appendChild(m); return w;
    },
    update(w, s, spec) {
        let frac;
        if (spec.done_key && spec.total_key) {
            const tot = s[spec.total_key] || 0, left = s[spec.done_key] || 0;
            frac = tot ? (spec.inverse ? (tot - left) / tot : left / tot) : 0;
        } else frac = Number(s[spec.key] || 0);
        w.querySelector('.fill').style.width = (100 * Math.max(0, Math.min(1, frac))).toFixed(1) + '%';
        if (spec.right_keys) w.querySelector('.right').textContent = fmt(spec.right_format || '{0}', ...spec.right_keys.map((k) => s[k]));
    },
});

register('kv', {
    create(spec) {
        const g = el('div', 'kv-grid');
        for (const it of spec.items || []) { const r = el('div', 'kv'); r.appendChild(el('span', 'k', it.label)); r.appendChild(el('span', 'v')); r._item = it; g.appendChild(r); }
        return g;
    },
    update(g, s) {
        for (const r of g.children) {
            const it = r._item; let v = s[it.key];
            if (it.format === 'duration') v = duration(v);
            else if (it.format === 'list') v = Array.isArray(v) ? (v.join(', ') || 'none') : (v || 'none');
            else if (it.format === 'json') v = (v && Object.keys(v).length) ? JSON.stringify(v) : 'none';
            else if (it.format) v = fmt(it.format, v);
            r.querySelector('.v').textContent = (v === undefined || v === null) ? '–' : v;
        }
    },
});
