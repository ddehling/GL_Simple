/** The director's widgets: ask box + status, director log, scenes. */
import { register } from '../registry.js';
import { el } from '../store.js';

register('ask', {
    create(spec, ctx) {
        const wrap = el('div');
        const row = el('div', 'ask');
        const inp = el('input'); inp.type = 'text'; inp.placeholder = spec.placeholder || '';
        const go = el('button', 'btn alt', 'ASK');
        const send = () => { const t = inp.value.trim(); if (t) { ctx.emit(spec.action, t); inp.value = ''; } };
        go.onclick = send; inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
        row.appendChild(inp); row.appendChild(go);
        wrap.appendChild(row); wrap.appendChild(el('div', 'status-line'));
        return wrap;
    },
    update(wrap, s, spec) {
        const d = s[spec.status_key] || {}; const st = wrap.querySelector('.status-line');
        if (!d.available) { st.className = 'status-line err'; st.textContent = 'director offline (install Claude Code `claude`, or pip install anthropic + ANTHROPIC_API_KEY) - gestures still work'; }
        else if (d.busy) { st.className = 'status-line busy'; st.textContent = 'director thinking about: ' + ((d.last && d.last.text) || ''); }
        else if (d.last && d.last.error) { st.className = 'status-line err'; st.textContent = 'director error: ' + d.last.error; }
        else if (d.last && d.last.say !== undefined) { st.className = 'status-line ok'; st.textContent = 'director: ' + (d.last.say || '(done)') + ((d.last.warn && d.last.warn.length) ? '  · ' + d.last.warn.join('; ') : ''); }
        else { st.className = 'status-line'; st.textContent = 'director ready (' + (d.mode || '') + ')'; }
    },
});

register('director_log', {
    create() { return el('div', 'log short'); },
    update(box, s, spec) {
        const d = s[spec.key] || {};
        box.innerHTML = (d.log || []).slice(-(spec.limit || 8)).reverse().map((r) =>
            `<div class="l"><span class="s">${r.kind}</span><span class="c">${r.text} → ${(r.done || []).join(', ') || r.say || ''}</span></div>`).join('');
    },
});

register('scenes', {
    create(spec, ctx) {
        const wrap = el('div');
        const row = el('div', 'row'); row.appendChild(el('label', '', 'scene'));
        const sel = el('select'); row.appendChild(sel); wrap.appendChild(row);
        const btns = el('div', 'buttons');
        const save = el('button', 'btn go', '＋ save as…');
        save.onclick = () => { const n = window.prompt('Scene name'); if (n && n.trim()) ctx.emit(spec.actions.save, n.trim()); };
        const load = el('button', 'btn alt', '▶ recall'); load.onclick = () => { if (sel.value) ctx.emit(spec.actions.load, sel.value); };
        const del = el('button', 'btn', '✕ delete'); del.onclick = () => { if (sel.value && window.confirm(`Delete scene "${sel.value}"?`)) ctx.emit(spec.actions.delete, sel.value); };
        btns.appendChild(save); btns.appendChild(load); btns.appendChild(del); wrap.appendChild(btns);
        wrap.appendChild(el('div', 'help', 'A scene is the steering surface: style, tempo, key, energy bias, density, swing, brightness, level, muted layers and patterns. Recalled at the next phrase.'));
        wrap._sel = sel; wrap._key = ''; return wrap;
    },
    update(wrap, s, spec) {
        const scenes = s[spec.key] || [];
        const key = scenes.map((x) => x.name).join('|');
        if (key === wrap._key) return;
        wrap._key = key; const cur = wrap._sel.value; wrap._sel.innerHTML = '';
        if (!scenes.length) { const o = el('option', '', '(no scenes saved yet)'); o.value = ''; wrap._sel.appendChild(o); }
        for (const sc of scenes) { const o = el('option', '', `${sc.name}  · ${sc.style || ''} ${sc.bpm ? sc.bpm + ' bpm' : ''} ${sc.key || ''}`); o.value = sc.name; wrap._sel.appendChild(o); }
        if ([...wrap._sel.options].some((o) => o.value === cur)) wrap._sel.value = cur;
    },
});
