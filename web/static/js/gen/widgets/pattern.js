/** Strudel code editor (advanced) and the phrase timeline. */
import { register } from '../registry.js';
import { el } from '../store.js';

register('code', {
    create(spec, ctx) {
        const wrap = el('div');
        const ta = el('textarea', 'code'); ta.spellcheck = false; ta.placeholder = spec.placeholder || '';
        ta.addEventListener('input', () => { ta.dataset.dirty = '1'; });
        ta.addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); ctx.emit(spec.action, ta.value); } });
        const btns = el('div', 'buttons'); btns.style.marginTop = '8px';
        const ev = el('button', 'btn go', '▶ EVAL (next phrase)'); ev.onclick = () => ctx.emit(spec.action, ta.value);
        const cl = el('button', 'btn', 'CLEAR → autonomous'); cl.onclick = () => { ctx.emit(spec.clear_action); ta.dataset.dirty = ''; };
        btns.appendChild(ev); btns.appendChild(cl);
        wrap.appendChild(ta); wrap.appendChild(btns); wrap.appendChild(el('div', 'status-line'));
        if (spec.help) wrap.appendChild(el('div', 'help', spec.help));
        wrap._ta = ta; return wrap;
    },
    update(wrap, s, spec) {
        const ta = wrap._ta, st = wrap.querySelector('.status-line');
        if (document.activeElement !== ta && s[spec.key] && ta.value !== s[spec.key] && !ta.dataset.dirty) ta.value = s[spec.key];
        const slots = s[spec.slots_key] || [];
        if (s[spec.available_key] === false) { st.className = 'status-line err'; st.textContent = 'Strudel unavailable (pip install mini-racer)'; }
        else if (s[spec.status_key]) { st.className = 'status-line err'; st.textContent = 'pattern error: ' + s[spec.status_key]; }
        else if (s.error && String(s.error).startsWith('pattern')) { st.className = 'status-line err'; st.textContent = s.error; }
        else if (s[spec.key]) { st.className = 'status-line ok'; st.textContent = `whole-rack pattern live (${s[spec.engine_key] || ''}) - form/energy still run underneath`; }
        else if (slots.length) { st.className = 'status-line ok'; st.textContent = `slot patterns live: ${slots.join(', ')} (${s[spec.engine_key] || ''})`; }
        else { st.className = 'status-line'; st.textContent = s.active ? 'autonomous (rule composer)' : 'idle - a pattern set now is applied at start'; }
    },
});

register('phrase_log', {
    create() { return el('div', 'log'); },
    update(box, s, spec) {
        const rows = (s[spec.key] || []).slice(-(spec.limit || 14)).reverse();
        box.innerHTML = rows.map((r) => {
            if (r.event !== 'phrase') return `<div class="l"><span class="b"></span><span class="s">${r.event}</span><span class="c">${r.style || r.key || r.seed || r.name || r.error || ''}</span></div>`;
            return `<div class="l"><span class="b">${r.bar}</span><span class="s">${r.section}</span><span class="e">${Number(r.energy).toFixed(2)}</span>` +
                   `<span class="c">${(r.chords || []).join(' ')} · ${r.key}</span><span class="o">${r.lead || ''}</span></div>`;
        }).join('');
    },
});
