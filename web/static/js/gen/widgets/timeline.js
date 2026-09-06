/** The song strip: what played, what is composed ahead, what the form
 *  knows beyond that (state.timeline from GenSystem.timeline()). */
import { register } from '../registry.js';
import { el } from '../store.js';

const COLOURS = { intro: '#4a5a7a', groove: '#3f7a5a', build: '#a8772a', drop: '#b03a3a', break: '#5a4a8a',
                  outro: '#4a4a4a', flow: '#3f7a7a', swell: '#a85a2a', calm: '#4a6a8a' };

register('timeline_strip', {
    create(spec) {
        const wrap = el('div', 'timeline-strip');
        const cv = document.createElement('canvas');
        cv.height = spec.height || 120; cv.style.width = '100%'; cv.style.display = 'block';
        wrap.appendChild(cv);
        const info = el('div', 'help'); wrap.appendChild(info);
        wrap._cv = cv; wrap._info = info; wrap._win = spec.window_s || 300;
        return wrap;
    },
    update(wrap, s, spec) {
        const tl = s[spec.key || 'timeline'];
        const cv = wrap._cv, ctx = cv.getContext('2d');
        const W = cv.width = Math.max(200, wrap.clientWidth || 600), H = cv.height;
        ctx.fillStyle = '#16181d'; ctx.fillRect(0, 0, W, H);
        if (!tl) { wrap._info.textContent = 'idle'; return; }
        const now = tl.now_s || 0, win = wrap._win, past = 0.35;
        const left = now - win * past;
        const x = (t) => (t - left) / win * W;
        ctx.font = '10px sans-serif';
        // minute grid
        ctx.strokeStyle = '#2a2e36'; ctx.fillStyle = '#5a606c';
        for (let t = Math.floor(left / 60) * 60; t < left + win; t += 60) {
            const xx = x(t); ctx.beginPath(); ctx.moveTo(xx, 0); ctx.lineTo(xx, H); ctx.stroke();
            ctx.fillText(`${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`, xx + 3, H - 3);
        }
        const hz = tl.horizon || {};
        // known future: rest of the section (hatched) + likely next
        const compTo = hz.composed_to_s || now, secEnd = hz.section_end_s || compTo;
        if (secEnd > compTo) {
            const x0 = x(compTo), x1 = x(secEnd);
            ctx.fillStyle = (COLOURS[hz.section] || '#555') + '66';
            ctx.fillRect(x0, 6, Math.max(1, x1 - x0), 26);
            ctx.strokeStyle = '#0e1013';
            for (let xx = x0; xx < x1; xx += 6) { ctx.beginPath(); ctx.moveTo(xx, 6); ctx.lineTo(xx + 6, 32); ctx.stroke(); }
            ctx.fillStyle = '#c8ccd4'; ctx.fillText(`${hz.section} (${hz.bars_left} bars left)`, x0 + 4, 18);
            let xs = x1;
            for (const [name, w] of (hz.next || []).slice(0, 3)) {
                const span = Math.max(20, 90 * w);
                ctx.fillStyle = (COLOURS[name] || '#555') + '44'; ctx.fillRect(xs, 10, span, 18);
                ctx.fillStyle = '#9aa0ac'; ctx.fillText(`${name} ${Math.round(w * 100)}%`, xs + 3, 28);
                xs += span + 2;
            }
        }
        // phrases
        let prevKey = null, last = null;
        ctx.lineWidth = 2;
        for (const p of tl.phrases || []) {
            const x0 = x(p.start_s), x1 = x(p.end_s);
            if (x1 >= 0 && x0 <= W) {
                ctx.fillStyle = (COLOURS[p.section] || '#555') + (p.played ? '99' : 'ff');
                ctx.fillRect(x0, 6, Math.max(1, x1 - x0 - 1), 26);
                if (x1 - x0 > 46) { ctx.fillStyle = '#f0f2f5'; ctx.fillText(p.section, x0 + 4, 18); ctx.fillStyle = '#c8ccd4'; ctx.fillText(`bar ${p.bar0}`, x0 + 4, 30); }
                if (prevKey && p.key !== prevKey) { ctx.strokeStyle = '#ffd166'; ctx.beginPath(); ctx.moveTo(x0, 6); ctx.lineTo(x0, H - 14); ctx.stroke(); ctx.fillStyle = '#ffd166'; ctx.fillText(p.key, x0 + 3, H - 16); }
                for (const d of p.drops || []) { const xd = x(d); ctx.strokeStyle = '#ff5c5c'; ctx.beginPath(); ctx.moveTo(xd, 6); ctx.lineTo(xd, H - 14); ctx.stroke(); ctx.fillStyle = '#ff5c5c'; ctx.fillText('DROP', xd + 3, 44); }
                if (p.lead === 'theme' || p.lead === 'theme_make') { ctx.fillStyle = '#ffd166'; ctx.fillText(p.lead === 'theme' ? 'T' : 't', x0 + 3, 56); }
                if (x1 - x0 > 96) { ctx.fillStyle = '#9aa0ac'; ctx.fillText((p.chords || []).join(' '), x0 + 4, 70); }
            }
            prevKey = p.key;
            // energy
            const y = H - 20 - p.energy * 40;
            ctx.strokeStyle = '#7fd1a8';
            if (last) { ctx.beginPath(); ctx.moveTo(last[0], last[1]); ctx.lineTo(x0, y); ctx.stroke(); }
            ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
            last = [x1, y];
        }
        // the arc beyond the composed edge
        ctx.setLineDash([3, 3]); ctx.strokeStyle = '#7fd1a8'; let prev = null;
        for (const [t, e] of hz.arc || []) { const xx = x(t), yy = H - 20 - e * 40; if (prev && xx <= W) { ctx.beginPath(); ctx.moveTo(prev[0], prev[1]); ctx.lineTo(xx, yy); ctx.stroke(); } prev = [xx, yy]; }
        ctx.setLineDash([]);
        if (hz.drop_s) { const xd = x(hz.drop_s); ctx.setLineDash([4, 3]); ctx.strokeStyle = '#ff5c5c'; ctx.beginPath(); ctx.moveTo(xd, 6); ctx.lineTo(xd, H - 14); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = '#ff5c5c'; ctx.fillText(`drop in ${Math.max(0, hz.drop_s - now).toFixed(0)}s`, xd + 3, 44); }
        // cursor
        const xc = x(now); ctx.strokeStyle = '#ffffff'; ctx.beginPath(); ctx.moveTo(xc, 0); ctx.lineTo(xc, H); ctx.stroke();
        ctx.fillStyle = '#c8ccd4'; ctx.fillText(`now ${Math.floor(now / 60)}:${String(Math.floor(now % 60)).padStart(2, '0')}`, xc + 4, 12);
        wrap._info.textContent = `${hz.section || '-'} · ${hz.bars_left ?? '-'} bars left · composed to +${Math.max(0, (hz.composed_to_s || now) - now).toFixed(0)}s · next ` +
            (hz.next || []).slice(0, 2).map(([n, w]) => `${n} ${Math.round(w * 100)}%`).join(', ') + (hz.ending ? ' · ENDING' : '') + (hz.hold ? ' · HOLD' : '');
    },
});
