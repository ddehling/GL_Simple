/** State + formatting helpers shared by every widget. */
export const store = {
    state: null,                 // latest gen_info from the show
    live() { return !!(this.state && this.state.active); },
    get(key, fallback) {
        const v = this.state ? this.state[key] : undefined;
        return (v === undefined || v === null) ? fallback : v;
    },
};

/** Tiny Python-ish format: "{0:.2f}", "{1:+.2f}", "{name}", "{bpm[0]}", "{0}". */
export function fmt(pattern, ...args) {
    const named = (args.length === 1 && args[0] && typeof args[0] === 'object' && !Array.isArray(args[0])) ? args[0] : null;
    return pattern.replace(/\{([^{}:]+)(?::([^{}]+))?\}/g, (m, ref, spec) => {
        let v;
        const idx = /^\d+$/.test(ref) ? parseInt(ref, 10) : null;
        if (idx !== null) v = args[idx];
        else {
            const mm = ref.match(/^([a-zA-Z_]+)\[(\d+)\]$/);
            if (mm) { const o = named ? named[mm[1]] : undefined; v = Array.isArray(o) ? o[parseInt(mm[2], 10)] : undefined; }
            else v = named ? named[ref] : undefined;
        }
        if (v === undefined || v === null) return '–';
        if (spec) {
            const s = spec.match(/^([+])?\.(\d)f$/);
            if (s && typeof v === 'number') { const t = v.toFixed(parseInt(s[2], 10)); return (s[1] && v >= 0) ? '+' + t : t; }
        }
        return Array.isArray(v) ? v.join(' / ') : String(v);
    });
}

export function duration(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}h ${m}m ${s}s` : (m ? `${m}m ${s}s` : `${s}s`);
}

export function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
}

export const CAMELOT = (() => {
    const names = {"1A":"Ab min","2A":"Eb min","3A":"Bb min","4A":"F min","5A":"C min","6A":"G min","7A":"D min","8A":"A min",
        "9A":"E min","10A":"B min","11A":"F# min","12A":"C# min","1B":"B maj","2B":"F# maj","3B":"Db maj","4B":"Ab maj",
        "5B":"Eb maj","6B":"Bb maj","7B":"F maj","8B":"C maj","9B":"G maj","10B":"D maj","11B":"A maj","12B":"E maj"};
    const out = [];
    for (let n = 1; n <= 12; n++) for (const ab of ['A', 'B']) out.push({ id: `${n}${ab}`, label: `${n}${ab}  ${names[n + ab]}` });
    return out;
})();
