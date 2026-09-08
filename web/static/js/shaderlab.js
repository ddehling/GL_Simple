/**
 * Shader Lab client — wish patterns onto the lights.
 *
 * The person NEVER picks a slot. A wish ('shaderlab_generate' with
 * slot:null) is routed server-side: "make the waves slower" refines the
 * waves layer, new subjects land on a free layer (replacing the oldest
 * when the stage is full), "remove the fire" / "clear everything" manage
 * the stage. Tapping a palette tile tosses a built-in pattern on
 * instantly (no LLM). The stage strip shows only what's actually
 * playing; each tile has direct-manipulation controls — intensity,
 * knobs, palette dots, dice (shuffle), audio cycle, ✕ — and tapping a
 * tile focuses it so chat and the editor target that layer (optional,
 * togglable).
 */
(function () {
    const socket = createSocket();

    const log = document.getElementById('lab-log');
    const statusEl = document.getElementById('lab-status');
    const promptEl = document.getElementById('lab-prompt');
    const genBtn = document.getElementById('lab-generate');
    const codeEl = document.getElementById('lab-code');
    const errEl = document.getElementById('lab-error');
    const frameEl = document.getElementById('lab-frame');
    const banner = document.getElementById('lab-banner');
    const savedList = document.getElementById('lab-saved-list');
    const builtinList = document.getElementById('lab-builtin-list');
    const saveNameEl = document.getElementById('lab-save-name');
    const engineList = document.getElementById('lab-engine-list');
    const engineMeta = document.getElementById('lab-engine-meta');
    const modelEl = document.getElementById('lab-model');
    const slotsEl = document.getElementById('lab-slots');
    const focusChip = document.getElementById('lab-focus');
    const focusName = document.getElementById('lab-focus-name');

    let numSlots = 16;
    let focusSlot = null;       // null = wish freely; server routes
    let slots = {};             // slot -> now metadata (or null)
    let params = {};            // str(slot) -> {intensity, knobs}
    let lastPrompt = '';
    let lastFrameUrl = null;
    let savedPatterns = [];

    const PALETTES = ['custom', 'duo', 'rainbow', 'sunset', 'neon', 'ice',
                      'pastel', 'candy'];
    const PAL_DOT_ART = {
        custom: 'conic-gradient(#4a6cf0, #4a6cf0)',
        duo: 'linear-gradient(90deg,#4a6cf0 50%,#c04af0 50%)',
        rainbow: 'conic-gradient(red,yellow,lime,cyan,blue,magenta,red)',
        sunset: 'linear-gradient(135deg,#ff7a3c,#e0447a,#7a3cb8)',
        neon: 'linear-gradient(135deg,#ff29c8,#29ffe8)',
        ice: 'linear-gradient(135deg,#cfe9ff,#1a4fd8)',
        pastel: 'linear-gradient(135deg,#ffc4d6,#c4ffd9,#c4d9ff)',
        candy: 'linear-gradient(135deg,#ff8ac2,#8ab8ff)',
    };
    const TILE_ART = {
        waves: ['🌊', 'linear-gradient(160deg,#0a2a5e,#1565c0,#4dd0e1)'],
        orb: ['🔮', 'radial-gradient(circle at 50% 45%,#7cffb0,#0a4d2e 70%)'],
        rain: ['🌧️', 'linear-gradient(180deg,#16324f,#3c6e9f)'],
        fire: ['🔥', 'linear-gradient(0deg,#e2571b,#f7b32b,#2a0a00)'],
        plasma: ['🌀', 'linear-gradient(135deg,#5b2a86,#a4508b,#e07be0)'],
        sparkles: ['✨', 'radial-gradient(circle at 30% 30%,#fff8,#0d0d24 40%), radial-gradient(circle at 70% 65%,#ffd70088,#0d0d24 35%)'],
        rays: ['🌟', 'conic-gradient(from 180deg at 50% 100%,#3d2b00,#ffb300,#3d2b00,#ffb300,#3d2b00)'],
        rings: ['🎯', 'repeating-radial-gradient(circle at 50% 100%,#00bcd4 0 6px,#08222e 6px 14px)'],
        aurora: ['🌌', 'linear-gradient(160deg,#031f1a,#0e8161,#4de0a8,#8a4de0)'],
        chase: ['☄️', 'linear-gradient(100deg,#0a0a2e 40%,#5c9dff 60%,#0a0a2e 75%)'],
        breathe: ['🫧', 'radial-gradient(circle at 50% 60%,#a678f0,#2a1548 75%)'],
    };

    try { modelEl.value = localStorage.getItem('shaderlab_model') || 'opus'; } catch (e) {}
    modelEl.addEventListener('change', () => {
        try { localStorage.setItem('shaderlab_model', modelEl.value); } catch (e) {}
    });

    function paramsFor(slot) {
        return params[String(slot)] || { intensity: 1.0, knobs: [0.5, 0.5, 0.5, 0.5] };
    }

    // ---- live preview ----
    socket.on('connect', () => { socket.emit('subscribe_preview'); refreshInfo(); });
    socket.on('frame', (data) => {
        const blob = new Blob([data], { type: 'image/png' });
        const url = URL.createObjectURL(blob);
        frameEl.onload = () => { if (lastFrameUrl) URL.revokeObjectURL(lastFrameUrl); lastFrameUrl = url; };
        frameEl.src = url;
    });

    // ---- conversation log ----
    function addMsg(cls, who, text) {
        const div = document.createElement('div');
        div.className = 'lab-msg ' + cls;
        const whoEl = document.createElement('span');
        whoEl.className = 'who';
        whoEl.textContent = who;
        div.appendChild(whoEl);
        div.appendChild(document.createTextNode(text));
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
    }

    function layerName(n) {
        if (!n) return 'layer';
        return n.name || (n.desc || n.prompt || 'layer').split(/[,.]/)[0].slice(0, 28);
    }

    // ---- status with elapsed ticker ----
    // Wishes run CONCURRENTLY (each layer builds on its own) — the wish
    // box never locks; busyCount tracks how many are still cooking.
    let busySince = null;
    let statusBase = '';
    let busyCount = 0;
    function paintStatus() {
        let t = statusBase;
        if (busyCount > 1) t += '  (' + busyCount + ' wishes cooking)';
        if (busySince) {
            const s = Math.round((Date.now() - busySince) / 1000);
            if (s >= 3) t += '  (' + s + 's)';
        }
        statusEl.textContent = t;
    }
    function beginWork(text) {
        busyCount += 1;
        statusBase = text || '';
        busySince = busySince || Date.now();
        statusEl.className = 'lab-status busy';
        paintStatus();
    }
    function endWork() {
        busyCount = Math.max(0, busyCount - 1);
        if (busyCount === 0) {
            statusBase = '';
            busySince = null;
            statusEl.className = 'lab-status';
        }
        paintStatus();
    }
    function setStatus(text, busy) {   // progress updates for in-flight work
        statusBase = text || '';
        if (busyCount > 0) statusEl.className = 'lab-status busy';
        paintStatus();
    }
    setInterval(paintStatus, 1000);

    // ---- param sliders (throttled) ----
    const paramTimers = {};
    function sendParams(slot, partial) {
        const key = String(slot);
        if (paramTimers[key]) return;
        paramTimers[key] = setTimeout(() => {
            paramTimers[key] = null;
            const p = paramsFor(slot);
            socket.emit('shaderlab_params',
                Object.assign({ slot: slot, intensity: p.intensity, knobs: p.knobs }, partial));
        }, 80);
    }
    socket.on('shaderlab_params', (all) => {
        if (all && typeof all === 'object') { params = all; syncSliders(); }
    });

    // ---- focus ----
    function setFocus(slot) {
        focusSlot = slot;
        if (slot == null) {
            focusChip.style.display = 'none';
            promptEl.placeholder = 'Wish something onto the lights...';
        } else {
            focusChip.style.display = 'inline';
            focusName.textContent = layerName(slots[slot] || slots[String(slot)]);
            promptEl.placeholder = 'Change “' + focusName.textContent + '”...';
        }
        renderStage();
    }
    document.getElementById('lab-unfocus').addEventListener('click', (e) => {
        e.preventDefault();
        setFocus(null);
    });

    // ---- the stage ----
    function occupiedSlots() {
        const out = [];
        for (let i = 0; i < numSlots; i++) {
            const n = slots[i] || slots[String(i)];
            if (n) out.push([i, n]);
        }
        return out;
    }

    function stActBtn(title, label, onClick) {
        const b = document.createElement('button');
        b.className = 'st-act';
        b.title = title;
        b.textContent = label;
        b.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
        return b;
    }

    function renderStage() {
        slotsEl.innerHTML = '';
        const occ = occupiedSlots();
        if (!occ.length) {
            const d = document.createElement('div');
            d.id = 'lab-empty-stage';
            d.textContent = 'Nothing playing — wish for something above, or tap a pattern tile.';
            slotsEl.appendChild(d);
            return;
        }
        occ.forEach(([i, n]) => {
            if (n.pending) {        // a wish still cooking: ghost tile
                const ghost = document.createElement('div');
                ghost.className = 'stage-tile pending';
                const gtop = document.createElement('div');
                gtop.className = 'st-top';
                const gname = document.createElement('span');
                gname.className = 'st-name';
                gname.textContent = '✦ making…';
                gtop.appendChild(gname);
                const gx = document.createElement('button');
                gx.className = 'st-x';
                gx.title = 'Cancel';
                gx.textContent = '✕';
                gx.addEventListener('click', (e) => {
                    e.stopPropagation();
                    socket.emit('shaderlab_clear', { slot: i });
                });
                gtop.appendChild(gx);
                ghost.appendChild(gtop);
                const gdesc = document.createElement('div');
                gdesc.className = 'st-desc';
                gdesc.textContent = n.desc || '';
                ghost.appendChild(gdesc);
                slotsEl.appendChild(ghost);
                return;
            }
            const tile = document.createElement('div');
            tile.className = 'stage-tile' + (focusSlot === i ? ' focus' : '');

            const top = document.createElement('div');
            top.className = 'st-top';
            const dot = document.createElement('span');
            dot.className = 'now-dot live';
            const name = document.createElement('span');
            name.className = 'st-name';
            name.textContent = layerName(n);
            top.appendChild(dot);
            top.appendChild(name);

            const actions = document.createElement('span');
            actions.className = 'st-actions';
            if (n.origin === 'builtin') {
                actions.appendChild(stActBtn('Shuffle this layer (new colors & pace)', '🎲', () => {
                    socket.emit('shaderlab_restyle', { slot: i, shuffle: true });
                }));
                const audioCycle = ['none', 'bass', 'beat', 'energy'];
                actions.appendChild(stActBtn('Cycle music reactivity (off → bass → beat → energy)', '♪', () => {
                    const cur = (n.params && n.params.audio) || 'none';
                    const next = audioCycle[(audioCycle.indexOf(cur) + 1) % audioCycle.length];
                    socket.emit('shaderlab_restyle', { slot: i, audio: next });
                }));
            }
            const x = document.createElement('button');
            x.className = 'st-x';
            x.title = 'Take this off the lights';
            x.textContent = '✕';
            x.addEventListener('click', (e) => {
                e.stopPropagation();
                if (focusSlot === i) setFocus(null);
                socket.emit('shaderlab_clear', { slot: i });
            });
            actions.appendChild(x);
            top.appendChild(actions);
            tile.appendChild(top);

            const desc = document.createElement('div');
            desc.className = 'st-desc';
            desc.textContent = n.desc || n.prompt || '';
            desc.title = desc.textContent;
            tile.appendChild(desc);

            // intensity — always present
            const intRow = document.createElement('div');
            intRow.className = 'knob-row';
            const intLab = document.createElement('label');
            intLab.textContent = 'level';
            intLab.title = 'How strongly this layer shows';
            const intInp = document.createElement('input');
            intInp.type = 'range'; intInp.min = 0; intInp.max = 1000;
            intInp.value = Math.round(paramsFor(i).intensity * 1000);
            intInp.addEventListener('click', (e) => e.stopPropagation());
            intInp.addEventListener('input', () => {
                const p = paramsFor(i);
                p.intensity = intInp.value / 1000;
                params[String(i)] = p;
                sendParams(i, { intensity: p.intensity });
            });
            intRow.appendChild(intLab);
            intRow.appendChild(intInp);
            tile.appendChild(intRow);

            // knobs — revealed when focused
            if (focusSlot === i) {
                (n.knobs || []).forEach((label, k) => {
                    const row = document.createElement('div');
                    row.className = 'knob-row';
                    const lab = document.createElement('label');
                    lab.textContent = label;
                    lab.title = label + ' (iKnob' + k + ')';
                    const inp = document.createElement('input');
                    inp.type = 'range'; inp.min = 0; inp.max = 1000;
                    inp.value = Math.round((paramsFor(i).knobs[k] != null ? paramsFor(i).knobs[k] : 0.5) * 1000);
                    inp.addEventListener('click', (e) => e.stopPropagation());
                    inp.addEventListener('input', () => {
                        const p = paramsFor(i);
                        p.knobs[k] = inp.value / 1000;
                        params[String(i)] = p;
                        sendParams(i, { knobs: p.knobs.slice() });
                    });
                    row.appendChild(lab);
                    row.appendChild(inp);
                    tile.appendChild(row);
                });
                if (n.origin === 'builtin') {
                    const dots = document.createElement('div');
                    dots.className = 'pal-dots';
                    PALETTES.forEach((pal) => {
                        const dotEl = document.createElement('span');
                        dotEl.className = 'pal-dot';
                        dotEl.title = pal + ' palette';
                        dotEl.style.background = PAL_DOT_ART[pal] || '#888';
                        dotEl.addEventListener('click', (e) => {
                            e.stopPropagation();
                            socket.emit('shaderlab_restyle', { slot: i, palette: pal });
                        });
                        dots.appendChild(dotEl);
                    });
                    tile.appendChild(dots);
                }
            }

            tile.addEventListener('click', () => {
                setFocus(focusSlot === i ? null : i);
            });
            slotsEl.appendChild(tile);
        });
    }

    function syncSliders() {
        slotsEl.querySelectorAll('.stage-tile').forEach((tile) => {
            // rebuilt wholesale by renderStage; only live-update via focus
        });
        renderStage();
    }

    socket.on('shaderlab_live', (d) => {
        if (d && d.slots) {
            slots = d.slots;
            if (focusSlot != null && !(slots[focusSlot] || slots[String(focusSlot)]))
                focusSlot = null;   // the focused layer went away
            setFocus(focusSlot);
            renderLibrary();
        }
    });

    // ---- stage toggle (built-in empty weather set = bare stage) ----
    const BLANK_SET = 'Blank Canvas';
    const stageBtn = document.getElementById('lab-stage');
    let currentSet = null;
    let lastRealSet = null;

    function paintStageBtn() {
        if (currentSet === BLANK_SET) {
            stageBtn.textContent = 'Restore ' + (lastRealSet || 'weather set');
            stageBtn.disabled = !lastRealSet;
        } else {
            stageBtn.textContent = 'Blank stage';
            stageBtn.disabled = currentSet == null;
        }
    }
    stageBtn.addEventListener('click', () => {
        if (currentSet === BLANK_SET) {
            if (lastRealSet)
                socket.emit('change_weather_set', { set_name: lastRealSet });
        } else {
            socket.emit('change_weather_set', { set_name: BLANK_SET });
            addMsg('lab', 'Shader Lab', 'Blank stage — the weather scene is '
                + 'silenced; only your layers show. The button restores it.');
        }
    });

    // ---- running in the engine (read-only) ----
    socket.on('state_update', (s) => {
        if (!s) return;
        currentSet = s.current_weather_set || currentSet;
        if (currentSet && currentSet !== BLANK_SET) lastRealSet = currentSet;
        paintStageBtn();
        const effects = s.active_effects || [];
        const liveCount = occupiedSlots().length;
        engineList.innerHTML = '';
        if (!effects.length) {
            engineList.innerHTML = '<span style="color:#667">no active events</span>';
        }
        effects.forEach((nm) => {
            const chip = document.createElement('span');
            const isSlot = nm === 'shader_live_shader' || nm === 'live_shader';
            chip.className = 'engine-chip' + (isSlot && liveCount ? ' you' : '');
            chip.textContent = isSlot
                ? 'your layers (' + liveCount + ' live)' : nm;
            chip.title = isSlot
                ? 'The Shader Lab layers — what this page controls'
                : 'Scheduled by the current weather set';
            engineList.appendChild(chip);
        });
        engineMeta.textContent = (s.current_weather_set || '')
            + (s.current_weather ? ' · ' + s.current_weather : '');
    });

    // ---- pattern palette ----
    function renderBuiltins(list) {
        builtinList.innerHTML = '';
        (list || []).forEach((p) => {
            const art = TILE_ART[p.id] || ['✦', 'linear-gradient(135deg,#333,#555)'];
            const tile = document.createElement('div');
            tile.className = 'pal-tile';
            tile.style.background = art[1];
            tile.title = p.desc + ' — tap to put it on the lights';
            const emoji = document.createElement('span');
            emoji.className = 'pt-emoji';
            emoji.textContent = art[0];
            const nm = document.createElement('span');
            nm.className = 'pt-name';
            nm.textContent = p.id;
            tile.appendChild(emoji);
            tile.appendChild(nm);
            tile.addEventListener('click', () => {
                beginWork('Compiling…');
                socket.emit('shaderlab_builtin', { id: p.id, slot: focusSlot });
            });
            builtinList.appendChild(tile);
        });
    }

    // ---- saved patterns (advanced drawer) ----
    function patBtn(label, title, cls, onClick) {
        const b = document.createElement('button');
        b.className = 'pat-btn' + (cls ? ' ' + cls : '');
        b.textContent = label;
        b.title = title;
        b.addEventListener('click', onClick);
        return b;
    }

    function renderLibrary() {
        savedList.innerHTML = '';
        if (!savedPatterns.length) {
            savedList.innerHTML = '<div style="color:#667; font-size:0.85em; margin-top:6px">'
                + 'Nothing saved yet — make a pattern, then name and save it.</div>';
            return;
        }
        const liveNames = {};
        occupiedSlots().forEach(([i, n]) => {
            if (n.origin === 'library' && n.name) liveNames[n.name] = true;
        });
        savedPatterns.forEach((p) => {
            const row = document.createElement('div');
            row.className = 'pat-row' + (liveNames[p.name] ? ' live' : '');
            const name = document.createElement('span');
            name.className = 'pat-name';
            name.textContent = p.name;
            row.appendChild(name);
            const prompt = document.createElement('span');
            prompt.className = 'pat-prompt';
            prompt.textContent = p.prompt || '';
            prompt.title = p.prompt || '';
            row.appendChild(prompt);
            if (liveNames[p.name]) {
                const tag = document.createElement('span');
                tag.className = 'pat-live-tag';
                tag.textContent = '● live';
                row.appendChild(tag);
            }
            row.appendChild(patBtn('▶ Play', 'Put this on the lights', '', () => {
                beginWork('Compiling…');
                socket.emit('shaderlab_load', { name: p.name, slot: focusSlot });
            }));
            row.appendChild(patBtn('✎ Edit', 'Load the code into the editor (nothing compiles)', '', () => {
                socket.emit('shaderlab_get', { name: p.name });
            }));
            row.appendChild(patBtn('✕', 'Delete this saved pattern', 'danger', () => {
                if (confirm('Delete saved pattern “' + p.name + '”?')) {
                    socket.emit('shaderlab_delete', { name: p.name });
                }
            }));
            savedList.appendChild(row);
        });
    }

    socket.on('shaderlab_library_changed', refreshInfo);
    socket.on('shaderlab_pattern', (p) => {
        if (!p || typeof p.glsl !== 'string') return;
        codeEl.value = p.glsl;
        saveNameEl.value = p.name || '';
        if (p.prompt) lastPrompt = p.prompt;
        document.getElementById('lab-editor-details').open = true;
        addMsg('lab', 'Shader Lab', '“' + p.name + '” loaded into the editor.');
    });

    // ---- progress + result ----
    socket.on('shaderlab_progress', (p) => {
        const stage = (p && p.stage) || '';
        if (stage === 'thinking') setStatus('Designing…', true);
        else if (stage === 'writing') {
            setStatus(p.coding ? 'Writing the code…'
                : (p.text ? 'Designing: ' + p.text : 'Designing…'), true);
        }
        else if (stage === 'compiling') setStatus('Compiling…', true);
        else if (stage === 'repairing') setStatus('Fixing a compile error (attempt ' + p.attempt + ')…', true);
    });

    socket.on('shaderlab_result', (r) => {
        endWork();
        errEl.style.display = 'none';
        if (!r) return;
        if (r.status === 'ok') {
            addMsg('lab', 'Shader Lab', r.message || 'On the lights now.');
            if (r.code) codeEl.value = r.code;
        } else if (r.status === 'cleared') {
            if (r.message) addMsg('lab', 'Shader Lab', r.message);
        } else {
            addMsg('err', 'Shader Lab', 'That one failed: ' + (r.error || 'unknown error'));
            if (r.code) codeEl.value = r.code;
            if (r.error) { errEl.textContent = r.error; errEl.style.display = 'block'; }
            document.getElementById('lab-editor-details').open = true;
        }
    });

    // ---- actions ----
    function generate() {
        const prompt = promptEl.value.trim();
        if (!prompt) return;
        lastPrompt = prompt;
        addMsg('you', focusSlot == null ? 'You'
            : 'You → ' + layerName(slots[focusSlot] || slots[String(focusSlot)]), prompt);
        promptEl.value = '';
        beginWork('Designing…');
        socket.emit('shaderlab_generate',
            { prompt, model: modelEl.value, slot: focusSlot });
    }
    genBtn.addEventListener('click', generate);
    promptEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') generate(); });

    document.getElementById('lab-compile').addEventListener('click', () => {
        const code = codeEl.value;
        if (!code.trim()) return;
        beginWork('Compiling…');
        socket.emit('shaderlab_compile', { code, slot: focusSlot });
    });
    codeEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            document.getElementById('lab-compile').click();
        }
    });

    document.getElementById('lab-clearstage').addEventListener('click', () => {
        setFocus(null);
        socket.emit('shaderlab_clear', { all: true });
    });

    document.getElementById('lab-newsession').addEventListener('click', () => {
        socket.emit('shaderlab_reset_session',
            { slot: focusSlot == null ? 0 : focusSlot });
        addMsg('lab', 'Shader Lab', 'Fresh conversation.');
    });

    const BLANK_TEMPLATE =
        'void mainImage(out vec4 fragColor, in vec2 fragCoord) {\n' +
        '    vec2 p = fanUV(fragCoord);          // physical space: x -1..1, y 0..1\n' +
        '    vec2 uv = fragCoord / iResolution;  // strip space: x = ray, y = outward\n' +
        '    fragColor = vec4(0.0);              // transparent = LEDs off\n' +
        '}\n';

    document.getElementById('lab-blank').addEventListener('click', () => {
        setFocus(null);
        socket.emit('shaderlab_clear', { all: true });
        codeEl.value = BLANK_TEMPLATE;
        saveNameEl.value = '';
        lastPrompt = '';
        errEl.style.display = 'none';
        addMsg('lab', 'Shader Lab', 'Starting over — the lights are clear.');
    });

    document.getElementById('lab-save').addEventListener('click', () => {
        const name = saveNameEl.value.trim();
        const code = codeEl.value;
        if (!name || !code.trim()) {
            addMsg('err', 'Shader Lab', !name
                ? 'Give the pattern a name first.'
                : 'Nothing to save — the editor is empty.');
            return;
        }
        socket.emit('shaderlab_save', { name, code, prompt: lastPrompt });
        addMsg('lab', 'Shader Lab', 'Saved as “' + name + '”.');
    });

    // ---- bootstrap ----
    function refreshInfo() {
        fetch('/api/shaderlab/info')
            .then((r) => r.json())
            .then((info) => {
                banner.style.display = info.llm_available ? 'none' : 'block';
                savedPatterns = info.saved_patterns || [];
                if (info.num_slots) numSlots = info.num_slots;
                if (info.slots) slots = info.slots;
                if (info.params) params = info.params;
                renderBuiltins(info.builtin_patterns);
                renderStage();
                renderLibrary();
            })
            .catch(() => {});
    }
    renderStage();
    refreshInfo();
})();
