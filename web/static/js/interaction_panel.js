/**
 * GL_Simple — generic interaction panel renderer.
 *
 * Draws whatever the live weather set published at /api/interaction/spec
 * and sends presses back as (control id, value) pairs. The client never
 * names an event, state or parameter: the server resolves the control id
 * against the live set's own spec, so a tab left open across a set change
 * simply stops working instead of firing the wrong set's events.
 *
 * Presentation is per set: the spec's theme block drives CSS variables,
 * so Ocean reads as water and a spooky set reads as candlelight while
 * sharing one renderer.
 */
(function () {
    const socket = createSocket();

    const heading = document.getElementById('ix-heading');
    const blurbEl = document.getElementById('ix-blurb');
    const idleEl = document.getElementById('ix-idle');
    const idleText = document.getElementById('ix-idle-text');
    const panelEl = document.getElementById('ix-panel');
    const sectionsEl = document.getElementById('ix-sections');

    let currentSet = null;      // set id the DOM was built for
    let currentSig = null;      // panel fingerprint, to skip idle rebuilds
    let suppressUntil = 0;      // don't clobber a slider the user is dragging

    function send(controlId, value) {
        socket.emit('interaction_action', {
            set: currentSet, control: controlId, value: value,
        });
    }

    function flash(el) {
        el.classList.remove('ix-flash');
        void el.offsetWidth;           // restart the animation
        el.classList.add('ix-flash');
    }

    // ---- control builders --------------------------------------------

    function buildButton(ctrl) {
        const btn = document.createElement('button');
        btn.className = 'ix-btn';
        if (ctrl.color) btn.style.setProperty('--btn-accent', ctrl.color);
        btn.innerHTML = '';
        if (ctrl.icon) {
            const icon = document.createElement('span');
            icon.className = 'ix-btn-icon';
            icon.textContent = ctrl.icon;
            btn.appendChild(icon);
        }
        const label = document.createElement('span');
        label.className = 'ix-btn-label';
        label.textContent = ctrl.label;
        btn.appendChild(label);
        if (ctrl.hint) {
            const hint = document.createElement('span');
            hint.className = 'ix-btn-hint';
            hint.textContent = ctrl.hint;
            btn.appendChild(hint);
        }
        btn.addEventListener('click', () => {
            send(ctrl.id, ctrl.value !== undefined ? ctrl.value : 1);
            flash(btn);
        });
        return btn;
    }

    function buildSlider(ctrl, values) {
        const wrap = document.createElement('div');
        wrap.className = 'ix-slider';

        const row = document.createElement('div');
        row.className = 'ix-slider-head';
        const label = document.createElement('span');
        label.textContent = ctrl.label;
        const readout = document.createElement('span');
        readout.className = 'ix-readout';
        row.appendChild(label);
        row.appendChild(readout);
        wrap.appendChild(row);

        const input = document.createElement('input');
        input.type = 'range';
        input.min = ctrl.min;
        input.max = ctrl.max;
        input.step = ctrl.step || 0.01;
        input.value = liveValue(ctrl, values, ctrl.default);
        readout.textContent = Number(input.value).toFixed(2);
        input.addEventListener('input', () => {
            readout.textContent = Number(input.value).toFixed(2);
            suppressUntil = Date.now() + 2000;
            send(ctrl.id, parseFloat(input.value));
        });
        wrap.appendChild(input);

        if (ctrl.hint) {
            const hint = document.createElement('div');
            hint.className = 'ix-hint';
            hint.textContent = ctrl.hint;
            wrap.appendChild(hint);
        }
        wrap._sync = (vals) => {
            if (Date.now() < suppressUntil) return;
            const v = liveValue(ctrl, vals, null);
            if (v === null) return;
            input.value = v;
            readout.textContent = Number(v).toFixed(2);
        };
        return wrap;
    }

    function buildToggle(ctrl, values) {
        const btn = document.createElement('button');
        btn.className = 'ix-toggle';
        btn.textContent = ctrl.label;
        let on = liveValue(ctrl, values, ctrl.off) === ctrl.on;
        const paint = () => btn.classList.toggle('on', on);
        paint();
        btn.addEventListener('click', () => {
            on = !on;
            paint();
            send(ctrl.id, on);
        });
        btn._sync = (vals) => {
            const v = liveValue(ctrl, vals, null);
            if (v === null) return;
            on = (v === ctrl.on);
            paint();
        };
        return btn;
    }

    function buildSelect(ctrl, values, currentState) {
        const wrap = document.createElement('div');
        wrap.className = 'ix-select';
        const label = document.createElement('div');
        label.className = 'ix-select-label';
        label.textContent = ctrl.label;
        wrap.appendChild(label);

        const sel = document.createElement('select');
        for (const opt of ctrl.options) {
            const o = document.createElement('option');
            o.value = opt.value;
            o.textContent = opt.label;
            sel.appendChild(o);
        }
        if (ctrl.action === 'state' && currentState) sel.value = currentState;
        else {
            const v = liveValue(ctrl, values, null);
            if (v !== null) sel.value = v;
        }
        sel.addEventListener('change', () => {
            suppressUntil = Date.now() + 2000;
            const raw = sel.value;
            send(ctrl.id, ctrl.action === 'state' ? raw : parseFloat(raw));
        });
        wrap.appendChild(sel);
        wrap._sync = (vals, state) => {
            if (Date.now() < suppressUntil) return;
            if (ctrl.action === 'state') { if (state) sel.value = state; return; }
            const v = liveValue(ctrl, vals, null);
            if (v !== null) sel.value = v;
        };
        return wrap;
    }

    /** Current value of a param/signal control, or `fallback`. */
    function liveValue(ctrl, values, fallback) {
        if (!values) return fallback;
        if (ctrl.action === 'param') {
            const v = (values.params || {})[ctrl.param];
            return v === undefined ? fallback : v;
        }
        if (ctrl.action === 'signal') {
            const rec = (values.signals || {})[ctrl.signal];
            return rec === undefined ? fallback : rec.value;
        }
        return fallback;
    }

    // ---- panel assembly ----------------------------------------------

    function fingerprint(panel) {
        return JSON.stringify([panel.set, panel.label, panel.sections]);
    }

    function applyTheme(theme) {
        const root = document.documentElement;
        root.style.setProperty('--ix-accent', theme.accent);
        root.style.setProperty('--ix-bg', theme.bg);
        root.style.setProperty('--ix-panel', theme.panel);
        root.style.setProperty('--ix-text', theme.text);
        document.body.classList.add('ix-themed');
    }

    let syncers = [];

    function render(spec) {
        const panel = spec.panel;
        applyTheme(panel.theme);
        heading.textContent = panel.title || panel.label;
        blurbEl.textContent = panel.blurb || '';
        blurbEl.style.display = panel.blurb ? '' : 'none';

        sectionsEl.innerHTML = '';
        syncers = [];
        for (const section of panel.sections) {
            const card = document.createElement('div');
            card.className = 'ix-section';
            if (section.title) {
                const h = document.createElement('h3');
                h.textContent = section.title;
                card.appendChild(h);
            }
            if (section.note) {
                const n = document.createElement('p');
                n.className = 'ix-note';
                n.textContent = section.note;
                card.appendChild(n);
            }
            const body = document.createElement('div');
            body.className = 'ix-controls ix-layout-' + (section.layout || 'grid');
            for (const ctrl of section.controls) {
                let el;
                if (ctrl.type === 'slider') el = buildSlider(ctrl, spec.values);
                else if (ctrl.type === 'toggle') el = buildToggle(ctrl, spec.values);
                else if (ctrl.type === 'select') el = buildSelect(ctrl, spec.values, spec.current_state);
                else el = buildButton(ctrl);
                if (el._sync) syncers.push(el._sync);
                body.appendChild(el);
            }
            card.appendChild(body);
            sectionsEl.appendChild(card);
        }

        idleEl.style.display = 'none';
        panelEl.style.display = '';
    }

    function showIdle(setName) {
        currentSet = null;
        currentSig = null;
        syncers = [];
        panelEl.style.display = 'none';
        idleEl.style.display = '';
        heading.textContent = 'Interaction';
        document.body.classList.remove('ix-themed');
        idleText.textContent = setName
            ? `The "${setName}" set has no hands-on controls.`
            : 'Nothing to play with right now.';
    }

    async function poll() {
        try {
            const r = await fetch('/api/interaction/spec');
            const spec = await r.json();
            if (!spec.available) { showIdle(spec.current_set); return; }
            if (spec.panel.page) {
                // The live set owns a bespoke page — send the operator there.
                window.location.href = spec.panel.page;
                return;
            }
            const sig = fingerprint(spec.panel);
            if (sig !== currentSig) {
                currentSet = spec.panel.set;
                currentSig = sig;
                render(spec);
            } else {
                for (const sync of syncers) sync(spec.values, spec.current_state);
            }
        } catch (e) { /* keep the last good panel on transient errors */ }
    }

    poll();
    setInterval(poll, 2000);
})();
