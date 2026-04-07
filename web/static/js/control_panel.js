/**
 * GL_Simple — Control Panel page logic
 * Handles global modifiers, weather parameter overrides,
 * audio visualization, and weather set management.
 */

// ---- State ----
let globalModifiers = {};
let globalSchema = {};
let weatherParams = {};
let activeOverrides = {};
let audioSummary = {};
let transitionState = {};
let socket = null;

// Interaction cooldown: after user changes a value, ignore server updates
// for that control briefly to prevent snapping back
const interactionCooldowns = {};  // key -> timestamp
const COOLDOWN_MS = 1500;  // 1.5 seconds

// Release cooldown: after releasing overrides, briefly ignore server override state
// to prevent the UI flickering back before the server processes the clear
let releaseCooldownUntil = 0;
const RELEASE_COOLDOWN_MS = 2000;

function setInteractionCooldown(key) {
    interactionCooldowns[key] = Date.now();
}

function isOnCooldown(key) {
    const ts = interactionCooldowns[key];
    if (!ts) return false;
    if (Date.now() - ts < COOLDOWN_MS) return true;
    delete interactionCooldowns[key];
    return false;
}

// Whether to show all params or only those relevant to the current weather set
let showAllParams = false;
let allowedOutputParams = null;  // null = show all, Set = filter

// Weather param categories for grouping in the UI
const PARAM_CATEGORIES = {
    "Atmosphere": ["fog_strength", "fog_color", "cloudyness", "starryness", "celestial_visibility"],
    "Precipitation": ["rain", "lightning_probability"],
    "Wind & Motion": ["wind", "sand_density", "wave_speed", "wave_amplitude", "tide_level"],
    "Life": ["firefly_density", "bioluminescence", "bubble_density", "marine_life_activity", "kelp_density", "tree_growth"],
    "Special": ["volcano_level", "meteor_rate"],
};

// Reasonable ranges for output parameters (for slider rendering)
const PARAM_RANGES = {
    "fog_strength": { min: 0, max: 1.5, step: 0.05 },
    "cloudyness": { min: 0, max: 1, step: 0.05 },
    "starryness": { min: 0, max: 1, step: 0.05 },
    "celestial_visibility": { min: 0, max: 1, step: 0.05 },
    "rain": { min: 0, max: 1, step: 0.05 },
    "lightning_probability": { min: 0, max: 1, step: 0.05 },
    "wind": { min: -2, max: 2, step: 0.05 },
    "sand_density": { min: 0, max: 1, step: 0.05 },
    "wave_speed": { min: 0, max: 1.5, step: 0.05 },
    "wave_amplitude": { min: 0, max: 1.5, step: 0.05 },
    "tide_level": { min: 0, max: 1, step: 0.05 },
    "firefly_density": { min: 0, max: 2, step: 0.05 },
    "bioluminescence": { min: 0, max: 1, step: 0.05 },
    "bubble_density": { min: 0, max: 1, step: 0.05 },
    "marine_life_activity": { min: 0, max: 1, step: 0.05 },
    "kelp_density": { min: 0, max: 1, step: 0.05 },
    "tree_growth": { min: 0, max: 1.5, step: 0.05 },
    "volcano_level": { min: 0, max: 1, step: 0.05 },
    "meteor_rate": { min: 0, max: 0.5, step: 0.01 },
};

const WEATHER_SET_NAMES = {
    "peaceful_forest": "Peaceful Forest",
    "storm_world": "Storm World",
    "desert_realm": "Desert Realm",
    "ethereal_mist": "Ethereal Mist",
    "cosmic_night": "Cosmic Night",
    "full_spectrum": "Full Spectrum",
    "cyberpunk": "Cyberpunk",
    "ocean": "Ocean",
    "test": "Test"
};

function formatParamName(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatStateName(state) {
    return (state || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// ---- WebSocket Connection ----
function connectSocket() {
    socket = createSocket();  // from socket-client.js

    // State updates from server (5 Hz)
    socket.on('state_update', (data) => {
        globalModifiers = data.global_modifiers || {};
        // Respect release cooldown: don't restore server overrides right after a release
        if (Date.now() < releaseCooldownUntil) {
            // Keep local activeOverrides (empty after release)
        } else {
            activeOverrides = data.active_overrides || {};
        }
        weatherParams = data.weather_params || {};
        transitionState = data.transition || {};

        // Track allowed output params for filtering
        if (data.allowed_output_params) {
            allowedOutputParams = new Set(data.allowed_output_params);
        }

        // Sync global sliders (skip if user recently interacted)
        for (const [key, val] of Object.entries(globalModifiers)) {
            if (isOnCooldown(`global-${key}`)) continue;
            const slider = document.getElementById(`global-${key}`);
            const display = document.getElementById(`global-val-${key}`);
            if (slider && !slider.matches(':active')) slider.value = val;
            if (display) display.textContent = parseFloat(val).toFixed(2) + 'x';
        }

        // Update weather info
        const setName = WEATHER_SET_NAMES[data.current_weather_set] || data.current_weather_set;
        document.getElementById('current-weather-set').textContent = setName;
        document.getElementById('current-weather').textContent = formatParamName(data.current_weather);
        document.getElementById('season-display').textContent = ((data.season || 0) * 100).toFixed(1) + '%';

        const factor = data.brightness_limiting_factor ?? 1.0;
        const bEl = document.getElementById('brightness-limiting-display');
        bEl.textContent = factor.toFixed(3);
        bEl.style.color = factor > 1.001 ? 'rgb(255, 200, 80)' : '#00ffff';

        document.getElementById('fps-display').textContent = (data.fps || '--') + ' FPS';

        // Update ambient sound display
        const ambientEl = document.getElementById('ambient-sound-display');
        if (data.ambient_sound) {
            // Strip file extension and clean up the name
            const name = data.ambient_sound.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' ');
            ambientEl.textContent = name;
            ambientEl.style.opacity = '1';
        } else {
            ambientEl.textContent = '--';
            ambientEl.style.opacity = '0.7';
        }

        updateTransitionDisplay();
        renderWeatherParams();
        updatePerformancePanel(data);

        const count = Object.keys(activeOverrides).length;
        document.getElementById('override-count').textContent = `${count} override${count !== 1 ? 's' : ''}`;
    });

    // Audio updates from server (10 Hz)
    socket.on('audio_update', (data) => {
        audioSummary = data;
    });

    // Weather change events
    socket.on('weather_changed', () => {
        loadWeatherSetInfo();
    });
}

// ---- Initialization ----
async function init() {
    await loadGlobalsSchema();
    renderGlobals();
    setupAudioCanvas();
    requestAnimationFrame(drawAudio);

    connectSocket();

    await loadWeatherSetInfo();
    setInterval(loadWeatherSetInfo, 3000);
}

// ---- Global Modifiers ----
async function loadGlobalsSchema() {
    try {
        const response = await fetch('/api/globals/schema');
        globalSchema = await response.json();
    } catch (e) {
        console.error('Failed to load globals schema:', e);
    }
}

function renderGlobals() {
    const container = document.getElementById('globals-container');
    container.innerHTML = '';

    for (const [key, schema] of Object.entries(globalSchema)) {
        const card = document.createElement('div');
        card.className = 'control-card';
        card.innerHTML = `
            <div class="control-label">
                <span>${schema.label}</span>
                <span class="control-value" id="global-val-${key}">${(globalModifiers[key] || schema.default).toFixed(2)}x</span>
            </div>
            <div class="control-sublabel">${schema.description}</div>
            <input type="range" id="global-${key}"
                min="${schema.min}" max="${schema.max}" step="${schema.step}"
                value="${globalModifiers[key] || schema.default}"
                oninput="onGlobalChange('${key}', this.value)">
        `;
        container.appendChild(card);
    }
}

function onGlobalChange(modifier, value) {
    value = parseFloat(value);
    globalModifiers[modifier] = value;
    setInteractionCooldown(`global-${modifier}`);
    const display = document.getElementById(`global-val-${modifier}`);
    if (display) display.textContent = value.toFixed(2) + 'x';

    if (socket && socket.connected) {
        socket.emit('update_global', { modifier, value });
    }
}

function resetGlobals() {
    for (const [key, schema] of Object.entries(globalSchema)) {
        onGlobalChange(key, schema.default);
        const slider = document.getElementById(`global-${key}`);
        if (slider) slider.value = schema.default;
    }
}

// ---- Transition Display ----
function updateTransitionDisplay() {
    const row = document.getElementById('transition-row');
    const barRow = document.getElementById('transition-bar-row');
    const targetEl = document.getElementById('transition-target');
    const bar = document.getElementById('transition-bar');

    if (transitionState.transitioning) {
        row.style.display = 'flex';
        barRow.style.display = 'block';
        targetEl.textContent = formatParamName(transitionState.target);
        bar.style.width = (transitionState.progress * 100).toFixed(1) + '%';
    } else {
        row.style.display = 'none';
        barRow.style.display = 'none';
    }
}

// ---- Weather Parameters ----
function renderWeatherParams() {
    const container = document.getElementById('params-container');

    // Merge weather params with override values (overrides take precedence for display)
    const numericParams = {};
    for (const [key, val] of Object.entries(weatherParams)) {
        if (typeof val === 'number') {
            // Filter by allowed params unless showing all
            if (!showAllParams && allowedOutputParams && !allowedOutputParams.has(key)) {
                // Still show if it has an active override
                if (!(key in activeOverrides)) continue;
            }
            numericParams[key] = (key in activeOverrides) ? activeOverrides[key] : val;
        }
    }

    const categorized = new Set();
    let html = '';

    for (const [category, keys] of Object.entries(PARAM_CATEGORIES)) {
        const activeKeys = keys.filter(k => k in numericParams);
        if (activeKeys.length === 0) continue;

        html += `<div class="params-category">
            <div class="params-category-header">${category}</div>
            <div class="controls-grid">`;
        for (const key of activeKeys) {
            html += buildParamCard(key, numericParams[key]);
            categorized.add(key);
        }
        html += '</div></div>';
    }

    const uncategorized = Object.keys(numericParams).filter(k => !categorized.has(k));
    if (uncategorized.length > 0) {
        html += `<div class="params-category">
            <div class="params-category-header">Other</div>
            <div class="controls-grid">`;
        for (const key of uncategorized) {
            html += buildParamCard(key, numericParams[key]);
        }
        html += '</div></div>';
    }

    // Check if any param slider is being dragged or on cooldown
    const anyActive = document.querySelector('.param-slider:active');
    const anyCooldown = Object.keys(numericParams).some(k => isOnCooldown(`param-${k}`));

    if (!anyActive && !anyCooldown) {
        container.innerHTML = html;
    } else {
        // Just update value displays without re-rendering sliders
        for (const [key, val] of Object.entries(numericParams)) {
            if (isOnCooldown(`param-${key}`)) continue;
            const display = document.getElementById(`param-val-${key}`);
            if (display) display.textContent = val.toFixed(3);
        }
    }
}

function buildParamCard(key, value) {
    const isOverridden = key in activeOverrides;
    const range = PARAM_RANGES[key] || { min: Math.min(0, value - 1), max: Math.max(1, value + 1), step: 0.05 };
    const cardClass = isOverridden ? 'control-card overridden' : 'control-card';
    const releaseBtn = isOverridden
        ? `<button class="release-btn" onclick="clearOverride('${key}')">Release</button>`
        : '';

    return `<div class="${cardClass}">
        <div class="control-label">
            <span>${formatParamName(key)}</span>
            <span>
                <span class="control-value" id="param-val-${key}">${value.toFixed(3)}</span>
                ${releaseBtn}
            </span>
        </div>
        <input type="range" class="param-slider" id="param-${key}"
            min="${range.min}" max="${range.max}" step="${range.step}"
            value="${value}"
            oninput="onParamOverride('${key}', this.value)">
    </div>`;
}

function onParamOverride(param, value) {
    value = parseFloat(value);
    activeOverrides[param] = value;
    setInteractionCooldown(`param-${param}`);
    const display = document.getElementById(`param-val-${param}`);
    if (display) display.textContent = value.toFixed(3);

    const card = document.getElementById(`param-${param}`)?.closest('.control-card');
    if (card) card.className = 'control-card overridden';

    if (socket && socket.connected) {
        socket.emit('set_override', { param, value });
    }
}

function clearOverride(param) {
    delete activeOverrides[param];
    releaseCooldownUntil = Date.now() + RELEASE_COOLDOWN_MS;
    if (socket && socket.connected) {
        socket.emit('clear_override', { param });
    }
    renderWeatherParams();
}

function clearAllOverrides() {
    activeOverrides = {};
    releaseCooldownUntil = Date.now() + RELEASE_COOLDOWN_MS;
    if (socket && socket.connected) {
        socket.emit('clear_all_overrides', {});
    }
    renderWeatherParams();
}

function toggleShowAllParams() {
    showAllParams = !showAllParams;
    const btn = document.getElementById('toggle-all-params-btn');
    btn.textContent = showAllParams ? 'Show Set Only' : 'Show All';
    renderWeatherParams();
}

// ---- Performance Panel ----
function updatePerformancePanel(data) {
    document.getElementById('perf-fps').textContent = (data.fps || '--');
    const effectCount = (data.active_effects || []).length;
    document.getElementById('perf-effect-count').textContent = effectCount;

    const factor = data.brightness_limiting_factor ?? 1.0;
    const perfBrightness = document.getElementById('perf-brightness');
    perfBrightness.textContent = factor.toFixed(3);
    perfBrightness.style.color = factor > 1.001 ? 'rgb(255, 200, 80)' : '#00ffff';

    const webBrightness = (data.global_modifiers || {}).brightness ?? 1.0;
    document.getElementById('perf-web-brightness').textContent = webBrightness.toFixed(2) + 'x';

    // Active effects list
    const listEl = document.getElementById('perf-effects-list');
    const effects = data.active_effects || [];
    if (effects.length === 0) {
        listEl.innerHTML = '<span style="opacity: 0.5;">None</span>';
    } else {
        listEl.innerHTML = effects.map(name =>
            `<span class="effect-tag">${formatParamName(name)}</span>`
        ).join('');
    }
}

// ---- Audio Visualization ----
let audioCanvas, audioCtx;
let peakHold = new Float32Array(32);  // Peak hold values
const PEAK_DECAY = 0.97;  // How fast peaks decay per frame

function setupAudioCanvas() {
    audioCanvas = document.getElementById('audio-canvas');
    audioCtx = audioCanvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
    const rect = audioCanvas.getBoundingClientRect();
    audioCanvas.width = rect.width * window.devicePixelRatio;
    audioCanvas.height = rect.height * window.devicePixelRatio;
    audioCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

function drawAudio() {
    const w = audioCanvas.width / window.devicePixelRatio;
    const h = audioCanvas.height / window.devicePixelRatio;
    audioCtx.clearRect(0, 0, w, h);

    const bands = audioSummary.bands;
    if (!bands || bands.length === 0) {
        requestAnimationFrame(drawAudio);
        return;
    }

    const n = bands.length;
    const barW = w / n - 1;
    const maxVal = 3.0;

    // Ensure peakHold array matches band count
    if (peakHold.length !== n) {
        peakHold = new Float32Array(n);
    }

    for (let i = 0; i < n; i++) {
        const val = Math.min(bands[i] / maxVal, 1.0);
        const barH = val * h;
        const x = i * (barW + 1);

        // Update peak hold
        if (val > peakHold[i]) {
            peakHold[i] = val;
        } else {
            peakHold[i] *= PEAK_DECAY;
        }

        // Color by frequency: cyan -> blue -> violet
        const hue = 180 + (i / n) * 140;
        audioCtx.fillStyle = `hsla(${hue}, 80%, 60%, 0.8)`;
        audioCtx.fillRect(x, h - barH, barW, barH);

        // Draw peak hold line
        const peakY = h - (peakHold[i] * h);
        audioCtx.fillStyle = `hsla(${hue}, 90%, 80%, 0.9)`;
        audioCtx.fillRect(x, peakY, barW, 2);
    }

    const powerEl = document.getElementById('audio-power');
    if (audioSummary.total_power !== undefined) {
        powerEl.textContent = `Power: ${audioSummary.total_power.toFixed(1)} | Sensitivity: ${(audioSummary.sensitivity || 1).toFixed(1)}x`;
    }

    requestAnimationFrame(drawAudio);
}

// ---- Weather Set Management ----
let _lastWeatherSet = null;
let _lastLocked = null;
let _weatherStateLocked = false;

function updateLockButton(locked) {
    const btn = document.getElementById('weather-state-lock-btn');
    if (locked) {
        btn.textContent = 'Locked';
        btn.style.background = 'rgba(255, 100, 100, 0.2)';
        btn.style.borderColor = 'rgba(255, 100, 100, 0.6)';
        btn.style.color = 'rgba(255, 180, 180, 1)';
    } else {
        btn.textContent = 'Unlocked';
        btn.style.background = 'rgba(100, 255, 100, 0.2)';
        btn.style.borderColor = 'rgba(100, 255, 100, 0.6)';
        btn.style.color = 'rgba(180, 255, 180, 1)';
    }
}

async function toggleWeatherStateLock() {
    const currentlyLocked = _lastLocked !== false;
    const newLocked = !currentlyLocked;
    try {
        const response = await fetch('/api/weather_state/set_lock', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ locked: newLocked })
        });
        const result = await response.json();
        if (result.success) {
            _lastLocked = result.locked;
            _lastWeatherSet = null;
            updateLockButton(result.locked);
            await loadWeatherSetInfo();
        }
    } catch (e) {
        console.error('Failed to toggle lock:', e);
    }
}

function updateStateLockButton(locked) {
    const btn = document.getElementById('weather-state-locked-btn');
    if (locked) {
        btn.textContent = 'On';
        btn.style.background = 'rgba(255, 200, 0, 0.2)';
        btn.style.borderColor = 'rgba(255, 200, 0, 0.7)';
        btn.style.color = 'rgba(255, 220, 100, 1)';
    } else {
        btn.textContent = 'Off';
        btn.style.background = 'rgba(255, 255, 255, 0.1)';
        btn.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        btn.style.color = '#fff';
    }
}

async function toggleWeatherStateLocked() {
    const newLocked = !_weatherStateLocked;
    try {
        const response = await fetch('/api/weather_state/set_state_locked', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ locked: newLocked })
        });
        const result = await response.json();
        if (result.success) {
            _weatherStateLocked = result.locked;
            updateStateLockButton(result.locked);
        }
    } catch (e) {
        console.error('Failed to toggle state lock:', e);
    }
}

let _instantTransitions = false;

function updateInstantTransitionButton(enabled) {
    const btn = document.getElementById('instant-transition-btn');
    if (enabled) {
        btn.textContent = 'On';
        btn.style.background = 'rgba(255, 200, 0, 0.2)';
        btn.style.borderColor = 'rgba(255, 200, 0, 0.7)';
        btn.style.color = 'rgba(255, 220, 100, 1)';
    } else {
        btn.textContent = 'Off';
        btn.style.background = 'rgba(255, 255, 255, 0.1)';
        btn.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        btn.style.color = '#fff';
    }
}

function toggleInstantTransitions() {
    _instantTransitions = !_instantTransitions;
    updateInstantTransitionButton(_instantTransitions);
    if (socket && socket.connected) {
        socket.emit('set_flag', { key: 'instant_transitions', value: _instantTransitions });
    }
}

let _flipX = false;

function updateFlipXButton(enabled) {
    const btn = document.getElementById('flip-x-btn');
    if (!btn) return;
    if (enabled) {
        btn.textContent = 'On';
        btn.style.background = 'rgba(255, 200, 0, 0.2)';
        btn.style.borderColor = 'rgba(255, 200, 0, 0.7)';
        btn.style.color = 'rgba(255, 220, 100, 1)';
    } else {
        btn.textContent = 'Off';
        btn.style.background = 'rgba(255, 255, 255, 0.1)';
        btn.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        btn.style.color = '#fff';
    }
}

function toggleFlipX() {
    _flipX = !_flipX;
    updateFlipXButton(_flipX);
    if (socket && socket.connected) {
        socket.emit('set_flag', { key: 'flip_x', value: _flipX });
    }
}

function populateStateSelector(states) {
    const stateSelector = document.getElementById('weather-state-selector');
    stateSelector.innerHTML = '<option value="">-- Select a weather --</option>';
    states.forEach(state => {
        const option = document.createElement('option');
        option.value = state;
        option.textContent = formatStateName(state);
        stateSelector.appendChild(option);
    });
}

async function loadWeatherSetInfo() {
    try {
        const response = await fetch('/api/weather_set/info');
        const data = await response.json();

        // Populate and update set selector
        const setSelector = document.getElementById('weather-set-selector');
        if (data.available_sets) {
            // Rebuild if first load or current set changed
            const needsRebuild = setSelector.options.length <= 1 ||
                !setSelector.querySelector(`option[value="${data.current_set}"]`)?.textContent.includes('(current)');
            if (needsRebuild) {
                setSelector.innerHTML = '<option value="">-- Select a set --</option>';
                data.available_sets.forEach(setKey => {
                    const option = document.createElement('option');
                    option.value = setKey;
                    option.textContent = WEATHER_SET_NAMES[setKey] || setKey;
                    if (setKey === data.current_set) {
                        option.textContent += ' (current)';
                        option.selected = true;
                    }
                    setSelector.appendChild(option);
                });
            }
        }

        // Sync lock buttons on first load
        if (_lastLocked === null) {
            _lastLocked = data.state_switch_locked;
            updateLockButton(data.state_switch_locked);
            _weatherStateLocked = data.weather_state_locked;
            updateStateLockButton(data.weather_state_locked);
        }

        // Repopulate state dropdown when set or lock changes
        const locked = data.state_switch_locked;
        if (data.current_set !== _lastWeatherSet || locked !== _lastLocked) {
            _lastWeatherSet = data.current_set;
            _lastLocked = locked;
            updateLockButton(locked);
            const states = locked ? data.available_weather_states : data.all_weather_states;
            if (states) populateStateSelector(states);
            _lastEventList = null;  // force event repopulate on set change
        }

        // Populate event selector based on lock state
        const eventList = _eventLocked
            ? (data.random_events || [])
            : (data.available_events || []);
        const eventKey = JSON.stringify(eventList) + _eventLocked;
        if (eventKey !== _lastEventList) {
            _lastEventList = eventKey;
            populateEventSelector(eventList);
        }
    } catch (e) {
        console.error('Failed to load weather set info:', e);
    }
}

// ---- Event listeners ----
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('weather-set-selector').addEventListener('change', (e) => {
        const newSet = e.target.value;
        if (!newSet) return;
        if (socket && socket.connected) {
            socket.emit('change_weather_set', { set_name: newSet });
        }
    });

    document.getElementById('weather-state-selector').addEventListener('change', (e) => {
        const newState = e.target.value;
        if (!newState) return;
        if (socket && socket.connected) {
            socket.emit('change_weather_state', { state_name: newState });
        }
    });
});

let _eventLocked = true;
let _lastEventList = null;

function toggleEventLock() {
    _eventLocked = !_eventLocked;
    updateEventLockButton(_eventLocked);
    _lastEventList = null;  // force repopulate
    loadWeatherSetInfo();   // repopulate immediately
}

function updateEventLockButton(locked) {
    const btn = document.getElementById('event-lock-btn');
    if (locked) {
        btn.textContent = 'Locked';
        btn.style.background = 'rgba(255, 100, 100, 0.2)';
        btn.style.borderColor = 'rgba(255, 100, 100, 0.6)';
        btn.style.color = 'rgba(255, 180, 180, 1)';
    } else {
        btn.textContent = 'Unlocked';
        btn.style.background = 'rgba(100, 255, 100, 0.2)';
        btn.style.borderColor = 'rgba(100, 255, 100, 0.6)';
        btn.style.color = 'rgba(180, 255, 180, 1)';
    }
}

function populateEventSelector(events) {
    const sel = document.getElementById('event-selector');
    sel.innerHTML = '<option value="">-- Select --</option>';
    events.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name.replace(/_/g, ' ');
        sel.appendChild(opt);
    });
}

function triggerSelectedEvent() {
    const selector = document.getElementById('event-selector');
    const eventName = selector.value;
    if (!eventName) return;
    if (socket && socket.connected) {
        socket.emit('trigger_random_event', { event_name: eventName });
    }
}

// Check admin availability
async function checkAdminAvailability() {
    try {
        const response = await fetch('/api/admin/check');
        const data = await response.json();
        if (!data.admin_enabled) {
            const adminTab = document.getElementById('admin-tab');
            if (adminTab) adminTab.style.display = 'none';
        }
    } catch (e) {
        const adminTab = document.getElementById('admin-tab');
        if (adminTab) adminTab.style.display = 'none';
    }
}

checkAdminAvailability();
init();
