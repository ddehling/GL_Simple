// Weather Editor JavaScript
let weatherData = {
    weather_states: [],
    weather_presets: {},
    weather_sets: {}
};

let currentSet = null;
let currentWeatherTab = null;
let hasUnsavedChanges = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadWeatherData();
    setupEventListeners();
});

function setupEventListeners() {
    document.getElementById('addSetBtn').addEventListener('click', showNewSetModal);
    document.getElementById('saveBtn').addEventListener('click', saveChanges);
    document.getElementById('validateBtn').addEventListener('click', validateData);
    
    // Warn before leaving with unsaved changes
    window.addEventListener('beforeunload', (e) => {
        if (hasUnsavedChanges) {
            e.preventDefault();
            e.returnValue = '';
        }
    });
}

async function loadWeatherData() {
    showLoading(true);
    try {
        const response = await fetch('/api/weather_editor/all_data');
        if (!response.ok) throw new Error('Failed to load weather data');
        
        weatherData = await response.json();
        renderSetList();
        showStatus('Weather data loaded successfully', 'success');
    } catch (error) {
        showStatus('Error loading weather data: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function renderSetList() {
    const setList = document.getElementById('setList');
    setList.innerHTML = '';
    
    for (const [setId, setData] of Object.entries(weatherData.weather_sets)) {
        const li = document.createElement('li');
        li.className = 'set-item';
        if (currentSet === setId) {
            li.classList.add('active');
        }
        
        li.innerHTML = `
            <div class="set-item-header">
                <span>${setData.name || setId}</span>
                <button class="set-delete-btn" onclick="deleteSet('${setId}', event)">×</button>
            </div>
        `;
        
        li.onclick = (e) => {
            if (!e.target.classList.contains('set-delete-btn')) {
                selectSet(setId);
            }
        };
        
        setList.appendChild(li);
    }
}

function selectSet(setId) {
    currentSet = setId;
    renderSetList();
    renderSetEditor();
}

function renderSetEditor() {
    if (!currentSet) return;
    
    const setData = weatherData.weather_sets[currentSet];
    const editorContent = document.getElementById('editorContent');
    
    editorContent.innerHTML = `
        <div class="editor-section">
            <div class="section-header">
                <div class="section-title">Set Configuration: ${setData.name}</div>
            </div>
            
            <div class="set-config-grid">
                <div class="form-group">
                    <label>Display Name</label>
                    <input type="text" id="setName" value="${setData.name || ''}" 
                           onchange="updateSetProperty('name', this.value)">
                </div>
                
                <div class="form-group">
                    <label>Season Speed</label>
                    <input type="number" step="0.1" id="setSeasonSpeed" 
                           value="${setData.season_speed || 1.0}"
                           onchange="updateSetProperty('season_speed', parseFloat(this.value))">
                </div>
                
                <div class="form-group">
                    <label>Season Extremity</label>
                    <input type="number" step="0.1" id="setSeasonExtremity" 
                           value="${setData.season_extremity || 1.0}"
                           onchange="updateSetProperty('season_extremity', parseFloat(this.value))">
                </div>
                
                <div class="form-group">
                    <label>Transition Speed</label>
                    <input type="number" step="0.1" id="setTransitionSpeed" 
                           value="${setData.transition_speed || 1.0}"
                           onchange="updateSetProperty('transition_speed', parseFloat(this.value))">
                </div>
            </div>
            
            <div class="form-group">
                <label>Description</label>
                <textarea id="setDescription" 
                          onchange="updateSetProperty('description', this.value)">${setData.description || ''}</textarea>
            </div>
        </div>
        
        <div class="editor-section">
            <div class="section-header">
                <div class="section-title">Weather States in Set</div>
                <button class="btn btn-primary btn-sm" onclick="toggleWeatherDropdown()">+ Add Weather</button>
            </div>
            
            <div class="weather-states-container">
                <div class="weather-pills" id="weatherPills"></div>
                
                <div class="add-weather-dropdown">
                    <div class="dropdown-content" id="weatherDropdown"></div>
                </div>
            </div>
        </div>
        
        <div class="editor-section">
            <div class="section-header">
                <div class="section-title">Edit Weather State Parameters</div>
                <button class="btn btn-primary add-state-btn" onclick="showNewStateModal()">+ New Weather State</button>
            </div>
            
            <div class="weather-state-tabs" id="weatherTabs"></div>
            <div id="weatherTabsContent"></div>
        </div>
    `;
    
    renderWeatherPills();
    renderWeatherTabs();
}

function renderWeatherPills() {
    const setData = weatherData.weather_sets[currentSet];
    const weatherPills = document.getElementById('weatherPills');
    
    weatherPills.innerHTML = '';
    
    (setData.states || []).forEach(state => {
        const pill = document.createElement('div');
        pill.className = 'weather-pill';
        pill.innerHTML = `
            <span>${state}</span>
            <button class="weather-pill-remove" onclick="removeWeatherFromSet('${state}')">×</button>
        `;
        weatherPills.appendChild(pill);
    });
}

function renderWeatherDropdown() {
    const setData = weatherData.weather_sets[currentSet];
    const dropdown = document.getElementById('weatherDropdown');
    
    dropdown.innerHTML = '';
    
    weatherData.weather_states.forEach(state => {
        const isInSet = setData.states && setData.states.includes(state);
        const item = document.createElement('div');
        item.className = 'dropdown-item' + (isInSet ? ' disabled' : '');
        item.textContent = state;
        
        if (!isInSet) {
            item.onclick = () => addWeatherToSet(state);
        }
        
        dropdown.appendChild(item);
    });
}

function toggleWeatherDropdown() {
    const dropdown = document.getElementById('weatherDropdown');
    renderWeatherDropdown();
    dropdown.classList.toggle('show');
    
    // Close dropdown when clicking outside
    setTimeout(() => {
        document.addEventListener('click', function closeDropdown(e) {
            if (!e.target.closest('.add-weather-dropdown')) {
                dropdown.classList.remove('show');
                document.removeEventListener('click', closeDropdown);
            }
        });
    }, 0);
}

function addWeatherToSet(state) {
    const setData = weatherData.weather_sets[currentSet];
    if (!setData.states) {
        setData.states = [];
    }
    
    if (!setData.states.includes(state)) {
        setData.states.push(state);
        hasUnsavedChanges = true;
        renderWeatherPills();
        renderWeatherTabs();
        document.getElementById('weatherDropdown').classList.remove('show');
        showStatus(`Added ${state} to set`, 'success', 2000);
    }
}

function removeWeatherFromSet(state) {
    const setData = weatherData.weather_sets[currentSet];
    if (setData.states) {
        setData.states = setData.states.filter(s => s !== state);
        hasUnsavedChanges = true;
        renderWeatherPills();
        renderWeatherTabs();
        showStatus(`Removed ${state} from set`, 'success', 2000);
    }
}

function renderWeatherTabs() {
    const setData = weatherData.weather_sets[currentSet];
    const tabsContainer = document.getElementById('weatherTabs');
    const tabsContent = document.getElementById('weatherTabsContent');
    
    tabsContainer.innerHTML = '';
    tabsContent.innerHTML = '';
    
    if (!setData.states || setData.states.length === 0) {
        tabsContent.innerHTML = '<p style="text-align: center; color: #999; padding: 30px;">No weather states in this set. Add some above!</p>';
        return;
    }
    
    setData.states.forEach((state, index) => {
        // Create tab button
        const tab = document.createElement('button');
        tab.className = 'weather-tab';
        tab.textContent = state;
        tab.onclick = () => selectWeatherTab(state);
        
        if (index === 0 && !currentWeatherTab) {
            currentWeatherTab = state;
            tab.classList.add('active');
        } else if (currentWeatherTab === state) {
            tab.classList.add('active');
        }
        
        tabsContainer.appendChild(tab);
        
        // Create tab content
        const tabContent = document.createElement('div');
        tabContent.className = 'weather-tab-content';
        tabContent.id = `tab-${state}`;
        
        if (index === 0 && !currentWeatherTab || currentWeatherTab === state) {
            tabContent.classList.add('active');
        }
        
        tabContent.innerHTML = renderWeatherStateEditor(state);
        tabsContent.appendChild(tabContent);
    });
}

function selectWeatherTab(state) {
    currentWeatherTab = state;
    
    // Update tab buttons
    document.querySelectorAll('.weather-tab').forEach(tab => {
        tab.classList.remove('active');
        if (tab.textContent === state) {
            tab.classList.add('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.weather-tab-content').forEach(content => {
        content.classList.remove('active');
        if (content.id === `tab-${state}`) {
            content.classList.add('active');
        }
    });
}

function renderWeatherStateEditor(state) {
    const preset = weatherData.weather_presets[state] || {};
    
    // Define all possible parameters with types
    const paramDefs = {
        wind_speed: { type: 'number', step: 0.1 },
        rain_rate: { type: 'number', step: 0.1 },
        lightning_probability: { type: 'number', step: 0.05 },
        starryness: { type: 'number', step: 0.1 },
        spookyness: { type: 'number', step: 0.1 },
        fog: { type: 'number', step: 0.05 },
        fog_color: { type: 'array', length: 3 },
        celestial_visibility: { type: 'number', step: 0.1 },
        firefly_density: { type: 'number', step: 0.1 },
        Aurora_probability: { type: 'number', step: 0.1 },
        Wolfy: { type: 'number', step: 0.1 },
        Switch_rate: { type: 'number', step: 0.1 },
        meteor_rate: { type: 'number', step: 0.05 },
        volcano_level: { type: 'number', step: 0.1 },
        sand_density: { type: 'number', step: 0.1 },
        skiptime: { type: 'number', step: 0.5 },
        tree_prob: { type: 'number', step: 0.1 },
        Weird: { type: 'number', step: 0.1 },
        Sound_volume: { type: 'number', step: 0.1 },
        season_preference: { type: 'number', step: 0.025 },
        mountain: { type: 'number', step: 0.1 },
        ambient_sound: { type: 'text' },
        ARI: { type: 'number', step: 1 },
        transition_duration: { type: 'number', step: 1 },
        possible_transitions: { type: 'array-string' },
        transition_weights: { type: 'array-number' }
    };
    
    let html = '<div class="param-grid">';
    
    // Render all parameters
    for (const [paramName, paramDef] of Object.entries(paramDefs)) {
        const value = preset[paramName];
        
        if (paramDef.type === 'number') {
            html += `
                <div class="param-item">
                    <label>${paramName}</label>
                    <input type="number" step="${paramDef.step}" 
                           value="${value !== undefined ? value : ''}"
                           onchange="updateWeatherParam('${state}', '${paramName}', parseFloat(this.value))">
                </div>
            `;
        } else if (paramDef.type === 'text') {
            html += `
                <div class="param-item">
                    <label>${paramName}</label>
                    <input type="text" 
                           value="${value !== undefined ? value : ''}"
                           onchange="updateWeatherParam('${state}', '${paramName}', this.value)">
                </div>
            `;
        } else if (paramDef.type === 'array') {
            const arrayValue = value || [0, 0, 0];
            html += `
                <div class="param-item">
                    <label>${paramName}</label>
                    <div class="array-input">
                        ${arrayValue.map((v, i) => `
                            <input type="number" step="0.1" value="${v}" 
                                   onchange="updateWeatherArrayParam('${state}', '${paramName}', ${i}, parseFloat(this.value))">
                        `).join('')}
                    </div>
                </div>
            `;
        } else if (paramDef.type === 'array-string') {
            const arrayValue = value || [];
            html += `
                <div class="param-item" style="grid-column: span 2;">
                    <label>${paramName}</label>
                    <input type="text" 
                           value="${arrayValue.join(', ')}"
                           placeholder="comma-separated values"
                           onchange="updateWeatherArrayStringParam('${state}', '${paramName}', this.value)">
                </div>
            `;
        } else if (paramDef.type === 'array-number') {
            const arrayValue = value || [];
            html += `
                <div class="param-item" style="grid-column: span 2;">
                    <label>${paramName}</label>
                    <input type="text" 
                           value="${arrayValue.join(', ')}"
                           placeholder="comma-separated numbers"
                           onchange="updateWeatherArrayNumberParam('${state}', '${paramName}', this.value)">
                </div>
            `;
        }
    }
    
    html += '</div>';
    
    html += `
        <div style="margin-top: 20px;">
            <button class="btn btn-danger" onclick="deleteWeatherState('${state}')">
                Delete Weather State: ${state}
            </button>
        </div>
    `;
    
    return html;
}

function updateSetProperty(property, value) {
    if (!currentSet) return;
    weatherData.weather_sets[currentSet][property] = value;
    hasUnsavedChanges = true;
}

function updateWeatherParam(state, param, value) {
    if (!weatherData.weather_presets[state]) {
        weatherData.weather_presets[state] = {};
    }
    weatherData.weather_presets[state][param] = value;
    hasUnsavedChanges = true;
}

function updateWeatherArrayParam(state, param, index, value) {
    if (!weatherData.weather_presets[state]) {
        weatherData.weather_presets[state] = {};
    }
    if (!weatherData.weather_presets[state][param]) {
        weatherData.weather_presets[state][param] = [0, 0, 0];
    }
    weatherData.weather_presets[state][param][index] = value;
    hasUnsavedChanges = true;
}

function updateWeatherArrayStringParam(state, param, value) {
    if (!weatherData.weather_presets[state]) {
        weatherData.weather_presets[state] = {};
    }
    weatherData.weather_presets[state][param] = value.split(',').map(s => s.trim()).filter(s => s);
    hasUnsavedChanges = true;
}

function updateWeatherArrayNumberParam(state, param, value) {
    if (!weatherData.weather_presets[state]) {
        weatherData.weather_presets[state] = {};
    }
    weatherData.weather_presets[state][param] = value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    hasUnsavedChanges = true;
}

function showNewSetModal() {
    document.getElementById('newSetModal').classList.add('show');
    document.getElementById('newSetId').value = '';
    document.getElementById('newSetName').value = '';
    document.getElementById('newSetDescription').value = '';
}

function closeNewSetModal() {
    document.getElementById('newSetModal').classList.remove('show');
}

function createNewSet() {
    const setId = document.getElementById('newSetId').value.trim();
    const setName = document.getElementById('newSetName').value.trim();
    const setDescription = document.getElementById('newSetDescription').value.trim();
    
    if (!setId) {
        alert('Set ID is required');
        return;
    }
    
    if (!/^[a-z0-9_]+$/.test(setId)) {
        alert('Set ID must be lowercase letters, numbers, and underscores only');
        return;
    }
    
    if (weatherData.weather_sets[setId]) {
        alert('A set with this ID already exists');
        return;
    }
    
    weatherData.weather_sets[setId] = {
        name: setName || setId,
        description: setDescription || 'Custom weather set',
        states: [],
        season_speed: 1.0,
        season_extremity: 1.0,
        transition_speed: 1.0
    };
    
    hasUnsavedChanges = true;
    closeNewSetModal();
    renderSetList();
    selectSet(setId);
    showStatus(`Created new set: ${setName || setId}`, 'success');
}

function deleteSet(setId, event) {
    event.stopPropagation();
    
    if (!confirm(`Are you sure you want to delete the set "${weatherData.weather_sets[setId].name}"?`)) {
        return;
    }
    
    delete weatherData.weather_sets[setId];
    hasUnsavedChanges = true;
    
    if (currentSet === setId) {
        currentSet = null;
        document.getElementById('editorContent').innerHTML = '<p style="text-align: center; color: #999; padding: 50px;">Select a set to edit or create a new one</p>';
    }
    
    renderSetList();
    showStatus('Set deleted', 'success');
}

function showNewStateModal() {
    document.getElementById('newStateModal').classList.add('show');
    document.getElementById('newStateName').value = '';
    document.getElementById('newStateValue').value = '';
}

function closeNewStateModal() {
    document.getElementById('newStateModal').classList.remove('show');
}

function createNewState() {
    const stateName = document.getElementById('newStateName').value.trim();
    const stateValue = document.getElementById('newStateValue').value.trim();
    
    if (!stateName || !stateValue) {
        alert('Both state name and value are required');
        return;
    }
    
    if (!/^[A-Z0-9_]+$/.test(stateName)) {
        alert('State name must be UPPERCASE_SNAKE_CASE');
        return;
    }
    
    if (!/^[a-z0-9_]+$/.test(stateValue)) {
        alert('State value must be lowercase');
        return;
    }
    
    if (weatherData.weather_states.includes(stateValue)) {
        alert('A state with this value already exists');
        return;
    }
    
    weatherData.weather_states.push(stateValue);
    weatherData.weather_presets[stateValue] = {
        wind_speed: 0.0,
        rain_rate: 0.0,
        possible_transitions: ["clear"],
        transition_weights: [1.0]
    };
    
    hasUnsavedChanges = true;
    closeNewStateModal();
    showStatus(`Created new weather state: ${stateValue}`, 'success');
    
    if (currentSet) {
        renderSetEditor();
    }
}

function deleteWeatherState(state) {
    if (!confirm(`Are you sure you want to delete the weather state "${state}"? This will remove it from all sets and delete all its parameters.`)) {
        return;
    }
    
    // Remove from weather_states array
    weatherData.weather_states = weatherData.weather_states.filter(s => s !== state);
    
    // Remove from presets
    delete weatherData.weather_presets[state];
    
    // Remove from all sets
    for (const setData of Object.values(weatherData.weather_sets)) {
        if (setData.states) {
            setData.states = setData.states.filter(s => s !== state);
        }
    }
    
    hasUnsavedChanges = true;
    showStatus(`Deleted weather state: ${state}`, 'success');
    
    if (currentSet) {
        renderSetEditor();
    }
}

async function validateData() {
    showLoading(true);
    try {
        const response = await fetch('/api/weather_editor/validate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(weatherData)
        });
        
        const result = await response.json();
        
        if (result.valid) {
            showStatus('✓ All data is valid!', 'success');
        } else {
            showStatus('❌ Validation errors: ' + result.errors.join(', '), 'error');
        }
    } catch (error) {
        showStatus('Error validating data: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

async function saveChanges() {
    if (!confirm('This will overwrite weather_params.py. Make sure you have a backup! Continue?')) {
        return;
    }
    
    showLoading(true);
    try {
        const response = await fetch('/api/weather_editor/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(weatherData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            hasUnsavedChanges = false;
            showStatus('✓ Changes saved successfully! Restart the application to see changes.', 'success');
        } else {
            showStatus('❌ Error saving: ' + result.error, 'error');
        }
    } catch (error) {
        showStatus('Error saving changes: ' + error.message, 'error');
    } finally {
        showLoading(false);
    }
}

function showStatus(message, type, timeout = 5000) {
    const statusEl = document.getElementById('statusMessage');
    statusEl.textContent = message;
    statusEl.className = 'status-message show ' + type;
    
    if (timeout > 0) {
        setTimeout(() => {
            statusEl.classList.remove('show');
        }, timeout);
    }
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('show');
    } else {
        overlay.classList.remove('show');
    }
}
