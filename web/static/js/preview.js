/**
 * GL_Simple — WebGL2 live preview renderer
 *
 * Receives PNG frames via Socket.IO and renders them in four display modes:
 *   flat-smooth, fan-smooth, flat-led, fan-led
 *
 * Geometry data (fan mesh, dot positions) is fetched once from the server
 * and used to build WebGL buffers.
 */

(function () {
    "use strict";

    // ---- Weather set display names (kept in sync with control_panel.js) ----
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

    function titleCase(s) {
        return (s || "").replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    }

    // ---- State ----
    let gl, canvas;
    let mode = "flat-smooth";
    let geometry = null;      // from /api/preview/geometry
    let frameTexture = null;  // WebGL texture updated each frame

    // GL resources per mode
    const programs = {};
    const vaos = {};
    const buffers = {};
    let flatQuadVAO = null;

    // Pan/zoom state (fan modes only)
    let zoom = 1.0;
    let panX = 0.0;
    let panY = 0.0;
    let dragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    // ---- WebGL init ----

    function initGL() {
        canvas = document.getElementById("preview-canvas");
        gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
        if (!gl) {
            console.error("WebGL2 not supported");
            return false;
        }

        frameTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, frameTexture);
        // 1x1 black pixel placeholder
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA,
            gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 255]));
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        return true;
    }

    // ---- Shader helpers ----

    function compileShader(src, type) {
        const s = gl.createShader(type);
        gl.shaderSource(s, src);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
            console.error("Shader compile error:", gl.getShaderInfoLog(s));
            return null;
        }
        return s;
    }

    function linkProgram(vsSrc, fsSrc) {
        const vs = compileShader(vsSrc, gl.VERTEX_SHADER);
        const fs = compileShader(fsSrc, gl.FRAGMENT_SHADER);
        const p = gl.createProgram();
        gl.attachShader(p, vs);
        gl.attachShader(p, fs);
        gl.linkProgram(p);
        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
            console.error("Program link error:", gl.getProgramInfoLog(p));
            return null;
        }
        return p;
    }

    // ---- Shaders ----

    const FLAT_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aTexCoord;
out vec2 vUV;
uniform vec2 uAspect;
uniform float uZoom;
uniform vec2 uPan;
void main() {
    vec2 pos = aPosition * uAspect * uZoom + uPan;
    gl_Position = vec4(pos, 0.0, 1.0);
    vUV = aTexCoord;
}`;

    const FLAT_FRAG = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uTexture;
out vec4 outColor;
void main() {
    outColor = texture(uTexture, vec2(vUV.x, 1.0 - vUV.y));
}`;

    // Fan continuous — same shaders, different mesh
    const FAN_VERT = FLAT_VERT;
    const FAN_FRAG = FLAT_FRAG;

    const DOT_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aQuad;
layout(location = 1) in vec2 iPos;
layout(location = 2) in vec2 iTexCoord;
out vec2 vQuad;
flat out vec2 vUV;
uniform float uRadius;
uniform vec2 uAspect;
uniform float uZoom;
uniform vec2 uPan;
void main() {
    vec2 world = iPos + aQuad * uRadius;
    gl_Position = vec4(world * uAspect * uZoom + uPan, 0.0, 1.0);
    vQuad = aQuad;
    vUV = iTexCoord;
}`;

    const DOT_FRAG = `#version 300 es
precision highp float;
in vec2 vQuad;
flat in vec2 vUV;
uniform sampler2D uTexture;
out vec4 outColor;
void main() {
    float d = length(vQuad);
    if (d > 1.0) discard;
    vec4 color = texture(uTexture, vec2(vUV.x, 1.0 - vUV.y));
    outColor = color;
}`;

    // ---- Build GL resources from geometry ----

    function buildFlatQuad() {
        // Full-screen quad: position + UV
        const data = new Float32Array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
             1,  1, 1, 1,
            -1, -1, 0, 0,
             1,  1, 1, 1,
            -1,  1, 0, 1,
        ]);
        const prog = linkProgram(FLAT_VERT, FLAT_FRAG);
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);
        gl.bindVertexArray(null);

        programs["flat-smooth"] = prog;
        vaos["flat-smooth"] = vao;
        buffers["flat-smooth"] = { count: 6 };
    }

    function buildFanMesh(geo) {
        const verts = new Float32Array(geo.fan_mesh.verts);
        const indices = new Uint32Array(geo.fan_mesh.indices);

        const prog = linkProgram(FAN_VERT, FAN_FRAG);
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 16, 0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 8);

        const ebo = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

        gl.bindVertexArray(null);

        programs["fan-smooth"] = prog;
        vaos["fan-smooth"] = vao;
        buffers["fan-smooth"] = { indexCount: indices.length };
    }

    function buildDots(geo, modeName, instanceData, dotRadius) {
        const instances = new Float32Array(instanceData);
        const nInstances = instances.length / 4;

        const prog = programs["_dots"] || linkProgram(DOT_VERT, DOT_FRAG);
        programs["_dots"] = prog;  // share across flat/fan dots

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        // Unit quad
        const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
        const qvbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, qvbo);
        gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

        // Instance data
        const ivbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, ivbo);
        gl.bufferData(gl.ARRAY_BUFFER, instances, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 16, 0);
        gl.vertexAttribDivisor(1, 1);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 16, 8);
        gl.vertexAttribDivisor(2, 1);

        gl.bindVertexArray(null);

        programs[modeName] = prog;
        vaos[modeName] = vao;
        buffers[modeName] = { nInstances, dotRadius };
    }

    function buildAllGeometry(geo) {
        buildFlatQuad();
        buildFanMesh(geo);
        buildDots(geo, "flat-led", geo.flat_dots.instances, geo.flat_dots.dot_radius);
        buildDots(geo, "fan-led", geo.fan_dots.instances, geo.fan_dots.dot_radius);
    }

    // ---- Render ----

    function render() {
        if (!geometry) return;

        // Set canvas size based on display mode and available space
        const isFan = mode.startsWith("fan");
        const aspect = isFan ? (900 / 500) : (geometry.num_strips / geometry.num_leds);
        const wrapper = canvas.parentElement;
        const maxW = wrapper.clientWidth;
        const maxH = window.innerHeight * 0.8;

        let cssW, cssH;
        if (aspect >= 1) {
            // Landscape: fill width, constrain height
            cssW = maxW;
            cssH = maxW / aspect;
            if (cssH > maxH) { cssH = maxH; cssW = maxH * aspect; }
        } else {
            // Portrait: fill height, constrain width
            cssH = maxH;
            cssW = maxH * aspect;
            if (cssW > maxW) { cssW = maxW; cssH = maxW / aspect; }
        }
        canvas.style.width = cssW + "px";
        canvas.style.height = cssH + "px";

        // Resize canvas backing store to match CSS layout
        const dpr = window.devicePixelRatio || 1;
        const w = Math.round(cssW * dpr);
        const h = Math.round(cssH * dpr);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }

        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(0.02, 0.02, 0.02, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, frameTexture);

        // Compute aspect correction: geometry was built for a specific
        // aspect ratio but the canvas may differ
        const canvasAspect = canvas.width / canvas.height;
        let ax = 1.0, ay = 1.0;
        if (mode.startsWith("fan")) {
            // Fan geometry was built for 900/500 = 1.8 aspect
            const geoAspect = 900 / 500;
            if (canvasAspect > geoAspect) {
                ax = geoAspect / canvasAspect;
            } else {
                ay = canvasAspect / geoAspect;
            }
        }
        // Flat modes: geometry fills [-1,1] in both axes, canvas
        // aspect matches frame aspect via CSS, so no correction needed

        // Only apply pan/zoom in fan modes
        const isFanMode = mode.startsWith("fan");
        const z = isFanMode ? zoom : 1.0;
        const px = isFanMode ? panX : 0.0;
        const py = isFanMode ? panY : 0.0;

        if (mode === "flat-smooth" || mode === "fan-smooth") {
            const prog = programs[mode];
            const vao = vaos[mode];
            const info = buffers[mode];
            gl.useProgram(prog);
            gl.uniform1i(gl.getUniformLocation(prog, "uTexture"), 0);
            gl.uniform2f(gl.getUniformLocation(prog, "uAspect"), ax, ay);
            gl.uniform1f(gl.getUniformLocation(prog, "uZoom"), z);
            gl.uniform2f(gl.getUniformLocation(prog, "uPan"), px, py);
            gl.bindVertexArray(vao);

            if (mode === "fan-smooth") {
                gl.drawElements(gl.TRIANGLES, info.indexCount, gl.UNSIGNED_INT, 0);
            } else {
                gl.drawArrays(gl.TRIANGLES, 0, info.count);
            }
        } else {
            // Dot modes
            const prog = programs[mode];
            const vao = vaos[mode];
            const info = buffers[mode];
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            gl.useProgram(prog);
            gl.uniform1i(gl.getUniformLocation(prog, "uTexture"), 0);
            gl.uniform1f(gl.getUniformLocation(prog, "uRadius"), info.dotRadius);
            gl.uniform2f(gl.getUniformLocation(prog, "uAspect"), ax, ay);
            gl.uniform1f(gl.getUniformLocation(prog, "uZoom"), z);
            gl.uniform2f(gl.getUniformLocation(prog, "uPan"), px, py);
            gl.bindVertexArray(vao);
            gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, info.nInstances);
            gl.disable(gl.BLEND);
        }

        gl.bindVertexArray(null);

        // FPS counter — updated via state_update from server (see below)

        requestAnimationFrame(render);
    }

    // ---- Frame reception ----

    function onFrame(pngBytes) {
        const blob = new Blob([pngBytes], { type: "image/png" });
        createImageBitmap(blob).then(bmp => {
            gl.bindTexture(gl.TEXTURE_2D, frameTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
            bmp.close();
        });
    }

    // ---- Mode buttons ----

    function setupModeButtons() {
        document.querySelectorAll(".mode-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                const oldMode = mode;
                mode = btn.dataset.mode;
                // Reset pan/zoom when switching between flat and fan
                if (oldMode.startsWith("fan") !== mode.startsWith("fan")) {
                    zoom = 1.0;
                    panX = 0.0;
                    panY = 0.0;
                }
            });
        });
    }

    // ---- Pan/zoom mouse controls (fan modes) ----

    function setupPanZoom() {
        // Scroll to zoom (centered on cursor)
        canvas.addEventListener("wheel", (e) => {
            if (!mode.startsWith("fan")) return;
            e.preventDefault();

            const rect = canvas.getBoundingClientRect();
            const clipX = ((e.clientX - rect.left) / rect.width) * 2.0 - 1.0;
            const clipY = 1.0 - ((e.clientY - rect.top) / rect.height) * 2.0;

            const oldZoom = zoom;
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const newZoom = Math.max(0.5, Math.min(20.0, oldZoom * factor));

            // Adjust pan so the point under cursor stays fixed
            panX = clipX - (clipX - panX) * newZoom / oldZoom;
            panY = clipY - (clipY - panY) * newZoom / oldZoom;
            zoom = newZoom;
        }, { passive: false });

        // Drag to pan
        canvas.addEventListener("mousedown", (e) => {
            if (e.button === 0) {
                dragging = true;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
                canvas.style.cursor = "grabbing";
            } else if (e.button === 1) {
                // Middle click resets
                e.preventDefault();
                zoom = 1.0;
                panX = 0.0;
                panY = 0.0;
            }
        });

        window.addEventListener("mousemove", (e) => {
            if (!dragging || !mode.startsWith("fan")) return;
            const rect = canvas.getBoundingClientRect();
            const dx = (e.clientX - lastMouseX) / rect.width * 2.0;
            const dy = -(e.clientY - lastMouseY) / rect.height * 2.0;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            panX += dx;
            panY += dy;
        });

        window.addEventListener("mouseup", (e) => {
            if (e.button === 0) {
                dragging = false;
                canvas.style.cursor = "";
            }
        });

        // Prevent context menu on middle click
        canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    }

    // ---- Bootstrap ----

    function init() {
        if (!initGL()) return;
        setupModeButtons();
        setupPanZoom();

        const socket = createSocket();

        // Fetch geometry once
        fetch("/api/preview/geometry")
            .then(r => r.json())
            .then(geo => {
                geometry = geo;
                buildAllGeometry(geo);
                requestAnimationFrame(render);
            })
            .catch(err => console.error("Geometry fetch failed:", err));

        // Subscribe to frame stream
        socket.on("connect", () => {
            socket.emit("subscribe_preview");
        });

        socket.on("frame", onFrame);

        socket.on("state_update", (data) => {
            const fpsEl = document.getElementById("preview-fps");
            if (fpsEl && data.fps != null) {
                fpsEl.textContent = data.fps + " FPS";
            }

            const setEl = document.getElementById("preview-weather-set");
            if (setEl && data.current_weather_set) {
                setEl.textContent = WEATHER_SET_NAMES[data.current_weather_set]
                    || titleCase(data.current_weather_set);
            }

            const stateEl = document.getElementById("preview-weather-state");
            if (stateEl && data.current_weather) {
                stateEl.textContent = titleCase(data.current_weather);
            }
        });
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
