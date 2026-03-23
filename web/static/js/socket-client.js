/**
 * GL_Simple — WebSocket connection manager
 * Shared across all pages that need real-time updates.
 */

function createSocket() {
    const socket = io({
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000
    });

    socket.on('connect', () => {
        const statusText = document.getElementById('status-text');
        const dot = document.querySelector('.status-dot');
        if (statusText) statusText.textContent = 'Connected';
        if (dot) dot.style.background = '#0f0';
    });

    socket.on('disconnect', () => {
        const statusText = document.getElementById('status-text');
        const dot = document.querySelector('.status-dot');
        if (statusText) statusText.textContent = 'Disconnected';
        if (dot) dot.style.background = '#f00';
    });

    socket.on('reconnecting', () => {
        const statusText = document.getElementById('status-text');
        const dot = document.querySelector('.status-dot');
        if (statusText) statusText.textContent = 'Reconnecting...';
        if (dot) dot.style.background = '#ff0';
    });

    return socket;
}
