/** One way out: every widget sends {action, value} through gen_action.
 *  The server whitelists and clamps (lib/gen/actions.py); nothing here
 *  names an engine, a synth or a composer. */
export const socket = createSocket();     // from /static/js/socket-client.js (classic script)
export function emit(action, value) { socket.emit('gen_action', { action, value }); }

/** Sliders held by the operator must not be yanked by state updates. */
export const dragging = new Set();

export function throttled(ms) {
    let last = 0;
    return (fn) => { const now = Date.now(); if (now - last > ms) { last = now; fn(); } };
}
