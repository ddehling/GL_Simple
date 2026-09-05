/**
 * GL_Simple — Generative (Gen) tab visibility.
 * Shown whenever the generative subsystem is available (config gen.enabled),
 * green while it is playing. Cheap poll, same idiom as club_tab.js.
 */
(function () {
    const tab = document.getElementById('gen-tab');
    if (!tab) return;
    const pinned = tab.classList.contains('active');

    async function check() {
        try {
            const r = await fetch('/api/gen/active');
            const j = await r.json();
            tab.style.display = (j.available || pinned) ? '' : 'none';
            tab.style.color = j.active ? '#8f8' : '';
        } catch (e) { /* keep current state on transient errors */ }
    }

    check();
    setInterval(check, 4000);
})();
