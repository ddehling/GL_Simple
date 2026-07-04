/**
 * GL_Simple — DJ tab visibility.
 * The DJ tab shows whenever the autonomous DJ is AVAILABLE (config-enabled),
 * lit green while it's actually playing. Cheap poll, same idiom as club_tab.
 */
(function () {
    const tab = document.getElementById('dj-tab');
    if (!tab) return;

    async function check() {
        try {
            const r = await fetch('/api/dj/active');
            const j = await r.json();
            tab.style.display = j.available ? '' : 'none';
            tab.style.color = j.active ? '#8f8' : '';
        } catch (e) { /* keep current state on transient errors */ }
    }

    check();
    setInterval(check, 4000);
})();
