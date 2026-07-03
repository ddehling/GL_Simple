/**
 * GL_Simple — Club tab visibility.
 * The Club page only matters while the club weather set is live, so its
 * nav tab stays hidden everywhere else. Cheap poll; no socket needed.
 */
(function () {
    const tab = document.getElementById('club-tab');
    if (!tab) return;

    async function check() {
        try {
            const r = await fetch('/api/club/active');
            const j = await r.json();
            tab.style.display = j.active ? '' : 'none';
        } catch (e) { /* keep current state on transient errors */ }
    }

    check();
    setInterval(check, 4000);
})();
