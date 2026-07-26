/**
 * GL_Simple — Interaction tab visibility.
 *
 * One nav slot, owned by whichever weather set is live. The server says
 * whether the live set publishes a panel, what to call it, and where it
 * points (the club set points at the bespoke /dj page; declarative sets
 * point at /interaction). Sets that publish nothing — the default — get
 * no tab at all.
 *
 * Cheap poll, same idiom as club_tab.js. A page that renders this tab
 * already active (the DJ page, the interaction page) keeps it visible
 * even when the panel goes away, so the tab you're standing on never
 * vanishes under you.
 */
(function () {
    const tab = document.getElementById('interaction-tab');
    if (!tab) return;

    const pinned = tab.classList.contains('active');

    async function check() {
        try {
            const r = await fetch('/api/interaction/info');
            const j = await r.json();
            if (j.available) {
                if (tab.textContent !== j.label) tab.textContent = j.label;
                if (!pinned) tab.href = j.href;
                tab.style.display = '';
                tab.style.color = j.live ? '#8f8' : '';
            } else {
                tab.style.display = pinned ? '' : 'none';
                tab.style.color = '';
            }
        } catch (e) { /* keep current state on transient errors */ }
    }

    check();
    setInterval(check, 4000);
})();
