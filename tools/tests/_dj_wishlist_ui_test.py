"""Offscreen gate for wishlist multi-select removal in the Discover tab.

Builds a real DiscoverTab against a temp music dir (so the live wishlist is
never touched), then drives selection + the two-step confirm.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.stdout.reconfigure(encoding="utf-8")

from PyQt6.QtCore import Qt                              # noqa: E402
from PyQt6.QtWidgets import QApplication                 # noqa: E402

from lib.dj import beatport as BP                        # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{(' — ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(name)


def main():
    tmp = tempfile.mkdtemp(prefix="wishtest_")
    items = [{"bp_id": 1000 + i, "title": f"Track {i}", "artist": f"Art {i}",
              "bpm": 120.0 + i, "camelot": "8A", "genre": "House",
              "price": "$1.49", "url": f"https://beatport.com/track/t{i}/{1000+i}",
              "preview": None} for i in range(10)]
    json.dump(items, open(os.path.join(tmp, "beatport_wishlist.json"), "w",
                          encoding="utf-8"))

    app = QApplication.instance() or QApplication([])

    # --- Wishlist.remove_many (pure, no UI) ---
    w = BP.Wishlist(tmp)
    check("wishlist loads", len(w.items) == 10, f"{len(w.items)} items")
    n = w.remove_many([1001, 1003, 1005])
    check("remove_many removes exactly the ids", n == 3 and len(w.items) == 7)
    check("remove_many persists", len(BP.Wishlist(tmp).items) == 7)
    check("remove_many no-op on empty", BP.Wishlist(tmp).remove_many([]) == 0)
    check("remove_many ignores unknown ids",
          BP.Wishlist(tmp).remove_many([999999]) == 0)

    # restore the full 10 for the UI test
    json.dump(items, open(os.path.join(tmp, "beatport_wishlist.json"), "w",
                          encoding="utf-8"))

    # --- the real DiscoverTab ---
    from tools.dj.planner.discover import DiscoverTab

    class FakePlanner:
        music_dir = tmp
        library = []
        entries = []

    tab = DiscoverTab(FakePlanner())
    wl = tab.wish_list
    check("list populated", wl.count() == 10, f"{wl.count()} rows")
    check("selection mode is ExtendedSelection",
          wl.selectionMode().name == "ExtendedSelection",
          wl.selectionMode().name)
    check("bp_id stored on items (not row index)",
          wl.item(0).data(Qt.ItemDataRole.UserRole) == 1000)

    # nothing selected -> Remove disabled, no accidental wipe
    check("remove disabled with no selection", not tab.wish_rm_btn.isEnabled())
    tab._wish_remove()
    check("remove with empty selection is a no-op", len(tab.wishlist.items) == 10)

    # select 3 non-contiguous rows
    for r in (0, 4, 9):
        wl.item(r).setSelected(True)
    check("3 rows selected", len(wl.selectedItems()) == 3)
    check("button shows count", tab.wish_rm_btn.text() == "Remove (3)",
          tab.wish_rm_btn.text())
    check("label shows selection", "(3 selected)" in tab.wish_lbl.text(),
          tab.wish_lbl.text())

    # first click ARMS only
    tab._wish_remove()
    check("first click does not delete", len(tab.wishlist.items) == 10)
    check("button armed", "Confirm remove 3?" == tab.wish_rm_btn.text(),
          tab.wish_rm_btn.text())

    # second click commits
    tab._wish_remove()
    check("second click removed exactly 3", len(tab.wishlist.items) == 7,
          f"{len(tab.wishlist.items)} left")
    left = {it["bp_id"] for it in tab.wishlist.items}
    check("removed the SELECTED ids", left == {1001, 1002, 1003, 1005, 1006,
                                               1007, 1008})
    check("list rebuilt", wl.count() == 7)
    check("persisted to disk", len(BP.Wishlist(tmp).items) == 7)
    check("disarmed after commit", tab._wish_armed is None)

    # selection change disarms a pending confirm
    wl.item(0).setSelected(True)
    tab._wish_remove()
    check("armed again", tab.wish_rm_btn.text().startswith("Confirm"))
    wl.item(1).setSelected(True)          # changing selection must disarm
    check("selection change disarms",
          tab._wish_armed is None and not tab.wish_rm_btn.text().startswith(
              "Confirm"), tab.wish_rm_btn.text())
    tab._wish_remove()                    # now only arms for the NEW set
    check("re-arm after change does not delete", len(tab.wishlist.items) == 7)

    # ctrl+A path: select all, remove everything
    wl.selectAll()
    check("select all", len(wl.selectedItems()) == 7)
    tab._wish_remove()
    tab._wish_remove()
    check("bulk remove all", len(tab.wishlist.items) == 0 and wl.count() == 0)

    # open guard: >8 needs confirm, and never opens a browser here
    opened = []
    tab.wishlist.items = list(items)
    tab._refresh_wishlist()
    tab.wishlist.open_in_browser = lambda b=None: opened.append(b)
    tab._wish_open()
    check("open all (10) arms instead of opening", not opened,
          f"opened={len(opened)}")
    tab._wish_open()
    check("second click opens all 10", len(opened) == 10, f"{len(opened)}")

    opened.clear()
    wl.item(2).setSelected(True)
    tab._wish_open()
    check("small selection opens immediately", opened == [1002], str(opened))

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
