"""Keyboard performance layer: number keys fire the first nine gestures,
Ctrl+K focuses the director's ask box, Ctrl+S saves a scene, and a Help
overlay lists every shortcut the console (and other plugins) registered."""
from PyQt6.QtWidgets import QInputDialog, QLineEdit, QMessageBox

from lib.gen.director import GESTURES


def register(console):
    names = list(GESTURES)[:9]
    for i, name in enumerate(names, start=1):
        console.add_shortcut(str(i), lambda n=name: console.ctx.emit("gesture", n), f"gesture: {GESTURES[name]['label']}")

    def focus_ask():
        page = console.pages.get("play")
        if page is None:
            return
        console.tabs.setCurrentWidget(page)
        for w in page.findChildren(QLineEdit):
            if "director" in (w.placeholderText() or ""):
                w.setFocus(); return
    console.add_shortcut("Ctrl+K", focus_ask, "ask the director")

    def save_scene():
        name, ok = QInputDialog.getText(console, "Save scene", "Scene name")
        if ok and name.strip():
            console.ctx.emit("scene_save", name.strip())
    console.add_shortcut("Ctrl+S", save_scene, "save scene")
    console.add_shortcut("H", lambda: console.ctx.emit("hold", not (console.state and console.state.get("state") == "hold")), "hold / release")
    console.add_shortcut("R", lambda: console.ctx.emit("reseed"), "new ideas")

    def show_help():
        rows = "\n".join(f"{seq:>10}   {label}" for seq, label in console.shortcuts())
        QMessageBox.information(console, "Keyboard", "<pre>" + rows + "</pre>")
    console.add_shortcut("F1", show_help, "this list")
    console.add_menu_action("Help", "Keyboard shortcuts", show_help, "F1")
