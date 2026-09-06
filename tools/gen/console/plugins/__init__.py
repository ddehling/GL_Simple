"""Console plugins. Any module here with `def register(console)` is loaded
at start (tools/gen/console/app.py). The console object offers:
    add_tab(title, widget)          widget.refresh(state) is called while visible
    add_shortcut(seq, fn, label)    global keyboard shortcut (listed in Help)
    add_status(label_widget)        a permanent status-bar item
    on_state(fn)                    fn(state) after every refresh (10 Hz)
    add_menu_action(menu, text, fn, shortcut=None)
    ctx.emit(action, value)         send an operator action (whitelisted)
    backend / state                 the current backend and last status dict
"""
