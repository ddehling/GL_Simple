/** Entry: register every widget module, then boot the renderer.
 *  To add a widget type: create widgets/<name>.js that calls register(),
 *  import it here, and reference the type from lib/gen/ui.py. */
import './widgets/basic.js';
import './widgets/controls.js';
import './widgets/director.js';
import './widgets/pattern.js';
import './widgets/timeline.js';
import './app.js';
