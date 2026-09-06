/** Widget registry: type -> { create(spec, ctx) -> element, update(el, state, spec, ctx) }.
 *  A widget module registers itself on import; lib/gen/ui.py's gate scans
 *  these files for register('type') so the spec can never name a widget
 *  the client does not have. */
const REGISTRY = new Map();
export function register(type, def) { REGISTRY.set(type, def); }
export function get(type) { return REGISTRY.get(type); }
export function types() { return [...REGISTRY.keys()]; }
