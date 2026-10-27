/*
  GL_Simple Mirror — receiver pattern for lib/pixelblaze_mirror.py.

  Paste this into a Pixelblaze's editor and save it as "GL_Simple Mirror".
  When the show machine sees the Pixelblaze it switches to this pattern
  (not saved to flash) and streams its output here; when the show shuts
  down it switches the Pixelblaze back to what it was running.

  A Pixelblaze's time goes on parsing numbers and on render() per LED, so
  the show sends at most ~600 SAMPLES a frame, not every pixel:
    seg     pixels per segment (a strand; the whole strip if there's one)
    k       samples per segment, spread evenly from its first pixel to its last
    interp  1 = blend between neighbouring samples, 0 = nearest sample
  Never blends across a segment's end. Each sample is one number,
  r + g/256 + b/65536.

  render() is kept to the bone because it runs per LED: no division, the
  sample position advances by a running counter (render is called in index
  order). Measured on a 2400-px Pixelblaze 3: nearest ~11.7 fps, blending
  ~7 fps. If frames stop arriving the pixels fade to black after 2 s.
*/

var MAXS = 1000
export var p0 = array(MAXS)   // packed samples, written by the show
export var seq = 0
export var seg = 1
export var k = 1
export var interp = 0

var tr = array(MAXS)
var tg = array(MAXS)
var tb = array(MAXS)

var lastSeq = -1
var sinceFrame = 0
var n = 0
var scale = 0
var lastK = 0
var loc = 0
var tt = 0
var base = 0

export function beforeRender(delta) {
  // Capped: Pixelblaze numbers overflow past 32767.
  sinceFrame = min(sinceFrame + delta, 10000)
  if (seq != lastSeq) {
    lastSeq = seq
    sinceFrame = 0
    n = min(ceil(pixelCount / seg) * k, MAXS)
    scale = k > 1 && seg > 1 ? (k - 1) / (seg - 1) : 0
    lastK = k - 1
    for (var i = 0; i < n; i++) {
      v = p0[i]
      r = floor(v)
      f = (v - r) * 256
      g = floor(f)
      tr[i] = r / 255
      tg[i] = g / 255
      tb[i] = (f - g) * 256 / 255
    }
  } else if (sinceFrame > 2000) {
    // Stale: fade the samples out over about a second.
    fade = clamp(1 - delta / 1000, 0, 1)
    for (var i = 0; i < n; i++) {
      tr[i] *= fade
      tg[i] *= fade
      tb[i] *= fade
    }
  }
  loc = 0
  base = 0
  tt = interp ? 0 : 0.5
}

export function render(index) {
  if (interp) {
    j = floor(tt)
    f = tt - j
    i0 = base + j
    i1 = j < lastK ? i0 + 1 : i0
    rgb(tr[i0] + (tr[i1] - tr[i0]) * f,
        tg[i0] + (tg[i1] - tg[i0]) * f,
        tb[i0] + (tb[i1] - tb[i0]) * f)
  } else {
    i0 = base + floor(tt)
    rgb(tr[i0], tg[i0], tb[i0])
  }
  tt += scale
  loc += 1
  if (loc >= seg) {
    loc = 0
    tt = interp ? 0 : 0.5
    base += k
  }
}
