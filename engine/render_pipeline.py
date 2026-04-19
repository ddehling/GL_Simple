"""Hardware integration layer: shader renderer + audio engine + DMX senders.

RenderPipeline wires together all hardware subsystems and drives the per-frame
render/send loop. It delegates event scheduling to an internal EventScheduler
and proxies its public API so callers can use a single object for both
scheduling and rendering.

The ``state`` dict (owned by EventScheduler) acts as a blackboard: hardware
references (``soundengine``, ``screens``, ``shader_renderer``) are seeded here
at init time and read by shader effects each frame.
"""

import time
import numpy as np
import cv2

import lib.audio_engine as sound
import lib.dmx_sender as imdmx
from renderer.shader_renderer import ShaderRenderer
from lib.event_scheduler import EventScheduler


class RenderPipeline:
    """Wires EventScheduler to ShaderRenderer, AudioEngine, and DMX senders.

    Owns hardware initialisation and the per-frame render/send loop.
    Delegates all event-queue operations to an internal EventScheduler so
    callers can use a single object for both scheduling and rendering.

    Args:
        frame_dimensions: List of (width, height) tuples, one per display.
        receivers: List of receiver lists (one per frame), each a list of
                   dicts with 'ip', 'pixel_count', and 'addressing_array'.
        magnification: Window scale factor (0 = auto).
        headless: Run without a visible window.
    """

    def __init__(self, frame_dimensions: list, receivers: list,
                 magnification: int = 1, headless: bool = False,
                 dmx_bind_ip: str = ""):
        self._scheduler = EventScheduler()
        self.should_exit = False
        self._cleaned_up = False

        # --- Renderer ---
        mode = "headless GPU" if headless else "GPU"
        print(f"[RenderPipeline] Initializing {mode} shader renderer...")
        self._shader_renderer = ShaderRenderer(
            frame_dimensions=frame_dimensions,
            headless=headless,
            magnification=magnification,
        )
        for frame_id in range(len(frame_dimensions)):
            self._shader_renderer.create_viewport(frame_id)
            if not headless:
                print(f"[RenderPipeline]   Viewport {frame_id}: {frame_dimensions[frame_id]}")

        # Seed the shared state dict with hardware references
        state = self._scheduler.state
        state['shader_renderer'] = self._shader_renderer
        state['event_scheduler'] = self._scheduler
        state['render'] = [None] * len(frame_dimensions)
        state['last_time'] = time.perf_counter()
        state['current_time'] = time.perf_counter()
        state['wind'] = 0
        state['tree_frame'] = np.zeros((60, 120, 4))
        state['rainrate'] = 0.5
        state['thunderrate'] = 0.0
        state['starryness'] = 0.0
        state['simulate'] = True
        print(f"[RenderPipeline] {mode} shader renderer initialized")

        # --- Audio ---
        engine = sound.ThreadedAudioEngine()
        engine.start()
        state['soundengine'] = engine

        # --- DMX / sACN ---
        self.frame_dimensions = frame_dimensions

        self.brightness_setpoint = 0.1
        self.brightness_config = {
            'red_factor': 1.0,
            'green_factor': 1.0,
            'blue_factor': 1.0,
            'threshold': 0.8,
            'smoothing': 0.05,
        }
        self.brightness_state = [
            {'divisor': 1.0, 'bright_factor': 0.0, 'last_logged_divisor': 0.0}
            for _ in frame_dimensions
        ]

        # Rate-limit preview PNG encoding to match the web emitter rate (15 Hz).
        # Encoding every render frame (40 Hz) wastes ~62% of the work on frames
        # the emitter never consumes.
        self._preview_encode_interval = 1.0 / 15.0
        self._last_preview_encode = 0.0

        total_pixels = sum(w * h for w, h in frame_dimensions)
        print(f"[RenderPipeline] Brightness limiting: {len(frame_dimensions)} displays, {total_pixels} total pixels")

        state['screens'] = []
        for frame_receivers in receivers:
            sender = imdmx.SACNPixelSender(
                frame_receivers, skip_network=False,
                use_raw_udp=True, per_receiver_universe=True,
                bind_ip=dmx_bind_ip,
            )
            sender.enable_async_send()
            state['screens'].append(sender)

        print("[RenderPipeline] sACN senders initialized with async mode enabled")

    # ------------------------------------------------------------------
    # EventScheduler delegation — callers use RenderPipeline as their
    # single handle for both scheduling and rendering.
    # ------------------------------------------------------------------

    @property
    def state(self) -> dict:
        return self._scheduler.state

    def has_action(self, action) -> bool:
        return self._scheduler.has_action(action)

    def schedule_event(self, delay, duration, action, *args, **kwargs):
        return self._scheduler.schedule_event(delay, duration, action, *args, **kwargs)

    def schedule_frame_event(self, delay, duration, action, frame_id=0, *args, **kwargs):
        return self._scheduler.schedule_frame_event(delay, duration, action, frame_id, *args, **kwargs)

    def cancel_all_events(self):
        self._scheduler.cancel_all_events()

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def update(self):
        self._shader_renderer.poll_events()
        if self._shader_renderer.should_close():
            print("[RenderPipeline] Window closed by user")
            self.cleanup()
            self.should_exit = True
            return

        current_time = time.perf_counter()
        self._scheduler.tick(current_time)

        dt = current_time - self.state['last_time']
        self.state['last_time'] = current_time

        frames = self._render_shader(dt)
        self._send_to_displays(frames)

        # PNG-encode first frame for web preview, ONLY if a client is subscribed.
        # Encoding is ~10-20ms/frame; rate-limited to the emitter rate (15 Hz)
        # so we don't burn CPU producing frames the emitter throws away.
        if frames and self.state.get('_preview_active'):
            now = time.perf_counter()
            if now - self._last_preview_encode >= self._preview_encode_interval:
                self._last_preview_encode = now
                bgr = np.ascontiguousarray(frames[0][:, :, ::-1])
                _, png_buf = cv2.imencode('.png', bgr)
                self.state['_frame_png'] = png_buf.tobytes()

    def _render_shader(self, dt: float) -> list:
        self._shader_renderer.clear_window()

        for viewport in self._shader_renderer.viewports:
            viewport.clear()
            viewport.update(dt, self.state)
            viewport.render(self.state)

        self._shader_renderer.sync_gpu()
        return [viewport.get_frame() for viewport in self._shader_renderer.viewports]

    def _send_to_displays(self, frames: list):
        gamma = 2.0
        for i, frame in enumerate(frames):
            frame_rgb = frame[:, :, :3] if frame.shape[2] == 4 else frame

            if gamma != 1:
                frame_corrected = np.power(frame_rgb / 255.0, gamma) * 255.0
            else:
                frame_corrected = frame_rgb.astype(np.float32)

            frame_corrected = self._apply_brightness_limiting(frame_corrected, i)

            screens = self.state['screens']
            if i < len(screens) and screens[i] is not None:
                try:
                    screens[i].send(frame_corrected[:, :, [0, 1, 2]])
                except OSError as e:
                    print(f"[RenderPipeline] Network error sending to display {i}: {e}")

        self._shader_renderer.swap_buffers()

    def _apply_brightness_limiting(self, frame_corrected: np.ndarray, frame_index: int) -> np.ndarray:
        """Scale frame down if total brightness exceeds the setpoint.

        Uses exponential smoothing on the divisor to prevent flickering.
        """
        cfg = self.brightness_config
        state = self.brightness_state[frame_index]

        height, width = frame_corrected.shape[:2]
        sum_of_weights = cfg['red_factor'] + cfg['green_factor'] + cfg['blue_factor']
        max_possible = height * width * 255.0 * sum_of_weights

        bright_factor = (
            np.sum(frame_corrected[:, :, 0]) * cfg['red_factor']
            + np.sum(frame_corrected[:, :, 1]) * cfg['green_factor']
            + np.sum(frame_corrected[:, :, 2]) * cfg['blue_factor']
        )
        normalized = bright_factor / max_possible
        state['bright_factor'] = normalized

        threshold_value = self.brightness_setpoint * cfg['threshold']
        if normalized <= threshold_value:
            target_divisor = 1.0
        else:
            target_divisor = max(1.0, normalized / self.brightness_setpoint)

        alpha = cfg['smoothing']
        state['divisor'] = alpha * target_divisor + (1 - alpha) * state['divisor']

        last = state['last_logged_divisor']
        if state['divisor'] > 1.001:
            frame_corrected = frame_corrected / state['divisor']
            # Log when limiting starts or value shifts by more than 20%
            if last <= 1.001 or abs(state['divisor'] - last) > 0.2:
                print(f"[RenderPipeline] Brightness limiting active: divisor={state['divisor']:.3f} (display {frame_index})")
                state['last_logged_divisor'] = state['divisor']
        elif last > 1.001:
            print(f"[RenderPipeline] Brightness limiting off (display {frame_index})")
            state['last_logged_divisor'] = state['divisor']

        # Apply web brightness modifier (can only dim, never exceed hardware limiter)
        web_brightness = self.state.get("web_brightness", 1.0)
        if web_brightness < 1.0:
            frame_corrected = frame_corrected * max(web_brightness, 0.0)

        return np.clip(frame_corrected, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self):
        if self._cleaned_up:
            return
        self._cleaned_up = True
        print("[RenderPipeline] Cleaning up...")
        self._scheduler.cancel_all_events()
        self._shader_renderer.cleanup()
        engine = self.state.get('soundengine')
        if hasattr(engine, 'stop'):
            engine.stop()
        print("[RenderPipeline] Cleanup complete")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
