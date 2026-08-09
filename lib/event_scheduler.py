"""Pure timed-event queue and shared state dict (the blackboard).

TimedEvent: a single scheduled action with timing bookkeeping.
EventScheduler: a min-heap queue that promotes events when their start time
arrives and drives them each tick. Owns the ``state`` dict that shader
effects and the pipeline read/write during every frame.

Has no renderer, audio, or network dependencies.
"""

import time
import heapq
import numpy as np

class EventScheduler:
    """Pure event queue: schedule, fire, and cancel timed events.

    Owns the shared state dict (a blackboard read and written by shader effects
    and the pipeline). Has no renderer, audio, or network dependencies.
    """

    def __init__(self):
        self.event_queue = []
        self.active_events = []
        self.state = {}

    def tick(self, current_time: float):
        """Promote due events to active and call update() on all active events."""
        self.state['current_time'] = current_time

        while self.event_queue and self.event_queue[0].state['start_time'] <= current_time:
            self.active_events.append(heapq.heappop(self.event_queue))

        i = 0
        while i < len(self.active_events):
            if self.active_events[i].update(self.state):
                i += 1
            else:
                self.active_events.pop(i)

    def schedule_event(self, delay, duration, action, *args, **kwargs):
        """Schedule an event; silently drops duplicates that are already active or queued.

        Duplicate = the SAME action already scheduled on the SAME frame
        (group). The frame_id is part of the key so a shared effect
        function can legitimately run on more than one group canvas at
        once — e.g. the WoL rain/theme shaders that paint both the Sky
        strips and the Ground arcs. Without the frame_id in the key the
        second group's copy was silently dropped.

        ``allow_duplicates=True`` opts an individual call out of the check,
        for effects that are safe to run as several concurrent instances —
        each gets its own effect object from ``add_effect``, so the copies
        don't share state. Only pass it for effects whose wrapper stores
        everything in its own ``state``/effect instance; a wrapper holding
        module-level or class-level state would have its copies fight.
        Popped before the kwargs reach the action.
        """
        allow_duplicates = kwargs.pop('allow_duplicates', False)
        frame_id = kwargs.get('frame_id', None)

        def _is_dup(event) -> bool:
            return (event.action == action
                    and event.state.get('frame_id', None) == frame_id)

        if not allow_duplicates and (any(_is_dup(e) for e in self.active_events) or
                                     any(_is_dup(e) for e in self.event_queue)):
            action_name = action.__name__ if hasattr(action, '__name__') else str(action)
            print(f"[EventScheduler] Skipping duplicate event: {action_name} "
                  f"on frame {frame_id} (already running or queued)")
            return None

        event_time = time.perf_counter() + delay
        name = kwargs.pop('name', None)
        frame_id = kwargs.pop('frame_id', None)

        event = TimedEvent(event_time, duration, action, args, kwargs, name=name, frame_id=frame_id)
        heapq.heappush(self.event_queue, event)
        return event

    def schedule_frame_event(self, delay, duration, action, frame_id=0, *args, **kwargs):
        """Convenience wrapper that passes frame_id as a kwarg."""
        kwargs['frame_id'] = frame_id
        return self.schedule_event(delay, duration, action, *args, **kwargs)

    def cancel_all_events(self):
        # Close every active event defensively: a single effect's teardown
        # raising (e.g. a wrapper's count==-1 branch hitting a stale
        # viewport.effects.remove) must NOT abort the loop and leave the
        # remaining events stranded in active_events -- that strands their
        # effects AND makes schedule_event() silently skip same-named new
        # events as "duplicates" on the next set load. Always clear the lists.
        for event in self.active_events:
            try:
                event.closeevent(self.state)
            except Exception as e:
                print(f"[EventScheduler] error closing event "
                      f"{getattr(event, 'name', '?')}: {e}")
        self.event_queue = []
        self.active_events = []

    def has_action(self, action) -> bool:
        return any(event.action == action for event in self.event_queue) or \
               any(event.action == action for event in self.active_events)



class TimedEvent:
    def __init__(self, start_time, duration, action, args=(), kwargs={}, name=None, frame_id=None):
        self.start_time = start_time
        self.duration = duration
        self.action = action
        self.args = args
        self.kwargs = kwargs
        self.name = name if name is not None else (action.__name__ if hasattr(action, '__name__') else str(action))
        self.state = {}
        self.state['count'] = 0
        self.state['start_time'] = start_time
        self.state['duration'] = duration
        self.state['elapsed_time'] = 0
        self.state['elapsed_fraction'] = 0
        self.frame_duration = []
        if frame_id is not None:
            self.state['frame_id'] = frame_id

    def __lt__(self, other):
        return self.start_time < other.start_time

    def update(self, outstate):
        start = time.perf_counter_ns()

        self.state['elapsed_time'] = outstate['current_time'] - self.state['start_time']
        self.state['elapsed_fraction'] = self.state['elapsed_time'] / self.state['duration']
        if self.state['elapsed_time'] > self.state['duration']:
            self.closeevent(outstate)
            return False

        self.action(self.state, outstate, *self.args, **self.kwargs)
        self.state['count'] += 1

        elapsed = (time.perf_counter_ns() - start) / 1.0E9
        if self.state['count'] < 1000:
            self.frame_duration.append(elapsed)
        return True

    def closeevent(self, outstate):
        median_duration = np.median(self.frame_duration) if self.frame_duration else 0
        print(f"[EventScheduler] Event closed: {self.name} Length:{median_duration:.6f}s")
        self.state['count'] = -1
        self.action(self.state, outstate, *self.args, **self.kwargs)