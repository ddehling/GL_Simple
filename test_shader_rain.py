import time
from corefunctions.Events import EventScheduler
from corefunctions.shader_events import shader_rain

def main():
    # Create scheduler with shader renderer enabled
    # Make window bigger to see both viewports clearly
    scheduler = EventScheduler(use_shader_renderer=True, window_width=1400, window_height=400)
    
    # Schedule shader rain for both frames
    print("Scheduling rain events...")
    event1 = scheduler.schedule_event(0, 60, shader_rain, intensity=1.5, frame_id=0)
    event2 = scheduler.schedule_event(0, 60, shader_rain, intensity=0.8, frame_id=1)
    print(f"Event 1 scheduled at {event1.start_time}, current time: {time.time()}")
    print(f"Event 2 scheduled at {event2.start_time}, current time: {time.time()}")
    print(f"Events in queue: {len(scheduler.event_queue)}")
    
    # Add some wind
    scheduler.state['wind'] = 0.5
    
    print("\nStarting shader rain test...")
    print("Rain will run for 60 seconds")
    print("Close the OpenGL window or press Ctrl+C to exit")
    
    last_time = time.time()
    FRAME_TIME = 1 / 50  # 50 FPS target
    start_time = time.time()
    frame_count = 0
    
    # Do a few updates to see what happens
    print("\nFirst few updates:")
    for i in range(5):
        print(f"\nUpdate {i}:")
        print(f"  Queue: {len(scheduler.event_queue)}, Active: {len(scheduler.active_events)}")
        print(f"  Current time: {scheduler.state['current_time']}")
        if scheduler.event_queue:
            print(f"  Next event at: {scheduler.event_queue[0].start_time}")
        scheduler.update()
        print(f"  After update - Queue: {len(scheduler.event_queue)}, Active: {len(scheduler.active_events)}")
        time.sleep(0.02)
    
    print("\nDebug info:")
    print(f"Number of viewports: {len(scheduler.shader_renderer.viewports)}")
    for vp in scheduler.shader_renderer.viewports:
        print(f"  Viewport {vp.frame_id}: {len(vp.effects)} effects")
        for eff in vp.effects:
            print(f"    - {eff.__class__.__name__}: enabled={eff.enabled}")
    
    print("\nContinuing main loop...")
    
    try:
        while time.time() - start_time < 65:  # Run for 65 seconds
            scheduler.update()
            
            current_time = time.time()
            elapsed = current_time - last_time
            sleep_time = max(0, FRAME_TIME - elapsed)
            time.sleep(sleep_time)
            
            frame_count += 1
            if frame_count % 50 == 0:  # Print FPS every second
                actual_fps = 1.0 / (elapsed + sleep_time)
                num_effects = sum(len(vp.effects) for vp in scheduler.shader_renderer.viewports)
                print(f"FPS: {actual_fps:.1f}, Active events: {len(scheduler.active_events)}, Active effects: {num_effects}")
            
            last_time = current_time
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    avg_fps = frame_count / (time.time() - start_time)
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Total frames rendered: {frame_count}")
    scheduler.cleanup()

if __name__ == "__main__":
    main()