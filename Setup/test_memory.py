import time
import psutil
import os
from Stories import *

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

cv2.namedWindow('Main', cv2.WINDOW_NORMAL)
cv2.namedWindow('Secondary', cv2.WINDOW_NORMAL)
scheduler = EventScheduler()
env_system = EnvironmentalSystem(scheduler)
# Monitor memory every 10 seconds
lasttime = time.time()
FRAME_TIME = 1/30
first_time=time.time()
while True:
    print_memory_usage()
    env_system.update()
    
    current_time = time.time()
    elapsed = current_time - lasttime
    sleep_time = max(0, FRAME_TIME - elapsed)
    time.sleep(sleep_time)
    
    #print([1/(time.time()-lasttime), len(scheduler.active_events), len(scheduler.event_queue),(lasttime-first_time)/3600])
    lasttime = time.time()