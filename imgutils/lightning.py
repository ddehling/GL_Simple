import numpy as np
import cv2
import random
import math
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Point:
    x: float
    y: float

def generate_lightning(
    width: int = 120,
    height: int = 60,
    start_x: float = None,
    start_y: float = None,
    end_x: float = None,
    end_y: float = None,
    color: Tuple[float, float, float] = (0.8, 0.9, 1.0),  # Default: blue-white
    branch_probability: float = 0.95,   # Almost certain branching
    max_branches: int = 25,            # Many more branches
    jitter: float = 0.35,             # Slightly more chaos
    sub_branch_prob: float = 0.85     # Very high sub-branching
) -> np.ndarray:
    """
    Generates a lightning bolt optimized for 120x60 resolution.
    Args:
        color: RGB tuple (red, green, blue) with values 0-1
    Returns: RGBA array (height, width, 4) with transparent background
    """
    scale_factor = min(width, height) / 60.0
    
    if start_x is None:
        start_x = width * random.uniform(0.4, 0.6)
    if start_y is None:
        start_y = 0
    if end_x is None:
        end_x = width * random.uniform(0.3, 0.7)
    if end_y is None:
        end_y = height - 1

    core = np.zeros((height, width), dtype=np.float32)
    glow = np.zeros((height, width), dtype=np.float32)

    def get_segment_points(p1: Point, p2: Point) -> List[Point]:
        points = [p1]
        dist = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        segments = max(2, min(3, int(dist / (10 * scale_factor))))
        
        dx = (p2.x - p1.x) / segments
        dy = (p2.y - p1.y) / segments
        segment_length = math.sqrt(dx*dx + dy*dy)
        max_offset = segment_length * jitter

        for i in range(1, segments):
            x = p1.x + dx * i
            y = p1.y + dy * i
            angle = math.atan2(dy, dx) + math.pi/2
            offset = random.uniform(-max_offset, max_offset)
            x += math.cos(angle) * offset
            y += math.sin(angle) * offset
            points.append(Point(x, y))

        points.append(p2)
        return points

    def create_branch(start: Point, direction: float, length: float, branch_scale: float = 1.0) -> List[Point]:
        angle = direction + random.uniform(-math.pi/3, math.pi/3)
        end_x = start.x + math.cos(angle) * length * branch_scale
        end_y = start.y + math.sin(angle) * length * branch_scale
        
        end_x = max(0, min(width-1, end_x))
        end_y = max(0, min(height-1, end_y))
        
        return get_segment_points(start, Point(end_x, end_y))

    def draw_electric_line(img: np.ndarray, points: List[Point], intensity: float = 1.0, thickness: float = 1.0):

        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            x1, y1 = int(p1.x), int(p1.y)
            x2, y2 = int(p2.x), int(p2.y)
            
            x1 = max(0, min(width-1, x1))
            y1 = max(0, min(height-1, y1))
            x2 = max(0, min(width-1, x2))
            y2 = max(0, min(height-1, y2))
            
            cv2.line(img, (x1, y1), (x2, y2), intensity, int(thickness))


    def create_recursive_branches(start: Point, direction: float, length: float, depth: int = 0):
        if depth >= 3:  # Increased depth for more sub-branches
            return
            
        branch_points = create_branch(start, direction, length, 1.0 / (depth*0.2 + 1))
        
    # Calculate thickness based on depth
        core_thickness = max(1, 3 - depth)  # Starts at 3px, reduces by 1 each depth
        glow_thickness = core_thickness + 1  # Glow is always slightly larger
        
        intensity_core = 1.0 / (depth + 1)
        intensity_glow = 0.4 / (depth + 1)
        
        draw_electric_line(core, branch_points, intensity_core, core_thickness)
        draw_electric_line(glow, branch_points, intensity_glow, glow_thickness)
        
        # Create sub-branches
        if random.random() < sub_branch_prob / (depth + 1):
            for point in branch_points[1:-1]:
                if random.random() < branch_probability:
                    new_length = length * 0.8  # Longer sub-branches
                    create_recursive_branches(point, direction, new_length, depth + 1)

    # Generate main path
    main_points = get_segment_points(Point(start_x, start_y), Point(end_x, end_y))
    draw_electric_line(core, main_points, 1.0, 3)
    draw_electric_line(glow, main_points, 0.75, 4)

    # Generate branches
    min_branch_length = 8 * scale_factor   # Longer minimum length
    max_branch_length = 35 * scale_factor  # Much longer maximum length
    
    for i, point in enumerate(main_points[1:-1]):
        if random.random() < branch_probability:
            dx = main_points[i+1].x - main_points[i].x
            dy = main_points[i+1].y - main_points[i].y
            direction = math.atan2(dy, dx)
            length = random.uniform(min_branch_length, max_branch_length)
            create_recursive_branches(point, direction, length)

    # Apply minimal glow for small size
    glow = cv2.GaussianBlur(glow, (5, 5), 0)
    
    # Combine and enhance
    result = np.maximum(core, glow * 0.8)
    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
    result = np.power(result, 0.5)  # Increased brightness
    #divide color to make maximum value 1, or if all vlaues are zero, make color=1
    if max(color)==0:
        color=(1,1,1)
    else:
        color=(color[0]/max(color),color[1]/max(color),color[2]/max(color))

    # Create RGBA with color
    rgba = np.zeros((height, width, 4), dtype=np.float32)
    rgba[..., 0] = result * color[2]  # Blue
    rgba[..., 1] = result * color[1]  # Green
    rgba[..., 2] = result * color[0]  # Red
    rgba[..., 3] = result  # Alpha

    return (rgba*255).astype(np.uint8)

if __name__ == "__main__":
    # Example colors
    colors = [
        (0.8, 0.9, 1.0),  # Blue-white
        (1.0, 0.5, 0.0),  # Orange
        (0.7, 0.0, 1.0),  # Purple
        (1.0, 1.0, 1.0),  # Pure white
    ]
    
    # Generate lightning with random color
    lightning = generate_lightning(
        width=300,
        height=200,
        color=random.choice(colors),
        branch_probability=0.95,
        max_branches=15,
        jitter=0.3,
        sub_branch_prob = 0.95
    )
    
    # Convert to uint8
    display_img = lightning
    
    # Create checker background
    checker_size = 10
    background = np.zeros((display_img.shape[0], display_img.shape[1], 3), dtype=np.uint8)
    for i in range(0, display_img.shape[0], checker_size):
        for j in range(0, display_img.shape[1], checker_size):
            if (i // checker_size + j // checker_size) % 2:
                background[i:i+checker_size, j:j+checker_size] = 128
    
    # Blend for display
    alpha = display_img[..., 3:] / 255.0
    foreground = display_img[..., :3]
    blended = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
    
    # Display larger version for better visibility
    display_scale = 4
    large_display = cv2.resize(blended, (120*display_scale, 60*display_scale), 
                             interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow('Lightning', large_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()