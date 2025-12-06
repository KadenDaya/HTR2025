#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
from collections import deque
import urllib.request
import urllib.parse
import time
import threading

class StairDetectionTracker:
    """Tracks stair detections across frames for temporal smoothing."""
    def __init__(self, history_size=5, detection_threshold=0.3):
        self.history = deque(maxlen=history_size)
        self.detection_threshold = detection_threshold
        self.current_state = {'detected': False, 'confidence': 0.0, 'num_steps': 0, 
                             'bbox': None, 'step_heights': [], 'distance_to_first_step': None,
                             'step_positions': []}
    
    def update(self, new_detection):
        """Update tracker with new detection and return smoothed result."""
        self.history.append(new_detection)
        
        if len(self.history) < 2:
            self.current_state = new_detection
            return self.current_state
        
        # Calculate average confidence over history
        confidences = [d['confidence'] for d in self.history if d['detected']]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Count detections in history
        detection_count = sum(1 for d in self.history if d['detected'])
        detection_ratio = detection_count / len(self.history)
        
        # Smooth the detection state
        if detection_ratio >= self.detection_threshold:
            # Use most recent detection with high confidence, or average of recent ones
            recent_detections = [d for d in self.history if d['detected']]
            if recent_detections:
                # Use the most confident recent detection
                best_detection = max(recent_detections, key=lambda x: x['confidence'])
                self.current_state = best_detection.copy()
                self.current_state['confidence'] = avg_confidence
            else:
                self.current_state = new_detection
        else:
            # Not enough detections to confirm
            self.current_state = {'detected': False, 'confidence': 0.0, 'num_steps': 0, 
                                 'bbox': None, 'step_heights': [], 'distance_to_first_step': None,
                                 'step_positions': []}
        
        return self.current_state

def detect_stairs(depth_frame, rgb_frame=None, min_step_height_mm=80, max_step_height_mm=350, 
                  min_steps=3, roi_bottom_percent=0.65):
    """
    Detect stairs in depth frame using depth discontinuity analysis.
    
    Args:
        depth_frame: Depth frame in millimeters (numpy array)
        rgb_frame: Optional RGB frame for visualization
        min_step_height_mm: Minimum step height in mm (default 10cm)
        max_step_height_mm: Maximum step height in mm (default 30cm)
        min_steps: Minimum number of steps to detect (default 2)
        roi_bottom_percent: Bottom portion of frame to analyze (0.0-1.0, default 0.6)
    
    Returns:
        dict with keys: 'detected', 'confidence', 'num_steps', 'bbox', 'step_heights', 'distance_to_first_step'
    """
    h, w = depth_frame.shape
    
    # Create region of interest (lower portion where stairs typically appear)
    roi_top = int(h * (1 - roi_bottom_percent))
    roi_depth = depth_frame[roi_top:, :].copy()
    
    # Filter invalid depth values (0 or too large)
    valid_mask = (roi_depth > 200) & (roi_depth < 5000)  # 20cm to 5m range
    if np.sum(valid_mask) < roi_depth.size * 0.05:  # Lower threshold: need at least 5% valid pixels
        return {'detected': False, 'confidence': 0.0, 'num_steps': 0, 'bbox': None, 
                'step_heights': [], 'distance_to_first_step': None, 'step_positions': []}
    
    # Smooth depth to reduce noise
    # Convert to uint8 for median blur (OpenCV requirement), then convert back
    roi_depth_float = roi_depth.astype(np.float32)
    
    # Normalize to 0-255 range for median blur
    if np.sum(valid_mask) > 0:
        depth_min = np.nanmin(roi_depth_float[valid_mask])
        depth_max = np.nanmax(roi_depth_float[valid_mask])
        depth_range = depth_max - depth_min
        
        if depth_range > 1.0:  # Avoid division by zero
            # Normalize to 0-255
            roi_depth_norm = np.zeros_like(roi_depth_float)
            roi_depth_norm[valid_mask] = ((roi_depth_float[valid_mask] - depth_min) / depth_range * 255)
            roi_depth_norm = roi_depth_norm.astype(np.uint8)
            
            # Apply median blur
            roi_depth_blurred = cv2.medianBlur(roi_depth_norm, 7)
            
            # Convert back to original scale
            roi_depth_smooth = roi_depth_blurred.astype(np.float32) / 255.0 * depth_range + depth_min
            roi_depth_smooth[~valid_mask] = np.nan
        else:
            roi_depth_smooth = roi_depth_float
            roi_depth_smooth[~valid_mask] = np.nan
    else:
        roi_depth_smooth = roi_depth_float
        roi_depth_smooth[~valid_mask] = np.nan
    
    # Apply Gaussian blur for additional smoothing (works with float32)
    roi_depth_smooth = cv2.GaussianBlur(roi_depth_smooth, (5, 5), 0)
    
    # Calculate vertical gradient (detect horizontal edges where depth changes)
    # Stairs have depth increasing upward, so we look for positive vertical gradients
    gradient_y = np.gradient(roi_depth_smooth, axis=0)
    
    # Find strong horizontal edges (potential step risers)
    # Look for gradients that indicate depth increase (step going up)
    edge_strength = np.abs(gradient_y)
    
    # Use lower percentile threshold to be more permissive
    if np.sum(valid_mask) > 0:
        edge_threshold = np.nanpercentile(edge_strength[valid_mask], 60)  # Top 40% of gradients (more permissive)
    else:
        return {'detected': False, 'confidence': 0.0, 'num_steps': 0, 'bbox': None,
                'step_heights': [], 'distance_to_first_step': None, 'step_positions': []}
    
    # Create binary mask of potential step edges (more permissive)
    edge_mask = (edge_strength > edge_threshold) & (gradient_y > 0) & valid_mask
    
    if np.sum(edge_mask) < 30:  # Lower threshold: need minimum edge pixels
        return {'detected': False, 'confidence': 0.0, 'num_steps': 0, 'bbox': None,
                'step_heights': [], 'distance_to_first_step': None, 'step_positions': []}
    
    # Morphological operations to connect nearby edge pixels
    kernel = np.ones((3, 5), np.uint8)
    edge_mask = cv2.morphologyEx(edge_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_DILATE, kernel)
    
    # Find horizontal lines (step edges) using Hough transform on edge mask
    edge_uint8 = edge_mask * 255
    # More permissive Hough parameters
    lines = cv2.HoughLinesP(edge_uint8, 1, np.pi/180, threshold=20, 
                            minLineLength=w//5, maxLineGap=30)
    
    if lines is None or len(lines) < min_steps:
        return {'detected': False, 'confidence': 0.0, 'num_steps': 0, 'bbox': None,
                'step_heights': [], 'distance_to_first_step': None}
    
    # Extract y-coordinates of detected horizontal lines (step edges)
    step_y_positions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 15:  # More permissive: horizontal line (within 15 pixels)
            y_avg = (y1 + y2) // 2
            step_y_positions.append(y_avg + roi_top)  # Adjust for ROI offset
    
    if len(step_y_positions) < min_steps:
        return {'detected': False, 'confidence': 0.0, 'num_steps': 0, 'bbox': None,
                'step_heights': [], 'distance_to_first_step': None, 'step_positions': []}
    
    # Sort step positions from bottom to top and remove duplicates (within 10 pixels)
    step_y_positions = sorted(set(step_y_positions), reverse=True)
    # Remove positions that are too close together
    filtered_positions = [step_y_positions[0]]
    for pos in step_y_positions[1:]:
        if abs(pos - filtered_positions[-1]) > 10:  # At least 10 pixels apart
            filtered_positions.append(pos)
    step_y_positions = filtered_positions
    
    if len(step_y_positions) < min_steps:
        return {'detected': False, 'confidence': 0.0, 'num_steps': 0, 'bbox': None,
                'step_heights': [], 'distance_to_first_step': None, 'step_positions': []}
    
    # Validate step heights
    step_heights = []
    valid_steps = []
    x_center = w // 2
    x_range = slice(max(0, x_center - w//3), min(w, x_center + w//3))  # Wider sampling area
    
    for i in range(len(step_y_positions) - 1):
        y_bottom = step_y_positions[i]
        y_top = step_y_positions[i + 1]
        
        # Sample depth at these y positions (use median across width)
        depth_bottom = np.nanmedian(depth_frame[y_bottom, x_range])
        depth_top = np.nanmedian(depth_frame[y_top, x_range])
        
        if np.isnan(depth_bottom) or np.isnan(depth_top):
            continue
        
        # Calculate step height (depth difference)
        step_height = depth_top - depth_bottom
        
        # More permissive validation - allow slightly out of range if pattern is consistent
        if step_height >= min_step_height_mm * 0.7 and step_height <= max_step_height_mm * 1.3:
            step_heights.append(step_height)
            valid_steps.append((y_bottom, y_top))
    
    if len(valid_steps) < min_steps:
        return {'detected': False, 'confidence': 0.0, 'num_steps': 0, 'bbox': None,
                'step_heights': [], 'distance_to_first_step': None, 'step_positions': []}
    
    # Calculate confidence based on number of steps and consistency
    num_steps = len(valid_steps)
    if len(step_heights) > 1:
        step_height_std = np.std(step_heights)
        step_height_mean = np.mean(step_heights)
        consistency = 1.0 - min(1.0, step_height_std / step_height_mean)  # Lower std = higher consistency
    else:
        consistency = 0.5
    
    confidence = min(1.0, (num_steps / 5.0) * 0.7 + consistency * 0.3)  # Scale confidence
    
    # Calculate bounding box
    if valid_steps:
        y_min = valid_steps[-1][1]  # Top of top step
        y_max = valid_steps[0][0]    # Bottom of bottom step
        x_min = max(0, x_center - w//3)
        x_max = min(w, x_center + w//3)
        bbox = (x_min, y_min, x_max, y_max)
    else:
        bbox = None
    
    # Calculate distance to first step
    if valid_steps:
        y_first_step = valid_steps[0][0]
        distance_to_first_step = np.nanmedian(depth_frame[y_first_step, x_range])
    else:
        distance_to_first_step = None
    
    return {
        'detected': True,
        'confidence': confidence,
        'num_steps': num_steps,
        'bbox': bbox,
        'step_heights': step_heights,
        'distance_to_first_step': distance_to_first_step,
        'step_positions': step_y_positions
    }

pipeline = dai.Pipeline()

# RGB camera
rgb_cam = pipeline.create(dai.node.ColorCamera)
rgb_cam.setPreviewSize(640, 480)
rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb_cam.setInterleaved(False)
rgb_cam.setFps(30)

# Mono cameras for depth
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
mono_left.setCamera("left")
mono_right.setCamera("right")
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# Stereo depth with better settings
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(False)
stereo.setSubpixel(True)

# Set depth range (for OAK-D Lite: 20cm to 20m works well)
stereo.initialConfig.setConfidenceThreshold(200)

mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# RGB output
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
rgb_cam.preview.link(xout_rgb.input)

# Depth output
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Initialize YOLO 11 model
model = YOLO('yolo11n.pt')  # Will download automatically if not found locally
print("Loaded YOLO 11 model: yolo11n.pt")

# Initialize stair detection tracker for temporal smoothing
stair_tracker = StairDetectionTracker(history_size=5, detection_threshold=0.3)

# Track last notification time to throttle HTTP requests
last_notification_time = 0
notification_cooldown = 5.0  # Seconds between notifications (increased to prevent spamming while walking up stairs)

# Track last notification time for YOLO hazards (combined cooldown to prevent TTS overlap)
last_yolo_notification_time = 0
yolo_notification_cooldown = 4.0  # Seconds between YOLO notifications (prevents TTS overlap)

# Notification lock to prevent overlapping TTS
notification_lock = threading.Lock()
notification_in_progress = False

# Maximum distance for YOLO hazard notifications (in meters) - STRICT threshold
MAX_HAZARD_DISTANCE_M = 2.0  # Only notify for objects within 2 meters

# YOLO class names for hazards we care about
HAZARD_CLASSES = {
    'person': 0,
    'chair': 56,
    'couch': 57,
    'bed': 59,
    'dining table': 60,
    'desk': None,  # May not be in COCO, will check for table-like objects
}

def _send_http_request(url, message):
    """Helper function to send HTTP request in background thread."""
    global notification_in_progress
    try:
        urllib.request.urlopen(url, timeout=0.5)
        print(f"Sent notification: {message}")
    except Exception as e:
        # Silently fail if server is not available
        pass
    finally:
        # Release lock after request completes
        with notification_lock:
            notification_in_progress = False

def send_notification(message, notification_type='stair'):
    """Send HTTP notification to localhost:8080/?msg= with throttling. Prevents TTS overlap."""
    global last_notification_time, last_yolo_notification_time, notification_in_progress
    
    current_time = time.time()
    
    # Check if a notification is already in progress (prevent TTS overlap)
    with notification_lock:
        if notification_in_progress:
            return  # Skip if TTS is already speaking
        notification_in_progress = True
    
    # Use different cooldown based on notification type
    if notification_type == 'stair':
        if current_time - last_notification_time < notification_cooldown:
            with notification_lock:
                notification_in_progress = False
            return
        last_notification_time = current_time
    elif notification_type == 'proximity_danger':
        # Proximity danger uses same cooldown as YOLO hazards
        if current_time - last_yolo_notification_time < yolo_notification_cooldown:
            with notification_lock:
                notification_in_progress = False
            return
        last_yolo_notification_time = current_time
    else:
        # For YOLO objects, use combined cooldown to prevent overlap
        if current_time - last_yolo_notification_time < yolo_notification_cooldown:
            with notification_lock:
                notification_in_progress = False
            return
        last_yolo_notification_time = current_time
    
    # URL encode the message
    encoded_msg = urllib.parse.quote(message)
    url = f"http://localhost:8080/?msg={encoded_msg}"
    
    # Send GET request in background thread (truly non-blocking)
    thread = threading.Thread(target=_send_http_request, args=(url, message), daemon=True)
    thread.start()

def send_stair_notification(distance_m=None):
    """Send HTTP notification for stairs."""
    if distance_m is not None:
        message = f"stairs ahead {distance_m:.1f}m"
    else:
        message = "stairs ahead"
    send_notification(message, 'stair')

def check_proximity_danger(depth_frame):
    """Check if any depth reading is < 20cm (about to hit something like a wall)."""
    h, w = depth_frame.shape
    
    # Filter out invalid readings (0, too small, or too large)
    # Valid depth range: 50mm to 5000mm (5cm to 5m)
    valid_mask = (depth_frame > 50) & (depth_frame < 5000)
    valid_depths = depth_frame[valid_mask]
    
    if len(valid_depths) == 0:
        return False
    
    # Check minimum depth - if closest point is < 20cm, danger!
    min_depth = np.min(valid_depths)
    if min_depth < 200:  # Less than 20cm (200mm)
        send_notification("DANGER OBJECT AHEAD", 'proximity_danger')
        return True
    
    # Also check if we have many pixels indicating close obstacle
    # Look for valid depth readings between 50mm and 200mm (5cm to 20cm)
    close_obstacles = valid_depths[valid_depths < 200]
    
    # If we have significant number of close pixels, warn
    if len(close_obstacles) > 30:  # At least 30 pixels indicating close obstacle
        send_notification("DANGER OBJECT AHEAD", 'proximity_danger')
        return True
    
    return False

def check_yolo_hazards(results, depth_frame, frame_rgb):
    """Check YOLO detections for close hazards and send notifications. Prioritizes closest objects."""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return
    
    boxes = results[0].boxes
    depth_h, depth_w = depth_frame.shape
    rgb_h, rgb_w = frame_rgb.shape[:2]
    
    # Calculate scale factors to map RGB coordinates to depth coordinates
    scale_x = depth_w / rgb_w
    scale_y = depth_h / rgb_h
    
    # Get class names from model
    class_names = results[0].names
    
    # Collect all hazards with their distances
    hazards = []
    
    for i, box in enumerate(boxes):
        # Get bounding box coordinates (in RGB frame)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class ID and confidence
        cls_id = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        class_name = class_names[cls_id].lower()
        
        # Only check for specific hazard classes
        is_hazard = False
        hazard_type = None
        
        if 'person' in class_name or cls_id == 0:
            is_hazard = True
            hazard_type = 'person'
        elif 'chair' in class_name or cls_id == 56:
            is_hazard = True
            hazard_type = 'chair'
        elif 'couch' in class_name or 'sofa' in class_name or cls_id == 57:
            is_hazard = True
            hazard_type = 'chair'  # Group furniture together
        elif 'table' in class_name or 'desk' in class_name or cls_id == 60:
            is_hazard = True
            hazard_type = 'desk'
        elif 'bed' in class_name or cls_id == 59:
            is_hazard = True
            hazard_type = 'chair'  # Group furniture together
        
        if not is_hazard:
            continue
        
        # Calculate sampling point - use bottom center for people (feet level), center for objects
        if hazard_type == 'person':
            # For people, sample at bottom of bounding box (feet level) for accurate ground distance
            sample_x_rgb = (x1 + x2) / 2
            sample_y_rgb = y2 - 20  # Bottom of box, slightly up to avoid edge
        else:
            # For furniture, use center
            sample_x_rgb = (x1 + x2) / 2
            sample_y_rgb = (y1 + y2) / 2
        
        # Convert to depth frame coordinates
        sample_x_depth = int(sample_x_rgb * scale_x)
        sample_y_depth = int(sample_y_rgb * scale_y)
        
        # Sample depth in a region around the sampling point
        # Use larger horizontal sampling for better accuracy
        sample_size_x = 30
        sample_size_y = 15
        x_start = max(0, sample_x_depth - sample_size_x)
        x_end = min(depth_w, sample_x_depth + sample_size_x)
        y_start = max(0, sample_y_depth - sample_size_y)
        y_end = min(depth_h, sample_y_depth + sample_size_y)
        
        depth_region = depth_frame[y_start:y_end, x_start:x_end]
        valid_depths = depth_region[(depth_region > 200) & (depth_region < 5000)]
        
        if len(valid_depths) == 0:
            continue
        
        # Use median depth to avoid outliers, but prefer closer values for accuracy
        # Sort and take median of closest half for better accuracy
        sorted_depths = np.sort(valid_depths)
        # Use median of closest 60% of readings for more accurate distance
        closest_median_idx = int(len(sorted_depths) * 0.3)
        distance_mm = np.median(sorted_depths[:closest_median_idx + len(sorted_depths) // 2])
        distance_m = distance_mm / 1000.0
        
        # STRICT: Only notify if object is close (within MAX_HAZARD_DISTANCE_M)
        if distance_m > MAX_HAZARD_DISTANCE_M:
            continue
        
        # Store hazard info
        hazards.append({
            'distance': distance_m,
            'hazard_type': hazard_type
        })
    
    # Combine multiple hazards into one message
    if hazards:
        # Sort by distance (closest first)
        hazards.sort(key=lambda x: x['distance'])
        
        # Group hazards by type (count duplicates)
        hazard_counts = {}
        for hazard in hazards:
            h_type = hazard['hazard_type']
            if h_type not in hazard_counts:
                hazard_counts[h_type] = []
            hazard_counts[h_type].append(hazard['distance'])
        
        # Build combined message
        hazard_names = []
        for h_type in ['person', 'chair', 'desk']:  # Order: person first, then furniture
            if h_type in hazard_counts:
                hazard_names.append(h_type)
        
        if len(hazard_names) == 0:
            return
        
        # Build message
        if len(hazard_names) == 1:
            # Single item: include distance
            h_type = hazard_names[0]
            distance = hazards[0]['distance']  # Use closest distance
            message = f"{h_type} ahead {distance:.1f}m"
        else:
            # Multiple items: combine names, no distance
            if len(hazard_names) == 2:
                message = f"{hazard_names[0]} and {hazard_names[1]} ahead"
            else:
                # 3+ items: use commas and "and"
                items = ", ".join(hazard_names[:-1])
                message = f"{items}, and {hazard_names[-1]} ahead"
        
        # Send combined notification
        send_notification(message, 'yolo_hazard')

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    
    print("Starting video feeds... Press 'q' to quit")
    print("Depth colors: Red/Yellow = CLOSE, Purple/Blue = FAR")
    
    while True:
        # Get RGB frame (non-blocking - queue was created with blocking=False)
        img_frame = q_rgb.get()
        if img_frame is None:
            continue
        frame_rgb = img_frame.getCvFrame()
        
        # Get depth frame (non-blocking - queue was created with blocking=False)
        depth_frame = q_depth.get()
        if depth_frame is None:
            continue
        frame_depth = depth_frame.getFrame()
        
        # Check for proximity danger first (depth < 20cm - about to hit something)
        # This checks the entire depth frame for any readings < 20cm
        danger_detected = check_proximity_danger(frame_depth)
        
        # Run YOLO object detection (optimized for speed)
        # Use smaller input size for faster inference
        results = model(frame_rgb, verbose=False, conf=0.65, imgsz=640)
        
        # Only check YOLO hazards if no immediate danger was detected
        if not danger_detected:
            # Check for close hazards (people, chairs, desks) - STRICT distance filtering
            check_yolo_hazards(results, frame_depth, frame_rgb)
        
        # Draw YOLO detections on the frame (already filtered by confidence)
        annotated_frame = results[0].plot()
        
        # Detect stairs using depth data (always process for responsiveness)
        raw_stair_result = detect_stairs(frame_depth, frame_rgb)
        
        # Update tracker with new detection (provides temporal smoothing)
        stair_result = stair_tracker.update(raw_stair_result)
        
        # Draw stair detection on RGB frame
        if stair_result['detected']:
            # Send HTTP notification only if confidence is above 80%
            if stair_result['confidence'] >= 0.80:
                distance_m = None
                if stair_result['distance_to_first_step']:
                    distance_m = stair_result['distance_to_first_step'] / 1000.0
                send_stair_notification(distance_m=distance_m)
            
            # Draw bounding box
            if stair_result['bbox']:
                x_min, y_min, x_max, y_max = stair_result['bbox']
                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw step edges
            if 'step_positions' in stair_result and stair_result['step_positions']:
                for y_pos in stair_result['step_positions']:
                    cv2.line(annotated_frame, (0, y_pos), (annotated_frame.shape[1], y_pos), 
                            (0, 255, 255), 2)
            
            # Add text information
            info_text = f"STAIRS DETECTED! Steps: {stair_result['num_steps']}, " \
                       f"Confidence: {stair_result['confidence']:.2f}"
            if stair_result['distance_to_first_step']:
                distance_m = stair_result['distance_to_first_step'] / 1000.0
                info_text += f", Distance: {distance_m:.2f}m"
            
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Normalize depth properly (inverse so close = bright, far = dark)
        # Clip to reasonable range (0-5000mm = 0-5 meters)
        frame_depth_clipped = np.clip(frame_depth, 0, 5000)
        
        # Invert: close objects (small values) = 255, far objects (large values) = 0
        frame_depth_normalized = (255 - (frame_depth_clipped / 5000.0 * 255)).astype(np.uint8)
        
        # Apply TURBO colormap (Red/Yellow = close, Purple/Blue = far)
        frame_depth_display = cv2.applyColorMap(frame_depth_normalized, cv2.COLORMAP_TURBO)
        
        # Draw stair detection on depth frame for visualization
        if stair_result['detected'] and stair_result['bbox']:
            x_min, y_min, x_max, y_max = stair_result['bbox']
            cv2.rectangle(frame_depth_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display both frames (RGB with YOLO 11 detections + Stair detection)
        cv2.imshow("OAK-D RGB Feed (YOLO 11 + Stairs)", annotated_frame)
        cv2.imshow("OAK-D Depth Feed", frame_depth_display)
        
        # Break on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
print("Video feeds stopped")