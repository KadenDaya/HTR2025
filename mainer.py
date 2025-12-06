#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO
from collections import deque
import urllib.request
import urllib.parse
import time

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

def send_stair_notification(distance_m=None):
    """Send HTTP notification to localhost:8080/?msg= when stairs are detected."""
    global last_notification_time
    current_time = time.time()
    
    # Throttle notifications (don't spam)
    if current_time - last_notification_time < notification_cooldown:
        return
    
    try:
        # Build message with distance to start of stairs
        if distance_m is not None:
            message = f"stairs ahead {distance_m:.1f}m"
        else:
            message = "stairs ahead"
        
        # URL encode the message
        encoded_msg = urllib.parse.quote(message)
        url = f"http://localhost:8080/?msg={encoded_msg}"
        
        # Send GET request (non-blocking, timeout after 1 second)
        urllib.request.urlopen(url, timeout=1.0)
        last_notification_time = current_time
        print(f"Sent notification: {message}")
    except Exception as e:
        # Silently fail if server is not available
        pass

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    
    print("Starting video feeds... Press 'q' to quit")
    print("Depth colors: Red/Yellow = CLOSE, Purple/Blue = FAR")
    
    while True:
        # Get RGB frame
        img_frame = q_rgb.get()
        frame_rgb = img_frame.getCvFrame()
        
        # Run YOLO object detection on RGB frame with confidence threshold >= 55%
        results = model(frame_rgb, verbose=False, conf=0.55)
        
        # Draw YOLO detections on the frame (already filtered by confidence)
        annotated_frame = results[0].plot()
        
        # Get depth frame
        depth_frame = q_depth.get()
        frame_depth = depth_frame.getFrame()
        
        # Detect stairs using depth data
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