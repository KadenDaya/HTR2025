#!/usr/bin/env python3
import cv2
import depthai as dai
import numpy as np
from ultralytics import YOLO

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

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    
    print("Starting video feeds... Press 'q' to quit")
    print("Depth colors: Red/Yellow = CLOSE, Purple/Blue = FAR")
    
    while True:
        # Get RGB frame
        img_frame = q_rgb.get()
        frame_rgb = img_frame.getCvFrame()
        
        # Run YOLO object detection on RGB frame
        results = model(frame_rgb, verbose=False)
        
        # Draw YOLO detections on the frame
        annotated_frame = results[0].plot()
        
        # Get depth frame
        depth_frame = q_depth.get()
        frame_depth = depth_frame.getFrame()
        
        # Normalize depth properly (inverse so close = bright, far = dark)
        # Clip to reasonable range (0-5000mm = 0-5 meters)
        frame_depth_clipped = np.clip(frame_depth, 0, 5000)
        
        # Invert: close objects (small values) = 255, far objects (large values) = 0
        frame_depth_normalized = (255 - (frame_depth_clipped / 5000.0 * 255)).astype(np.uint8)
        
        # Apply TURBO colormap (Red/Yellow = close, Purple/Blue = far)
        frame_depth_display = cv2.applyColorMap(frame_depth_normalized, cv2.COLORMAP_TURBO)
        
        # Display both frames (RGB with YOLO 11 detections)
        cv2.imshow("OAK-D RGB Feed (YOLO 11)", annotated_frame)
        cv2.imshow("OAK-D Depth Feed", frame_depth_display)
        
        # Break on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
print("Video feeds stopped")