#!/usr/bin/env python3

import cv2
import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Create color camera
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setPreviewSize(640, 480)

# Create output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

# Connect to device
print("Connecting to OAK-D...")
with dai.Device(pipeline) as device:
    print("Connected! Starting video feed. Press 'q' to quit.")
    
    video_queue = device.getOutputQueue("video", maxSize=4, blocking=False)
    
    while True:
        video_frame = video_queue.tryGet()
        
        if video_frame is not None:
            frame = video_frame.getCvFrame()
            cv2.imshow("OAK-D Video Feed", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

print("Closing...")
cv2.destroyAllWindows()
