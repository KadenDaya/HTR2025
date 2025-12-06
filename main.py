#!/usr/bin/env python3
import cv2
import depthai as dai

pipeline = dai.Pipeline()

# RGB camera
rgb_cam = pipeline.create(dai.node.ColorCamera)
rgb_cam.setPreviewSize(640, 480)
rgb_cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgb_cam.setInterleaved(False)
rgb_cam.setFps(30)

# Create output
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
rgb_cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    
    print("Starting video feed... Press 'q' to quit")
    
    while True:
        # Get frame from queue
        img_frame = q.get()
        frame = img_frame.getCvFrame()
        
        # Display the frame
        cv2.imshow("OAK-D RGB Video Feed", frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
print("Video feed stopped")