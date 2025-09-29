import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

from pose_utils import draw_skeleton, annotate_keypoints

model = YOLO("runs_nano/pose/train15/weights/best.pt")
cap = cv2.VideoCapture(0)

plt.ion()  # interactive mode
fig, ax = plt.subplots()

# Display the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to capture frame. Exiting...")
    cap.release()
    raise RuntimeError("Webcam not accessible.")

im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis('off')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame, verbose=False)

        # Extract keypoints for the first detected person
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # (num_keypoints, 2)
            draw_skeleton(frame, keypoints)
            annotate_keypoints(frame, keypoints)

        annotated_frame = results[0].plot()
        im.set_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        fig.canvas.flush_events()  # Update the plot in real time

        # Control the frame rate
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    cap.release()
    plt.close()
