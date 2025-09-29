import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("runs_nano/pose/train15/weights/best.pt")

cap = cv2.VideoCapture(0)

plt.ion()  # interactive mode
fig, ax = plt.subplots()

ret, frame = cap.read()
if not ret:
    print("Failed to capture frame. Exiting...")
    cap.release()
    raise RuntimeError("Webcam not accessible.")

# init with first frame
im = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.axis('off')

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # inference
        results = model(frame, verbose=False)

        # Annotate frame with detections
        annotated_frame = results[0].plot()
        im.set_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        fig.canvas.flush_events()

        # control the frame rate
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    cap.release()
    plt.close()


