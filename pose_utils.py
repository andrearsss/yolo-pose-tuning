import numpy as np
import cv2

skeleton_mapping = {
    # legs and hips
    (0, 1): (255, 0, 0),
    (1, 2): (255, 69, 0),
    (2, 3): (255, 140, 0),
    (3, 4): (255, 0, 127),
    (4, 5): (255, 99, 71),
    (3, 9): (255, 165, 0),
    (2, 8): (255, 20, 147),
    # arms
    (9, 10): (255, 69, 0),
    (10, 11): (255, 0, 0),
    (8, 7): (255, 140, 0),
    (7, 6): (255, 165, 0),
    # head
    (8, 12): (255, 0, 127),
    (9, 12): (255, 20, 147),
    (12, 13): (255, 99, 71),
}
# skeleton_mapping = {
#     # legs and hips
#     (0, 1):     (255, 0, 0),
#     (1, 2):     (255, 0, 0),        ##
#     (2, 3):     (255, 0, 0),
#     (3, 4):     (255, 0, 0),
#     (4, 5):     (255, 0, 0),
#     (3, 9):     (255, 0, 0),
#     (2, 8):     (255, 0, 0),
#     # arms
#     (9, 10):    (255, 0, 0),
#     (10, 11):   (255, 0, 0),
#     (8, 7):     (255, 0, 0),
#     (7, 6):     (255, 0, 0),
#     # head
#     (8, 12):    (255, 0, 0),
#     (9, 12):    (255, 0, 0),
#     (12, 13):   (255, 0, 0),
# }

# Function to draw the skeleton
def draw_skeleton(frame, keypoints, skeleton_mapping=skeleton_mapping):
    for (start_idx, end_idx), color in skeleton_mapping.items():
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            if np.all(start_point > 0) and np.all(end_point > 0):
                start_point = tuple(map(int, start_point))
                end_point = tuple(map(int, end_point))
                cv2.line(frame, start_point, end_point, color, 2)

# Function to draw keypoints
def annotate_keypoints(frame, keypoints):
    for i, keypoint in enumerate(keypoints):
        x, y = map(int, keypoint)
        if x > 0 and y > 0: 
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

# Function to calculate angle between three points
def calculate_squat_angle(pointA, pointB, pointC):
    BA = np.array(pointA) - np.array(pointB)
    BC = np.array(pointC) - np.array(pointB)
    
    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees

# Function to draw the skeleton and detect a squat execution
def draw_skeleton_squat(frame, keypoints, skeleton_mapping):
    squat = False
    for (start_idx, end_idx), color in skeleton_mapping.items():
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            if np.all(start_point > 0) and np.all(end_point > 0):
                start_point = tuple(map(int, start_point))
                end_point = tuple(map(int, end_point))
                if (start_idx, end_idx) == (1,2):
                    r_ankle, r_knee, r_hip = keypoints[0], keypoints[1], keypoints[2]
                    degrees = calculate_squat_angle(r_ankle, r_knee, r_hip)
                    if degrees < 90:
                        color = (0, 255, 0)
                        squat = True
                cv2.line(frame, start_point, end_point, color, 2)
    return squat

# Function to draw counter in the top-right corner
def draw_count_squat(frame, count):
    frame_height, frame_width = frame.shape[:2]
    font_scale = frame_height * 0.1 / 32 
    font_thickness = 2
    text_size = cv2.getTextSize(str(count), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
    text_x = frame_width - text_size[0] - 20  # 20 pixels from the right edge
    text_y = text_size[1] + 20  # 20 pixels from the top edge

    cv2.putText(frame, str(count), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
# Function for video inference and squat detection
def video_inference_squat(model, video_path):
    output_path = f'{video_path}_squat.mp4'
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise FileNotFoundError(f"Video not found at {video_path}")

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    squat_count = 0
    squat_executed = False
    skip_frames = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break  # exit if no more frames

        # inference
        results = model(frame, verbose=False)

        # extract keypoints (assuming single person detected)
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # (num_keypoints, 2)
            squat_executed_previous = squat_executed
            squat_executed = draw_skeleton_squat(frame, keypoints, skeleton_mapping)
            if skip_frames > 0:
                skip_frames -=1
            # detect high-low transition
            if (squat_executed != squat_executed_previous) and squat_executed == True and skip_frames == 0:
                squat_count += 1
                skip_frames = 2*fps     # skip next 2 seconds to avoid instability around the threshold angle
            
            draw_count_squat(frame, squat_count)
            #annotate_keypoints(frame, keypoints)
        else:
            print("No keypoints detected in this frame.")

        out.write(frame)

    video.release()
    out.release()
    print(f"Output video saved at {output_path}")