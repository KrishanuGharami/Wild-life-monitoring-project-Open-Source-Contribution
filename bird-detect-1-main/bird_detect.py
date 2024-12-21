import cv2
import datetime
import os
from ultralytics import YOLO

# Loading pretrained YOLO model
model = YOLO("model/yolov8n.pt", "v8")

# Set dimensions of video frames
frame_width = 1280
frame_height = 720

# Video source is the default webcam
cap = cv2.VideoCapture(0)

# Variables to track bird detection duration
bird_detected_time = None
bird_total_time = []
bird_in_frame = False

if not cap.isOpened():
    print("Cannot open video stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("No video frame available")
        break

    # Resize the frame
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Perform object detection on the frame
    detect_params = model.predict(source=[frame], conf=0.8, save=False)
    DP = detect_params[0].numpy()

    bird_present = False  # Flag to check if a bird is in the current frame

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            c = box.cls.numpy()[0]
            class_name = model.names[int(c)]

            if 'bird' in class_name.lower():
                bird_present = True
                # Draw bounding box and label
                bb = box.xyxy.numpy()[0]
                conf = box.conf.numpy()[0]
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 3)
                cv2.putText(frame, f"{class_name} {round(conf, 3)}%", (int(bb[0]), int(bb[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Start timer if bird is detected
    if bird_present and not bird_in_frame:
        bird_in_frame = True
        bird_detected_time = datetime.datetime.now()
        print("Bird detected. Timer started.")

    # Stop timer if bird is no longer detected
    if not bird_present and bird_in_frame:
        bird_in_frame = False
        elapsed_time = datetime.datetime.now() - bird_detected_time
        bird_total_time.append(elapsed_time.total_seconds())
        print(f"Bird left. Total duration: {elapsed_time.total_seconds():.2f} seconds.")
        bird_detected_time = None

    # Display elapsed time if bird is currently in frame
    if bird_in_frame:
        elapsed_time = datetime.datetime.now() - bird_detected_time
        cv2.putText(frame, f"Time: {elapsed_time.total_seconds():.2f} sec", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # End program when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Output total detection durations
print("\nSummary of Bird Detection Durations:")
for i, duration in enumerate(bird_total_time, 1):
    print(f"Event {i}: {duration:.2f} seconds")
