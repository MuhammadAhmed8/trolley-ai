import cv2
import torch

# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# Define COCO classes (these can be modified for your use case)
COCO_CLASSES = model.names

def detect_items_in_frame(frame):
    """
    Perform item detection on a single frame using YOLOv5.
    Returns the detections (bounding boxes, labels, etc.).
    """
    # Inference
    results = model(frame)

    # Non-max suppression to filter out unnecessary detections
    detections = results.pandas().xyxy[0]  # Bounding box coords and class labels

    return detections

def main():
    print("Starting Program!")

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect items using YOLOv5
        detections = detect_items_in_frame(frame)

        # Annotate the frame with detection results
        for _, row in detections.iterrows():
            # Draw bounding box
            x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
            label = f"{cls} {conf:.2f}"

            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Smart Trolley", frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

