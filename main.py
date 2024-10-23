import cv2
import torch
from pyzbar import pyzbar

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

def decode_barcode(frame):
    # Detect barcodes in the frame
    barcodes = pyzbar.decode(frame)

    if not barcodes:
        print("No barcodes detected")

    for barcode in barcodes:
        print(f"Detected barcode: {barcode.data.decode('utf-8')}, Type: {barcode.type}")
        
        # Extract the bounding box location of the barcode and draw a rectangle
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Decode the barcode data and convert it to a string
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # Display the barcode type and data
        text = f"{barcode_type}: {barcode_data}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def main():
    print("Starting Program!")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect items using YOLOv5
        # detections = detect_items_in_frame(frame)
        frame = decode_barcode(frame)

        # Annotate the frame with detection results
        # for _, row in detections.iterrows():
        #     # Draw bounding box
        #     x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
        #     label = f"{cls} {conf:.2f}"

        #     # Draw bounding box and label on frame
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Barcode Scanner", frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

