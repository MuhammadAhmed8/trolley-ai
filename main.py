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

def decode_barcode_in_image(image_path):
    # Load the image
    frame = cv2.imread(image_path)
    # Convert image to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect barcodes in the image
    barcodes = pyzbar.decode(gray_frame)

    if not barcodes:
        print("No barcodes detected")

    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        text = f"{barcode_type}: {barcode_data}"
        print(f"Detected barcode: {text}")
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with detected barcodes
    cv2.imshow("Barcode Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("Starting static image reader")
    decode_barcode_in_image("barcode-upc-ean.png")
# def main():
#     print("Starting Program!")

#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
    
#     if not cap.isOpened():
#         print("Error: Could not open video stream.")
#         return
    
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break

#         # Detect items using YOLOv5
#         # detections = detect_items_in_frame(frame)
#         frame = decode_barcode(frame)

#         # Annotate the frame with detection results
#         # for _, row in detections.iterrows():
#         #     # Draw bounding box
#         #     x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
#         #     label = f"{cls} {conf:.2f}"

#         #     # Draw bounding box and label on frame
#         #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         #     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Display the frame
#         cv2.imshow("Barcode Scanner", frame)

#         # Press 'q' to quit the application
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# WARNING: barcode data was not detected in some image(s)
# Things to check:
#   - is the barcode type supported? Currently supported symbologies are:
#         . EAN/UPC (EAN-13, EAN-8, EAN-2, EAN-5, UPC-A, UPC-E, ISBN-10, ISBN-13)
#         . DataBar, DataBar Expanded
#         . Code 128
#         . Code 93
#         . Code 39
#         . Codabar
#         . Interleaved 2 of 5
#         . QR code
#         . SQ code
#   - is the barcode large enough in the image?
#   - is the barcode mostly in focus?
#   - is there sufficient contrast/illumination?
#   - If the symbol is split in several barcodes, are they combined in one image?
#   - Did you enable the barcode type?
#     some EAN/UPC codes are disabled by default. To enable all, use:
#     $ zbarimg -S*.enable <files>
#     Please also notice that some variants take precedence over others.
#     Due to that, if you want, for example, ISBN-10, you should do:
#     $ zbarimg -Sisbn10.enable <files>