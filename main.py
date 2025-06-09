import numpy as np
import cv2
import os

# --- File Paths (Use relative paths) ---
image_path = 'image02.jpg'
prototxt_path = 'MobileNetSSD_deploy.prototxt'
model_path = 'MobileNetSSD_deploy.caffemodel'
min_confidence = 0.01  # Lowered for testing/debugging

# --- Class Labels (No duplicates) ---
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor', 'book','bat','ball']

# --- Load and Validate Files ---
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    exit()
if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    print(f"❌ Model files not found. Check {prototxt_path} and {model_path}")
    exit()

# --- Load Model & Image ---
print("✅ Loading model and image...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
image = cv2.imread(image_path)
if image is None:
    print("❌ Failed to load image.")
    exit()

height, width = image.shape[:2]
np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- Create Blob and Perform Detection ---
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
net.setInput(blob)
detected_objects = net.forward()


# --- Process Detections ---
for i in range(detected_objects.shape[2]):
    confidence = detected_objects[0, 0, i, 2]
    class_index = int(detected_objects[0, 0, i, 1])
    class_name = classes[class_index % len(classes)]
    if confidence > min_confidence:
        upper_left_x = int(detected_objects[0, 0, i, 3] * width)
        upper_left_y = int(detected_objects[0, 0, i, 4] * height)
        lower_right_x = int(detected_objects[0, 0, i, 5] * width)
        lower_right_y = int(detected_objects[0, 0, i, 6] * height)

        label = f"{class_name}: {confidence:.2f}"
        color = colors[class_index % len(colors)]

        print(f"🔲 Drawing box for {label} at ({upper_left_x},{upper_left_y}) to ({lower_right_x},{lower_right_y})")

        cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), color, 3)
        cv2.putText(image, label,
                    (upper_left_x, upper_left_y - 10 if upper_left_y > 30 else upper_left_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# --- Show Result ---
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
