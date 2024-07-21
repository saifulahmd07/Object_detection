import numpy as np
import cv2
import json

def hex_to_bgr(hex_color):
    # Menghapus tanda #
    hex_color = hex_color.lstrip('#')
    # Mengonversi hex ke desimal
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV menggunakan format BGR

classes = ["Fresh Apple", "Rotten Apple"]
colors = {
    "Fresh Apple": hex_to_bgr("#0020ff"),  
    "Rotten Apple": hex_to_bgr("#20cfff")   
}
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNetFromONNX("best.onnx")

while True:
    _, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=[640, 640], mean=[0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()[0]

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width / 640
    y_scale = img_height / 640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.2:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.2:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx - w / 2) * x_scale)
                y1 = int((cy - h / 2) * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1, y1, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

    for i in indices:
        x1, y1, w, h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = f"{label} {conf:.2f}"
        color = colors[label]
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)

    cv2.imshow("Object Detection", img)
    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
