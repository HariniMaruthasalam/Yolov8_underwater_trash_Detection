# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 12:02:51 2023

@author: harin
"""

from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
model = YOLO("C:/Users/harin/Downloads/best (3).pt")
classNames = ['bottle', 'can', 'plastic-bag', 'waste']

# Adjust the confidence threshold (0.5 in this example)
confidence_threshold = 0.5

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > confidence_threshold:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cls = int(box.cls[0])
                if cls < len(classNames):  # Check if the class index is within the range of classNames
                    class_name = classNames[cls]
                    label = f'{class_name} {conf:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_x, label_y = x1, y1 - 5
                    cv2.rectangle(img, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), (255, 0, 255), -1)
                    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255],
                                thickness=2, lineType=cv2.LINE_AA)
    out.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

out.release()




