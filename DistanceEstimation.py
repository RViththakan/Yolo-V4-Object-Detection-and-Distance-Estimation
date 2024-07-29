import cv2 as cv
import numpy as np

# Distance constants 
KNOWN_DISTANCE = 45 * 0.0254  # Convert inches to meters
PERSON_WIDTH = 16 * 0.0254  # Convert inches to meters
MOBILE_WIDTH = 3.0 * 0.0254  # Convert inches to meters

# Object detector constants 
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detection
BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (0,0,255)

# Defining fonts 
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file 
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up OpenCV net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Object detector function
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    # Convert lists or tuples to numpy arrays if necessary
    if isinstance(classes, tuple):
        classes = np.array(classes)
    if isinstance(scores, tuple):
        scores = np.array(scores)
    if isinstance(boxes, tuple):
        boxes = np.array(boxes)

    data_list = []
    for (classid, score, box) in zip(classes.flatten(), scores.flatten(), boxes):
        color = BLACK  # Set bounding box color to black
        label = "%s : %.2f" % (class_names[classid].upper(), score)  # Convert label to uppercase and format score
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
        
        # Update data_list with class information and bounding box width
        if classid == 0:  # 'pedestrian'
            data_list.append([class_names[classid].upper(), box[2], (box[0], box[1])])
        elif classid == 67:  # 'cell phone'
            data_list.append([class_names[classid].upper(), box[2], (box[0], box[1])])
            
    return data_list

def focal_length_finder(measured_distance, real_width, width_in_rf):
    return (width_in_rf * measured_distance) / real_width

def distance_finder(focal_length, real_object_width, width_in_frame):
    return (real_object_width * focal_length) / width_in_frame

# Reading the reference images
ref_person = cv.imread('ReferenceImages/image14.png')
ref_mobile = cv.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1] if mobile_data else 0

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1] if person_data else 0

print(f"Person width in pixels: {person_width_in_rf} mobile width in pixels: {mobile_width_in_rf}")

# Finding focal lengths
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    data = object_detector(frame)
    show_popup = False

    for d in data:
        if d[0] == 'PEDESTRIAN':  # Ensure this matches the class name from classes.txt, in uppercase
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            distance_meters = round(distance, 2)  # Convert distance to meters
            x, y = d[2]
            x = int(x)
            y = int(y)
            
            # Draw "Distance in meters" text below the bounding box
            distance_text = f'Distance in meters: {distance_meters} m'
            distance_y_position = y + 15  # Position for distance text below the bounding box
            cv.putText(frame, distance_text, (x + 5, distance_y_position), FONTS, 0.6, BLACK, 2)

            print(f"Detected pedestrian at {distance_meters} meters")

            if distance_meters < 1:
                show_popup = True

        elif d[0] == 'CELL PHONE':  # Ensure this matches the class name from classes.txt, in uppercase
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            distance_meters = round(distance, 2)  # Convert distance to meters
            x, y = d[2]
            x = int(x)
            y = int(y)
            
            # Draw "Distance in meters" text for cell phone
            distance_text = f'Distance in meters: {distance_meters} m'
            distance_y_position = y + 15  # Position for distance text below the bounding box
            cv.putText(frame, distance_text, (x + 5, distance_y_position), FONTS, 0.6, BLACK, 2)

            print(f"Detected cell phone at {distance_meters} meters")

            if distance_meters < 1:
                show_popup = True

    # Add text to the top right corner with a black box around it
    top_right_text = 'I-Bike - EN17360692'
    (text_width, text_height), _ = cv.getTextSize(top_right_text, FONTS, 0.6, 2)
    frame_height, frame_width = frame.shape[:2]
    box_x_start = frame_width - text_width - 20
    box_y_start = 20
    box_x_end = frame_width - 10
    box_y_end = 20 + text_height + 10
    # Draw black rectangle
    cv.rectangle(frame, (box_x_start, box_y_start), (box_x_end, box_y_end), BLACK, -1)
    # Draw white text on top of the rectangle
    cv.putText(frame, top_right_text, (box_x_start + 10, box_y_end - 10), FONTS, 0.6, WHITE, 2)

    # Draw red popup text if needed
    if show_popup:
        popup_text = 'Obstacle at Close Proximity!'
        (popup_width, popup_height), _ = cv.getTextSize(popup_text, FONTS, 0.8, 2)
        popup_x = int(frame_width / 2 - popup_width / 2)
        popup_y = int(frame_height / 2 + popup_height / 2)
        # Draw a red rectangle as background for the text
        cv.rectangle(frame, (popup_x - 10, popup_y - popup_height - 10), 
                             (popup_x + popup_width + 10, popup_y + 10), RED, -1)
        # Draw the red text
        cv.putText(frame, popup_text, (popup_x, popup_y), FONTS, 0.8, WHITE, 2)

    cv.imshow('Vehicle Detection', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
