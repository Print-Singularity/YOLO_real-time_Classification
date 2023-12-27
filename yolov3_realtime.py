
# Importing needed libraries
import numpy as np
import cv2
import time

PROTOCOL = "rtsp"
IP = "00.00.00.00"
USERNAME = "admin"
PASSWORD = "Passw0rd"
PORT = "554"

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

connection_string = "{}://{}:{}@{}:{}/stream0".format(PROTOCOL, USERNAME, PASSWORD, IP, PORT)   # rtsp://admin:Passw0rd@192.168.7.57:554/stream0
print("Connection String: ", connection_string)

camera = cv2.VideoCapture(connection_string, cv2.CAP_FFMPEG)


# codec = 0x47504A4D  # MJPG
camera.set(cv2.CAP_PROP_FPS, 30)
# cap.set(cv2.CAP_PROP_FOURCC, codec)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)         # 1920, 1280, 640
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)        # 1080,  720, 480

vid_fps = int(camera.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS: {}, WIDTH: {}, HEIGHT:{}".format(vid_fps, vid_width, vid_height))

cv2.namedWindow("Streaming", cv2.WINDOW_AUTOSIZE)

# Preparing variables for spatial dimensions of the frames
h, w = None, None

with open("models\\coco.names") as f:
    labels = [line.strip() for line in f]


network = cv2.dnn.readNetFromDarknet("models\\yolov3.cfg",
                                    "weights\\yolov3.weights")

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()] 
probability_minimum = 0.3
threshold = 0.7
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Defining loop for catching frames
while True:
    # Capturing frame-by-frame from camera
    status, frame = camera.read()
    if status:
        

        if w is None or h is None:
            # Slicing from tuple only first two elements
            h, w = frame.shape[:2]

        # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)

        network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

        # Showing spent time for single current frame
        print('Current frame took {:.5f} seconds'.format(end - start))

        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # Eliminating weak predictions with minimum probability
                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min,
                                        int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                probability_minimum, threshold)

        # Checking if there is at least one detected object
        # after non-maximum suppression
        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                colour_box_current = colours[class_numbers[i]].tolist()

                # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                    confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

        cv2.namedWindow('Streaming', cv2.WINDOW_NORMAL)
        cv2.imshow('Streaming', frame)

        # Breaking the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()                # Releasing camera
cv2.destroyAllWindows()         # Destroying all opened OpenCV windows