# yolo: used for object detection 
# the object of our concern is a person 
# we use object tracking
# after detection of person class, we assign an id to them, create a box over them, find the centroid
# euclidean distance will be calculated among each centroid

# distance measurement algorithm

import numpy as np
import cv2
from scipy.spatial import distance as dist
import argparse
import imutils
import os

# initialize minimum probability to filter weak detections along with the threshold when applying non-maxima suppression

MIN_CONF = 0.3 
NMS_THRESH = 0.3 #used for drawing the boxes over the detection

# define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE = 50

# people detection function  

def detect_people(frame, net, ln, personIdx=0):
    # initialize the dimensions and results of the frame 
    (H, W) = frame.shape[:2]
    results = []

    # constructing a blob from the input frame and perform forward pass of YOLO object detector, which gives bounding boxes and associated probabilities
    # A blob is a 4D numpy array object (images, channels, width, height). 

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = [] #bounding boxes
    centroids = [] 
    confidences = []

    # looping over each layer
    for output in layerOutputs:
        # loop over each detected object
        for detection in output:
            # get classID and confidence
            # first 4 returns the coordinates of the bounding box (centerx, centery, width, height)
            # 5th returns the confidence of the box
            # remaining 80 returns class confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # ensure the object detected was a person and minimum confidence is met

            if classID == personIdx and confidence > MIN_CONF:
                # rescale the bounding box coordinates back relative to size of image as YOLO returns the center of the bounding box and width and height of the box

                box = detection[:4]*np.array([W, H, W, H])
                (centerx, centery, width, height) = box.astype("int")

                # use these coordinates to get top left corner

                x = int(centerx - (width/2))
                y = int(centery - (height/2))

                # update the lists

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerx, centery))
                confidences.append(float(confidence))

    # non-maxima suppression: bounding boxes are eliminated because their confidence is low or because they are enclosing the same object as another bounding box with very high confidence score

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)


    # to ensure atleast one detection exists

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x,y,w,h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

            r = (confidences[i], (x,y,x+w,y+h), centroids[i])
            results.append(r)
    
    return results


# grab frames from video and make prediction measuring distances of detected people

# setting command line args

# --input: optional video file, if not provided computer's web cam will be used
# --output: optional path to output video file
# --display: by default will show output on screen

# ap = argparse.ArgumentParser()

# ap.add_argument("-o", "--output", type=str, default="",
# 	help="path to (optional) output video file")

# ap.add_argument("-d", "--display", type=int, default=1,
# 	help="whether or not output frame should be displayed")

# args = vars(ap.parse_args(["--input","/Users/pranjalbhadu/Documents/smart-workplace/social-distancing/vtest.mp4","--output","my_output.avi","--display","1"]))

# load the COCO class labels out YOLO model was trained on

def social_distancing_func():

    labelsPath = os.path.sep.join(["/Users/pranjalbhadu/Documents/smart-workplace/social_distancing_model/yolo-coco/coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration

    weightsPath = os.path.sep.join(["/Users/pranjalbhadu/Documents/smart-workplace/social_distancing_model/yolo-coco/yolov3.weights"])
    configPath = os.path.sep.join(["/Users/pranjalbhadu/Documents/smart-workplace/social_distancing_model/yolo-coco/yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # determine only the *output* layer names that we need from YOLO

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream and pointer to output video file

    # vs = cv2.VideoCapture("/Users/pranjalbhadu/Documents/smart-workplace/social_distancing_model/vtest.mp4")
    vs = cv2.VideoCapture(0)
    writer = None

    # loop over frame from video capture

    while True:
        # read next frame from the file

        (grabbed, frame) = vs.read();

        if not grabbed:
            break

        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the minimum social distance

        violate = set()

        # ensure that there are minimum two detections and

        if len(results) >= 2:
            # extract centroids and compute euclidean distance
            centroids = np.array([r[2] for r in results])
            d = dist.cdist(centroids, centroids, metric='euclidean')

            for i in range(0, d.shape[0]):
                for j in range(i+1, d.shape[1]):
                    if d[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)
        print(len(violate))
        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            (startx, starty, endx, endy) = bbox
            (cx, cy) = centroid
            color = (0, 255, 0)

            # if index pair exist in violation, update color

            if i in violate: 
                color = (0, 0, 255)
            
            # draw bounding box and centroid

            cv2.rectangle(frame, (startx, starty), (endx, endy), color, 5, 1)

        # show total violations

        text = "Social Distancing Violations: {}".format(len(violate))

        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        # cv2.imshow('video',frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    