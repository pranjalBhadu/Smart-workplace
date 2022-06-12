import numpy as numpy
import time
import imutils
import cv2 
import os
dirname = os.path.dirname(__file__)

def in_out_func():
    # video = cv2.VideoCapture('/Users/pranjalbhadu/Documents/smart-workplace/person-count/people-capture.mp4')
    video = cv2.VideoCapture(0)
    print("yes")
    avg = None

    xvalues = list() # stores the x values of all the contours formed
    motion = list() # finds the direction of motion i.e. in or out

    count1 = 0 # in count
    count2 = 0 # out count

    def find_majority(k):
        mp={}
        maximum = ('', 0)

        for n in k:
            if n in mp:
                mp[n]+=1
            else:
                mp[n] = 1
            
            if mp[n] > maximum[1]:
                maximum = (n, mp[n])
        
        return maximum


    while 1:
        ret, frame = video.read()

        flag = True

        frame = cv2.resize(frame, (500,500))
        # converted to grayscale to increase the accuracy of feature detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # bluring the image smoothens the image and motion detection becomes easier
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        # if statement to check if it is the first frame or not because we need the first frame as our reference frame
        # motion is detected from a reference point
        if avg is None:
            avg = gray.copy().astype(float)
            continue
        # background subtraction: seprating foreground elements from the background, done by generating foreground mask
        # running average: used to achieve the above mentioned task
        # running avg is calculated over current frame and previous frame
        # parameters: src, dest, alpha-how fast accumulator forgets about previous images
        cv2.accumulateWeighted(gray, avg, 0.5)
        # difference of current frame from the reference or first frame
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        # threshhold is a limit upto which we want the motion to be detected as we don't want noises to be detected
        # parameters: delta frame, intensity, color, object
        thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
        # add another layer of smoothening
        # no. of iterations define how accurate the smoothening is. increasing them to more than required also increases the noises
        thresh = cv2.dilate(thresh, None, iterations=2)
        # contours are the points where the motion is happening
        # the frame is still and the object is moving so object is the contour
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # define what particular size the contour is then the area is said to be  motion
        # if this is not done even a small noise would be considered as motion
        for c in cnts:
            if cv2.contourArea(c)<5000:
                continue
            (x,y,w,h) = cv2.boundingRect(c)
            xvalues.append(x)
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,255), 2)
            flag = False
        
        # no. of contours
        no_x = len(xvalues) 
        # checks the direction of the motion 
        # if diff < 0 then person is leaving the room 
        # else the person is entering the room
        if no_x > 2:
            difference = xvalues[no_x-1] - xvalues[no_x-2]
            if(difference>0):
                motion.append(1)
            else:
                motion.append(0)
        
        if flag is True:
            if no_x > 5:
                val, times = find_majority(motion)
                if val==1 and times>15:
                    count1+=1
                else:
                    count2+=1
            xvalues = list()
            motion = list()
            

        # displays the boundary line

        cv2.line(frame, (260,0), (260, 480), (0,255,0), 2)

        cv2.line(frame, (420, 0), (420, 480), (0, 255, 0), 2)

        # displays the in and out count
        
        cv2.putText(frame, "IN: {}".format(count1), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(frame, "OUT: {}".format(count2), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # displays the screen or frame

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')