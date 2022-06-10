from flask import Flask,redirect, url_for,render_template,request,Response
import os
import webserver
from webserver import templates,static
import cv2
from social_distancing_model import social_distancing as sd
from mask_detection import face_mask_and_nose_detection as md
from person_count import in_out_counter as io

secret_key = str(os.urandom(24))

app = Flask(__name__, template_folder='webserver/templates',static_folder='webserver/static')
app.config['TESTING'] = True
app.config['DEBUG'] = True
app.config['FLASK_ENV'] = 'development'
app.config['SECRET_KEY'] = secret_key
app.config['DEBUG'] = True

camera = cv2.VideoCapture(0)

def generate_frames():
    # read camera frame
    # while True:
    #     success, frame = camera.read()

    #     if not success:
    #         break
    #     else:
    #         ret,buffer=cv2.imencode('.jpg',frame)
    #         frame=buffer.tobytes()

    #     yield(b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    yield sd.social_distancing()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/social_distancing')
def social_distancing():
    return render_template('social_distancing.html')

@app.route('/mask_detection')
def mask_detection():
    return render_template('mask_detection.html')

@app.route('/in_out_count')
def in_out_count():
    return render_template('in_out_count.html')

@app.route('/social_distance_video')
def social_distance_video():
    return Response(sd.social_distancing_func(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mask_detection_video')
def mask_detection_video():
    return Response(md.mask_detect_func(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/in_out_video')
def in_out_video():
    return Response(io.in_out_func(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)