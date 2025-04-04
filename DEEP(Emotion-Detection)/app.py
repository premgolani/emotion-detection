from flask import Flask, render_template, request, redirect, url_for, Response  
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = tf.keras.models.load_model('emotion_model(1).h5')
CLASS_NAMES = ['Angry', 'Happy', 'Sad', 'Neutral']

# Preprocess image for model input
def preprocess_image(img):
    img = img.convert('L').resize((48, 48))  # Convert to grayscale and resize
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=(0, -1))  # Adding batch and channel dimensions

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Image emotion detection
@app.route('/image', methods=['GET', 'POST'])
def image_detection():
    if request.method == 'POST':
        # Handle file upload
        if 'image' not in request.files:
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        
        # Save the uploaded file
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)
        
        # Preprocess and predict
        img = Image.open(file.stream)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        emotion = CLASS_NAMES[np.argmax(prediction)]
        confidence = f"{np.max(prediction) * 100:.2f}%"
        
        return render_template('image_result.html', 
                             emotion=emotion, 
                             confidence=confidence, 
                             img_path=img_path)
    
    return render_template('image_result.html')

# Real-time webcam emotion detection
@app.route('/live')
def live_detection():
    return render_template('live.html')

# Video feed for real-time detection
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Generate frames for real-time detection
def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocess frame for prediction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized / 255.0
            input_tensor = np.expand_dims(normalized, axis=(0, -1))
            
            # Predict emotion
            prediction = model.predict(input_tensor)
            emotion = CLASS_NAMES[np.argmax(prediction)]
            
            # Display emotion on the frame
            cv2.putText(frame, f"Emotion: {emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)