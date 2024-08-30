from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

def process_signature(image, threshold_value):
    signature_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, signature_binary = cv2.threshold(signature_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)

    blue_array = np.zeros_like(image)
    blue_array[:] = (255, 0, 0)  
    signature_colored = cv2.addWeighted(image, 1, blue_array, 0.5, 0)
    
    b, g, r = cv2.split(signature_colored)
    signature_with_alpha = cv2.merge([b, g, r, signature_binary])

    return signature_with_alpha

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['signature']
    np_img = np.frombuffer(file.read(), np.uint8)
    signature = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    thresholds = [100, 125, 150, 175, 200, 225]
    images = []

    for threshold in thresholds:
        processed_image = process_signature(signature, threshold)
        _, buffer = cv2.imencode('.png', processed_image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        images.append({'threshold': threshold, 'image_data': encoded_image})

    return jsonify(images)

if __name__ == '__main__':
    app.run(debug=True)
