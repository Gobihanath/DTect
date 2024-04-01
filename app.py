import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model('/Users/sajeeth/Python/flask_pro/Dtect_Confusion_FinalModel.h5')

# Store small box images temporarily as base64 strings
temp_ext_sign = []

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html', ext_signs=temp_ext_sign)

@app.route('/input', methods=['POST'])
def process_image():
    # Get the uploaded image
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Read the uploaded image
    image = cv2.imread(image_path)
    # Resize the image to a specific width and height
    new_width = 300
    new_height = 400
    resized_image = cv2.resize(image, (new_width, new_height))

    # resized image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    largest_box_image = resized_image[y:y + h, x:x + w]

    # Divide the largest_box_image into 10 equal parts
    num_parts = 10
    part_height = h // num_parts

    for i in range(num_parts):
        ext_img = largest_box_image[i * part_height: (i + 1) * part_height, :]

        # Resize small box and encode it in base64
        ext_image = cv2.resize(ext_img, (256, 256))
        _, buffer = cv2.imencode('.jpg', ext_image)
        ext_image_encoded = base64.b64encode(buffer).decode('utf-8')

        # Store the base64 encoded small box image
        temp_ext_sign.append(ext_image_encoded)

    os.remove(image_path)

    return render_template('index.html', ext_signs=temp_ext_sign)

@app.route('/predict', methods=['POST'])
def predict():
    # Process the small box images with the model
    accuracies = []
    for i, ext_image_encoded in enumerate(temp_ext_sign):
        # Decode the base64 encoded small box image
        ext_image_decoded = base64.b64decode(ext_image_encoded)
        ext_image_np = np.frombuffer(ext_image_decoded, np.uint8)
        ext_image = cv2.imdecode(ext_image_np, cv2.IMREAD_COLOR)

        # Preprocess the small box
        small_image = cv2.resize(ext_image, (256, 256))
        ext_image = img_to_array(ext_image)
        ext_image = np.expand_dims(ext_image, axis=0)

        # Make predictions for the small box
        prediction = model.predict(ext_image)

        # Assuming your model has 10 output nodes
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index] * 100
        class_name = f"Person {class_index} with {confidence:.2f}% accuracy"

        # Add signature identification based on the accuracy threshold
        signature_identified = "-Signature Identified" if confidence > 75 else "-Signature Not Identified"

        # Append the correct image number
        accuracies.append(f"Signature {i + 1} : {class_name} {signature_identified}")

    # Clear the temporary small boxes list
    temp_ext_sign.clear()

    return render_template('index.html', predictions=accuracies)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
