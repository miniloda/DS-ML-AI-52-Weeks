from flask import Flask, redirect, request, url_for, render_template, jsonify
import tensorflow as tf
from model_inference import predict

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Render the homepage
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    # Read the uploaded image file
    file = request.files['file']
    
    image_path = 'uploaded_image.jpg'
    file.save(image_path)

    # Make prediction using the loaded model
    prediction = predict(image_path)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
