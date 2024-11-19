from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model
model_fpn = load_model('fpn_model.h5')

def preprocess_image(image_path, size=128):
    """Preprocess the uploaded image"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (size, size))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image/255.
    return image

def array_to_base64(arr):
    """Convert numpy array to base64 string"""
    # Ensure array is 2D
    if len(arr.shape) == 3:
        arr = arr.squeeze()  # Remove single-dimensional entries
    
    # Normalize to 0-255 range and convert to uint8
    arr = (arr * 255).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(arr, mode='L')  # 'L' mode for grayscale
    
    # Save to base64
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Preprocess the image
        processed_image = preprocess_image(filepath)
        
        # Make prediction
        prediction = model_fpn.predict(np.expand_dims(processed_image, 0), verbose=0)[0]
        
        # Convert images to base64 for display
        original_b64 = array_to_base64(processed_image)
        prediction_b64 = array_to_base64(prediction)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'original': original_b64,
            'prediction': prediction_b64
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
