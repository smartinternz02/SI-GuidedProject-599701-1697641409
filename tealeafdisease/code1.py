from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the saved model

loaded_model = joblib.load('tealeaf_model.joblib')

from flask import Flask, render_template, request

app = Flask(__name__, template_folder='C:/Users/sruta/OneDrive/Desktop/tealeafdisease')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'No image uploaded!', 400
    
    image = request.files['image']
    # Read and preprocess the uploaded image
    img = Image.open(image)
    img = img.resize((64, 64))  # Resize to match the model's input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Reshape the image to match the model's input shape
    
    # Predict the class probabilities
    probs = loaded_model.predict(img)
    predicted_class = np.argmax(probs, axis=1)
    
    # Define the class names
    class_names = ['Red leaf spot',
'Algal leaf spot',
'Bird eyespot',
'Gray blight',
'White spot',
'Anthracnose',
'Brown blight',
'health'
]
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class[0]]
    
    return f'The predicted class is: {predicted_class_name}'

if __name__ == '__main__':
    app.run()

