from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

app = Flask(__name__)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the first model (NORMAL vs SICK, trained on thermal RGB images)
model1 = models.resnet18(pretrained=False)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, 2)  # Binary classification (NORMAL vs SICK)
model1.load_state_dict(torch.load('best_model101.pth', map_location=device))
model1 = model1.to(device)
model1.eval()

# Load the second model (Malignant vs Benign, trained on 3-channel input)
def load_second_model():
    model2 = models.resnet18(pretrained=False)
    num_ftrs = model2.fc.in_features
    model2.fc = nn.Linear(num_ftrs, 2)  # Binary classification (Malignant vs Benign)
    model2.load_state_dict(torch.load('breast_cancer_model02.pth', map_location=device))
    model2 = model2.to(device)
    model2.eval()
    return model2

# Define transform for the first model (RGB thermal images)
transform_rgb = transforms.Compose([
    transforms.ToTensor()  # Convert RGB image to tensor (3 channels)
])

# Define transform for the second model (grayscale -> 3-channel input)
transform_grayscale_to_rgb = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
    transforms.ToTensor(),  # Convert to tensor (1 channel)
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat grayscale channel to 3 channels
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file temporarily
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        # Open the image
        image = Image.open(file_path).convert('RGB')

        # Preprocess for the first model (RGB)
        image_tensor_rgb = transform_rgb(image).unsqueeze(0).to(device)

        # First model prediction (NORMAL vs SICK)
        with torch.no_grad():
            output1 = model1(image_tensor_rgb)
            print("Model 1 Raw output (RGB):", output1.tolist())  # Debug output
            _, predicted1 = torch.max(output1, 1)
            initial_prediction = 'SICK' if predicted1.item() == 0 else 'NORMAL'

        # If SICK, preprocess for the second model (grayscale -> 3-channel) and predict
        if initial_prediction == 'SICK':
            # Convert to grayscale and then to 3-channel input
            image_tensor_gray_to_rgb = transform_grayscale_to_rgb(image).unsqueeze(0).to(device)

            # Load and evaluate the second model
            model2 = load_second_model()
            with torch.no_grad():
                output2 = model2(image_tensor_gray_to_rgb)
                print("Model 2 Raw output (Grayscale-to-RGB):", output2.tolist())  # Debug output
                _, predicted2 = torch.max(output2, 1)
                final_prediction = 'SICK - Malignant' if predicted2.item() == 0 else 'SICK - Benign'
        else:
            final_prediction = 'NORMAL'

        # Clean up
        os.remove(file_path)

        # Return the result with raw outputs for debugging
        return jsonify({
            'prediction': final_prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)