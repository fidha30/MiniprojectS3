from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import os
import numpy as np
import base64
from io import BytesIO
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'stable.pt'  # Path to your saved model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DR Stage definitions
DR_STAGES = {
    0: {
        'name': 'No DR',
        'description': 'Healthy image - No signs of diabetic retinopathy',
        'color': '#10B981',  # Green
        'severity': 'Normal',
        'recommendations': 'Continue regular eye checkups annually'
    },
    1: {
        'name': 'Mild DR',
        'description': 'Microaneurysms present - Early stage diabetic retinopathy',
        'color': '#F59E0B',  # Yellow
        'severity': 'Mild',
        'recommendations': 'Schedule follow-up in 6-12 months, maintain good diabetes control'
    },
    2: {
        'name': 'Moderate DR',
        'description': 'More microaneurysms, hemorrhages, and exudates present',
        'color': '#F97316',  # Orange
        'severity': 'Moderate',
        'recommendations': 'Follow-up every 3-6 months, consider referral to retina specialist'
    },
    3: {
        'name': 'Severe DR',
        'description': 'Significant hemorrhages and vascular abnormalities',
        'color': '#EF4444',  # Red
        'severity': 'Severe',
        'recommendations': 'Urgent referral to retina specialist, follow-up within 2-4 weeks'
    },
    4: {
        'name': 'Proliferative DR',
        'description': 'Neovascularization and severe damage - Most advanced stage',
        'color': '#991B1B',  # Dark Red
        'severity': 'Critical',
        'recommendations': 'Immediate referral for treatment, laser therapy or surgery may be needed'
    }
}

# Model Architecture (same as training)
class StableResNet(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.3):
        super().__init__()
        self.backbone = resnet18(pretrained=False)
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Load model
def load_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StableResNet(num_classes=5)
        
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.to(device)
        
        print(f"✅ Model loaded successfully on {device}")
        return model, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# Initialize model
model, device = load_model()

# Image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Predict DR stage from image"""
    if model is None:
        return None
    
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = get_transform()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probs.tolist(),
            'stage_info': DR_STAGES[predicted_class]
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

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
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = f"temp_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Make prediction
            result = predict_image(filepath)
            
            if result is None:
                return jsonify({'error': 'Prediction failed'}), 500
            
            # Convert image to base64 for display
            with open(filepath, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode()
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'prediction': result,
                'image_data': f"data:image/jpeg;base64,{img_base64}"
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'None'
    })

if __name__ == '__main__':
    print("🚀 Starting Diabetic Retinopathy Prediction App...")
    print(f"📱 Model status: {'✅ Loaded' if model else '❌ Failed to load'}")
    print("🌐 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)