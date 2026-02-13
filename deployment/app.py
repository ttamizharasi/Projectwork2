from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from utils.predict import FracturePredictor

# ---------------- Flask Setup ----------------
app = Flask(__name__)

# Render/Docker writable temp directory
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -------- Load AI Predictor --------
predictor = FracturePredictor()


# ---------------- File Validation ----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------- Treatment Engine ----------------
def get_treatment_recommendation(fracture_type, severity):

    treatment_map = {

        'No Fracture': {
            'primary': 'No fracture detected',
            'secondary': 'Rest and monitor pain',
            'duration': 'N/A',
            'follow_up': 'Consult doctor if symptoms persist'
        },

        'elbow fracture': {
            'primary': 'Immobilization using elbow splint or cast',
            'secondary': 'Pain management and physiotherapy',
            'duration': '6–8 weeks',
            'follow_up': 'X-ray review every 2 weeks'
        },

        'fingers fracture': {
            'primary': 'Finger splint or buddy taping',
            'secondary': 'Ice therapy and elevation',
            'duration': '4–6 weeks',
            'follow_up': 'Weekly monitoring'
        },

        'forearm fracture': {
            'primary': 'Cast immobilization',
            'secondary': 'Elevation and anti-inflammatory medication',
            'duration': '6–10 weeks',
            'follow_up': 'Radiographic evaluation every 2 weeks'
        },

        'humerus fracture': {
            'primary': 'Arm sling or surgical fixation depending on severity',
            'secondary': 'Physical rehabilitation therapy',
            'duration': '8–12 weeks',
            'follow_up': 'Orthopedic consultation'
        },

        'shoulder fracture': {
            'primary': 'Shoulder immobilization or surgical repair',
            'secondary': 'Rehabilitation exercises',
            'duration': '8–12 weeks',
            'follow_up': 'Regular physiotherapy review'
        },

        'wrist fracture': {
            'primary': 'Wrist casting or surgical fixation',
            'secondary': 'Hand therapy exercises',
            'duration': '6–8 weeks',
            'follow_up': 'X-ray follow-up after healing'
        }
    }

    return treatment_map.get(fracture_type, {
        'primary': 'Consult orthopedic specialist',
        'secondary': 'Immobilization and pain management',
        'duration': 'Depends on clinical evaluation',
        'follow_up': 'Doctor guided'
    })


# ---------------- Routes ----------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # -------- AI Prediction --------
        prediction = predictor.predict(filepath)

        treatment = get_treatment_recommendation(
            prediction['fracture_type'],
            prediction['severity']
        )

        return jsonify({
            'success': True,
            'fracture_type': prediction['fracture_type'],
            'severity': prediction['severity'],
            'confidence': prediction['confidence'],
            'treatment': treatment,
            'image_path': filepath
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():

    return jsonify({
        'model_name': 'FractureSense AI Dual Model',
        'model_type': 'MobileNetV2 Transfer Learning',
        'classes': predictor.get_classes(),
        'description': 'Multi-stage fracture detection and classification AI'
    })


# ---------------- Run Server ----------------
if __name__ == '__main__':

    print("\n🏥 FractureSense AI Started")

    # Render provides PORT dynamically
    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)
