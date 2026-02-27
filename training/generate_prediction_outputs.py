"""
Sample Prediction Output Generator
Creates professional prediction result images for PPT presentation
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import json
from pathlib import Path
from datetime import datetime

class PredictionVisualizer:
    """
    Generate professional prediction output visualizations
    """
    
    def __init__(self, detection_model_path, classification_model_path, class_json_path):
        """
        Initialize with both models
        """
        print("Loading models...")
        self.detection_model = keras.models.load_model(detection_model_path, compile=False)
        self.classification_model = keras.models.load_model(classification_model_path)
        
        # Load class names
        with open(class_json_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'classes' in data:
                self.class_names = data['classes']
            else:
                self.class_names = list(data.keys())
        
        print(f"✓ Models loaded successfully")
        print(f"✓ Classes: {self.class_names}")
    
    def create_prediction_card(self, image_path, output_path):
        """
        Create a professional prediction result card
        """
        # Load image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for classification
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        predictions = self.classification_model.predict(img_batch, verbose=0)[0]
        pred_class = np.argmax(predictions)
        confidence = predictions[pred_class] * 100
        
        # Determine severity
        fracture_type = self.class_names[pred_class]
        if confidence > 90:
            confidence_level = "Very High"
            confidence_color = '#10b981'
        elif confidence > 75:
            confidence_level = "High"
            confidence_color = '#3b82f6'
        elif confidence > 60:
            confidence_level = "Moderate"
            confidence_color = '#f59e0b'
        else:
            confidence_level = "Low"
            confidence_color = '#ef4444'
        
        # Determine severity based on fracture type
        if 'No Fracture' in fracture_type or 'Normal' in fracture_type:
            severity = "None"
            severity_color = '#10b981'
        elif 'Hairline' in fracture_type or 'Greenstick' in fracture_type:
            severity = "Minor"
            severity_color = '#fbbf24'
        elif 'Simple' in fracture_type or 'Spiral' in fracture_type:
            severity = "Moderate"
            severity_color = '#f59e0b'
        else:
            severity = "Severe"
            severity_color = '#ef4444'
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Original X-ray (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_rgb)
        ax1.set_title('Input X-ray Image', fontsize=16, fontweight='bold', pad=15)
        ax1.axis('off')
        
        # Add image info
        h, w = img_rgb.shape[:2]
        ax1.text(0.5, -0.05, f'Resolution: {w}×{h} pixels', 
                transform=ax1.transAxes, ha='center', fontsize=10, 
                color='gray')
        
        # 2. AI Analysis Results (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        # Title
        ax2.text(0.5, 0.95, 'AI ANALYSIS RESULTS', 
                ha='center', va='top', fontsize=18, fontweight='bold',
                transform=ax2.transAxes)
        
        # Prediction box
        ax2.add_patch(patches.FancyBboxPatch(
            (0.05, 0.65), 0.9, 0.25,
            boxstyle="round,pad=0.02", 
            edgecolor='#2563eb', facecolor='#eff6ff',
            linewidth=2, transform=ax2.transAxes
        ))
        
        ax2.text(0.5, 0.83, 'FRACTURE TYPE', 
                ha='center', fontsize=12, fontweight='bold',
                color='#1e40af', transform=ax2.transAxes)
        ax2.text(0.5, 0.75, fracture_type, 
                ha='center', fontsize=20, fontweight='bold',
                color='#1f2937', transform=ax2.transAxes)
        
        # Confidence box
        ax2.add_patch(patches.FancyBboxPatch(
            (0.05, 0.35), 0.9, 0.25,
            boxstyle="round,pad=0.02", 
            edgecolor=confidence_color, facecolor='#f9fafb',
            linewidth=2, transform=ax2.transAxes
        ))
        
        ax2.text(0.5, 0.53, 'CONFIDENCE SCORE', 
                ha='center', fontsize=12, fontweight='bold',
                color='#374151', transform=ax2.transAxes)
        ax2.text(0.5, 0.43, f'{confidence:.1f}%', 
                ha='center', fontsize=32, fontweight='bold',
                color=confidence_color, transform=ax2.transAxes)
        
        # Severity box
        ax2.add_patch(patches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.25,
            boxstyle="round,pad=0.02", 
            edgecolor=severity_color, facecolor='#fef3c7',
            linewidth=2, transform=ax2.transAxes
        ))
        
        ax2.text(0.5, 0.23, 'SEVERITY LEVEL', 
                ha='center', fontsize=12, fontweight='bold',
                color='#92400e', transform=ax2.transAxes)
        ax2.text(0.5, 0.13, severity, 
                ha='center', fontsize=20, fontweight='bold',
                color=severity_color, transform=ax2.transAxes)
        
        # 3. Probability Distribution (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Sort predictions for better visualization
        sorted_indices = np.argsort(predictions)[::-1][:5]  # Top 5
        sorted_probs = predictions[sorted_indices] * 100
        sorted_names = [self.class_names[i] for i in sorted_indices]
        
        # Horizontal bar chart
        colors = ['#10b981' if i == pred_class else '#94a3b8' for i in sorted_indices]
        bars = ax3.barh(range(len(sorted_names)), sorted_probs, color=colors, height=0.6)
        
        ax3.set_yticks(range(len(sorted_names)))
        ax3.set_yticklabels(sorted_names, fontsize=11)
        ax3.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Probability Distribution (Top 5)', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlim([0, 100])
        ax3.grid(axis='x', alpha=0.3, linestyle='--')
        ax3.invert_yaxis()
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
            ax3.text(prob + 2, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1f}%', 
                    va='center', fontsize=10, fontweight='bold')
        
        # 4. Treatment Recommendation (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        ax4.text(0.5, 0.95, 'TREATMENT RECOMMENDATION', 
                ha='center', va='top', fontsize=16, fontweight='bold',
                color='#1f2937', transform=ax4.transAxes)
        
        # Get treatment recommendation
        treatment = self.get_treatment(fracture_type, severity)
        
        # Primary treatment
        ax4.add_patch(patches.FancyBboxPatch(
            (0.05, 0.70), 0.9, 0.20,
            boxstyle="round,pad=0.02", 
            edgecolor='#3b82f6', facecolor='#eff6ff',
            linewidth=2, transform=ax4.transAxes
        ))
        ax4.text(0.1, 0.85, '🏥 Primary Treatment:', 
                fontsize=11, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.77, treatment['primary'], 
                fontsize=10, wrap=True, transform=ax4.transAxes)
        
        # Duration
        ax4.add_patch(patches.FancyBboxPatch(
            (0.05, 0.45), 0.9, 0.20,
            boxstyle="round,pad=0.02", 
            edgecolor='#f59e0b', facecolor='#fef3c7',
            linewidth=2, transform=ax4.transAxes
        ))
        ax4.text(0.1, 0.60, '⏱️ Expected Duration:', 
                fontsize=11, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.52, treatment['duration'], 
                fontsize=10, transform=ax4.transAxes)
        
        # Follow-up
        ax4.add_patch(patches.FancyBboxPatch(
            (0.05, 0.20), 0.9, 0.20,
            boxstyle="round,pad=0.02", 
            edgecolor='#10b981', facecolor='#f0fdf4',
            linewidth=2, transform=ax4.transAxes
        ))
        ax4.text(0.1, 0.35, '📅 Follow-up Schedule:', 
                fontsize=11, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, 0.27, treatment['follow_up'], 
                fontsize=10, transform=ax4.transAxes)
        
        # Disclaimer
        ax4.text(0.5, 0.05, 
                '⚠️ For informational purposes only. Consult a medical professional.',
                ha='center', fontsize=8, color='#6b7280', style='italic',
                transform=ax4.transAxes)
        
        # Overall title
        fig.suptitle('FractureSense AI - Prediction Output', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                ha='right', fontsize=8, color='gray')
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return {
            'fracture_type': fracture_type,
            'confidence': confidence,
            'severity': severity
        }
    
    def get_treatment(self, fracture_type, severity):
        """
        Get treatment recommendation based on fracture type and severity
        """
        treatments = {
            ('No Fracture', 'None'): {
                'primary': 'No treatment needed',
                'duration': 'N/A',
                'follow_up': 'Only if symptoms develop'
            },
            ('Hairline Fracture', 'Minor'): {
                'primary': 'Rest and immobilization with splint',
                'duration': '4-6 weeks',
                'follow_up': 'X-ray at 2 weeks'
            },
            ('Simple Fracture', 'Moderate'): {
                'primary': 'Cast immobilization, elevation',
                'duration': '6-8 weeks',
                'follow_up': 'Every 2 weeks'
            },
            ('Comminuted Fracture', 'Severe'): {
                'primary': 'Surgical intervention required',
                'duration': '12-20 weeks',
                'follow_up': 'Bi-weekly for 3 months'
            },
            ('Compound Fracture', 'Severe'): {
                'primary': 'Emergency surgery, antibiotics',
                'duration': '12-16 weeks',
                'follow_up': 'Intensive monitoring'
            },
        }
        
        # Find matching treatment
        key = (fracture_type, severity)
        if key in treatments:
            return treatments[key]
        
        # Default treatment
        return {
            'primary': 'Consult orthopedic specialist',
            'duration': 'Varies based on assessment',
            'follow_up': 'As directed by physician'
        }


def generate_sample_predictions(model_dir, test_images_dir, output_dir, num_samples=5):
    """
    Generate sample prediction outputs for presentation
    """
    print("=" * 70)
    print("SAMPLE PREDICTION OUTPUT GENERATOR")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    detection_model = os.path.join(model_dir, 'fracture_detection_model.h5')
    classification_model = os.path.join(model_dir, 'fracture_classification_model.h5')
    class_json = os.path.join(model_dir, 'classify_classes.json')
    
    visualizer = PredictionVisualizer(detection_model, classification_model, class_json)
    
    # Find test images
    print(f"\nSearching for test images in {test_images_dir}...")
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(test_images_dir).rglob(ext)))
    
    if not image_files:
        print("⚠️  No images found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Select diverse samples
    np.random.seed(42)
    sample_images = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    print(f"\nGenerating {len(sample_images)} prediction outputs...")
    print("-" * 70)
    
    results = []
    for idx, img_path in enumerate(sample_images, 1):
        print(f"\n[{idx}/{len(sample_images)}] Processing: {img_path.name}")
        
        output_path = os.path.join(output_dir, f'prediction_output_{idx}.png')
        result = visualizer.create_prediction_card(img_path, output_path)
        results.append(result)
        
        print(f"  Fracture: {result['fracture_type']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Severity: {result['severity']}")
        print(f"  ✓ Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTION GENERATION COMPLETED!")
    print("=" * 70)
    print(f"\n✓ Generated {len(results)} prediction outputs")
    print(f"✓ All files saved in: {output_dir}/")
    
    print("\n📌 USAGE IN PRESENTATION:")
    print("-" * 70)
    print("1. Use these as 'Output & Results' slide images")
    print("2. Show different cases: No Fracture, Minor, Moderate, Severe")
    print("3. Highlight: Confidence scores, Treatment recommendations")
    print("4. Emphasize: Professional medical dashboard-style output")
    print("-" * 70)


if __name__ == "__main__":
    """
    Usage Example
    """
    
    # Configuration
    MODEL_DIR = "../deployment/model"
    TEST_IMAGES_DIR = "../deployment/uploads"
    OUTPUT_DIR = "prediction_outputs"
    NUM_SAMPLES = 5
    
    # Generate sample predictions
    generate_sample_predictions(
        model_dir=MODEL_DIR,
        test_images_dir=TEST_IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES
    )