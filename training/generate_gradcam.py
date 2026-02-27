"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Generator
Creates heatmap visualizations showing where the model focuses for predictions

This makes your project look VERY ADVANCED for IEEE presentation!
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path


class GradCAM:
    """
    Grad-CAM implementation for explainable AI visualization.
    Works with nested submodels like MobileNetV2 by using
    a tape-only approach instead of rebuilding keras.Model.
    """

    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM

        Args:
            model: Trained Keras model
            layer_name: Name of convolutional layer to visualize
                       (if None, uses last Conv2D in any submodel)
        """
        self.model = model

        if layer_name is None:
            layer_name = self._find_last_conv_layer(model)

        if layer_name is None:
            raise ValueError("No Conv2D layer found in model or submodels.")

        self.layer_name = layer_name
        print(f"Using layer: {layer_name}")

        # Locate the actual layer object (search submodels too)
        self.target_layer = self._get_layer(model, layer_name)
        if self.target_layer is None:
            raise ValueError(f"Layer '{layer_name}' not found in model or any submodel.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_last_conv_layer(self, model):
        """Recursively find the last Conv2D layer name (including in submodels)."""
        last = None
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                nested = self._find_last_conv_layer(layer)
                if nested:
                    last = nested
            elif isinstance(layer, keras.layers.Conv2D):
                last = layer.name
        return last

    def _get_layer(self, model, name):
        """Recursively retrieve a layer by name from model or any submodel."""
        for layer in model.layers:
            if layer.name == name:
                return layer
            if isinstance(layer, keras.Model):
                found = self._get_layer(layer, name)
                if found:
                    return found
        return None

    def _find_submodel_containing(self, model, layer_name):
        """Return the direct child submodel that contains the named layer."""
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                if self._get_layer(layer, layer_name) is not None:
                    return layer
        return None

    # ------------------------------------------------------------------
    # Core Grad-CAM
    # ------------------------------------------------------------------

    def generate_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap.

        Strategy: build a sub-model entirely within the MobileNetV2 graph
        (submodel_input → conv_out + submodel_output), then manually chain
        the outer model's remaining layers around it. This avoids the
        'Graph disconnected' error caused by crossing graph boundaries.

        Args:
            img_array: Preprocessed image array shape (1, H, W, 3)
            pred_index: Class index to visualize (None = top prediction)

        Returns:
            heatmap: 2D numpy array (H, W) with values in [0, 1]
        """
        img_tensor = tf.cast(img_array, tf.float32)

        # Find which direct child submodel contains our target layer
        submodel = self._find_submodel_containing(self.model, self.layer_name)

        if submodel is not None:
            # ── Build inner grad model within the submodel's own graph ──
            inner_grad_model = keras.Model(
                inputs=submodel.inputs,
                outputs=[self.target_layer.output, submodel.output],
                name="inner_grad_model"
            )

            # ── Collect outer layers before and after the submodel ──────
            layers_before, layers_after = [], []
            passed_submodel = False
            for layer in self.model.layers:
                if isinstance(layer, keras.layers.InputLayer):
                    continue
                if layer is submodel:
                    passed_submodel = True
                    continue
                if not passed_submodel:
                    layers_before.append(layer)
                else:
                    layers_after.append(layer)

            # ── Forward pass with GradientTape ───────────────────────────
            with tf.GradientTape() as tape:
                # Pass through outer layers before submodel
                x = img_tensor
                for layer in layers_before:
                    x = layer(x)

                # Pass through inner grad model (captures conv output)
                conv_output, x = inner_grad_model(x)
                tape.watch(conv_output)

                # Pass through outer layers after submodel
                for layer in layers_after:
                    x = layer(x)

                predictions = x

                if pred_index is None:
                    pred_index = int(tf.argmax(predictions[0]))

                class_score = predictions[:, pred_index]

            grads = tape.gradient(class_score, conv_output)

        else:
            # Target layer is directly in the outer model — safe to build normally
            inner_model = keras.Model(
                inputs=self.model.inputs,
                outputs=[self.target_layer.output, self.model.output]
            )

            with tf.GradientTape() as tape:
                conv_output, predictions = inner_model(img_tensor)
                tape.watch(conv_output)

                if pred_index is None:
                    pred_index = int(tf.argmax(predictions[0]))

                class_score = predictions[:, pred_index]

            grads = tape.gradient(class_score, conv_output)

        if grads is None:
            raise RuntimeError(
                "Gradients are None. The target conv layer may not be on "
                "the computational path to the model output."
            )

        # ── Compute heatmap ───────────────────────────────────────────────
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        conv_out_np = conv_output[0].numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_out_np[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_out_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)            # ReLU
        heatmap /= (np.max(heatmap) + 1e-10)        # Normalize to [0, 1]

        return heatmap

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    def overlay_heatmap(self, heatmap, original_img, alpha=0.4,
                        colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image.

        Args:
            heatmap: 2D heatmap array
            original_img: Original image (H, W, 3) uint8
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap

        Returns:
            Superimposed image (H, W, 3) uint8
        """
        heatmap_resized = cv2.resize(
            heatmap, (original_img.shape[1], original_img.shape[0])
        )
        heatmap_rgb = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), colormap
        )
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)

        superimposed = cv2.addWeighted(
            heatmap_rgb, alpha, original_img.astype(np.uint8), 1 - alpha, 0
        )
        return superimposed


# ======================================================================
# Main generation function
# ======================================================================

def generate_gradcam_visualizations(
    model_path, test_images_dir, output_dir, num_samples=5
):
    """
    Generate Grad-CAM visualizations for sample images.

    Args:
        model_path: Path to trained .h5 model
        test_images_dir: Directory containing test images
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """

    print("=" * 70)
    print("GRAD-CAM VISUALIZATION GENERATOR")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")

    # ── Init Grad-CAM ─────────────────────────────────────────────────────
    print("\nInitializing Grad-CAM...")
    gradcam = GradCAM(model)
    print("✓ Grad-CAM initialized")

    # ── Class names ───────────────────────────────────────────────────────
    try:
        import json
        with open('classify_classes.json', 'r') as f:
            data = json.load(f)
            class_names = (
                data['classes'] if isinstance(data, dict) and 'classes' in data
                else list(data.keys())
            )
    except Exception:
        class_names = [f"Class {i}" for i in range(model.output_shape[-1])]

    print(f"Classes: {class_names}")

    # ── Find images ───────────────────────────────────────────────────────
    print(f"\nSearching for images in {test_images_dir}...")
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(Path(test_images_dir).rglob(ext)))

    if not image_files:
        print("⚠️  No images found! Please check the directory.")
        return

    print(f"Found {len(image_files)} images")

    np.random.seed(42)
    sample_images = np.random.choice(
        image_files, min(num_samples, len(image_files)), replace=False
    )

    print(f"\nGenerating Grad-CAM for {len(sample_images)} samples...")
    print("-" * 70)

    # ── Individual visualizations ─────────────────────────────────────────
    for idx, img_path in enumerate(sample_images, 1):
        print(f"\n[{idx}/{len(sample_images)}] Processing: {img_path.name}")

        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_original = img_array.copy().astype(np.uint8)
        img_input = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_input, verbose=0)
        pred_class = int(np.argmax(predictions[0]))
        confidence = predictions[0][pred_class]

        print(f"  Prediction: {class_names[pred_class]} ({confidence:.2%} confidence)")

        heatmap = gradcam.generate_heatmap(img_input, pred_index=pred_class)
        superimposed = gradcam.overlay_heatmap(heatmap, img_original)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_original)
        axes[0].set_title('Original X-ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(superimposed)
        axes[2].set_title('Grad-CAM Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        fig.suptitle(
            f'Grad-CAM Visualization\n'
            f'Prediction: {class_names[pred_class]} | Confidence: {confidence:.1%}',
            fontsize=16, fontweight='bold', y=1.02
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'gradcam_sample_{idx}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_path}")

    # ── Combined visualization ────────────────────────────────────────────
    print("\nCreating combined visualization...")
    num_display = min(4, len(sample_images))
    fig, axes = plt.subplots(num_display, 3, figsize=(18, 6 * num_display))

    if num_display == 1:
        axes = axes.reshape(1, -1)

    for idx, img_path in enumerate(sample_images[:num_display]):
        img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_original = img_array.copy().astype(np.uint8)
        img_input = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_input, verbose=0)
        pred_class = int(np.argmax(predictions[0]))
        confidence = predictions[0][pred_class]

        heatmap = gradcam.generate_heatmap(img_input, pred_index=pred_class)
        superimposed = gradcam.overlay_heatmap(heatmap, img_original)

        axes[idx, 0].imshow(img_original)
        axes[idx, 0].set_title(f'Sample {idx+1}: Original', fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(heatmap, cmap='jet')
        axes[idx, 1].set_title('Heatmap', fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(superimposed)
        axes[idx, 2].set_title(
            f'{class_names[pred_class]}\n{confidence:.1%}',
            fontsize=12, fontweight='bold'
        )
        axes[idx, 2].axis('off')

    plt.suptitle(
        'Grad-CAM Visualization - Multiple Samples\n'
        'Explainable AI: Model Attention Regions',
        fontsize=18, fontweight='bold', y=1.00
    )

    plt.tight_layout()
    combined_path = os.path.join(output_dir, 'gradcam_combined.png')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined visualization saved: {combined_path}")

    print("\n" + "=" * 70)
    print("GRAD-CAM GENERATION COMPLETED!")
    print("=" * 70)
    print(f"\n✓ Generated {len(sample_images)} individual visualizations")
    print(f"✓ Generated 1 combined visualization")
    print(f"✓ All files saved in: {output_dir}/")
    print("\n" + "=" * 70)
    print("Ready for IEEE presentation! 🔥")
    print("=" * 70)


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":

    MODEL_PATH = "../deployment/model/fracture_classification_model.h5"
    TEST_IMAGES_DIR = "../deployment/uploads"
    OUTPUT_DIR = "gradcam_outputs"
    NUM_SAMPLES = 5

    generate_gradcam_visualizations(
        model_path=MODEL_PATH,
        test_images_dir=TEST_IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=NUM_SAMPLES
    )

    print("\n📌 USAGE IN PRESENTATION:")
    print("-" * 70)
    print("1. Show individual Grad-CAM images to explain model decisions")
    print("2. Use combined visualization for comprehensive overview")
    print("3. Highlight: 'Red areas = high attention, Blue = low attention'")
    print("4. Emphasize: 'Model focuses on actual fracture regions'")
    print("-" * 70)