"""
Test Image Generator for FractureSense AI
Generates sample X-ray-like test images for testing the system
"""

from PIL import Image, ImageDraw, ImageFont
import random
import os


def generate_test_xray(filename="test_xray.png", has_fracture=True):
    """
    Generate a simple test X-ray image
    
    Args:
        filename: Output filename
        has_fracture: Whether to draw a fracture line
    """
    # Create a grayscale image (typical X-ray appearance)
    width, height = 400, 400
    img = Image.new('L', (width, height), color=40)  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Draw a bone-like structure (ellipse)
    bone_color = 200  # Light gray (bone appears light in X-rays)
    draw.ellipse([100, 50, 300, 350], fill=bone_color, outline=220)
    
    # Add some texture/noise to make it look more realistic
    for _ in range(1000):
        x = random.randint(0, width)
        y = random.randint(0, height)
        noise = random.randint(30, 60)
        draw.point((x, y), fill=noise)
    
    # Draw fracture if requested
    if has_fracture:
        # Draw a jagged fracture line
        fracture_color = 80  # Darker line
        start_x = random.randint(120, 150)
        start_y = random.randint(100, 150)
        
        # Create zigzag fracture line
        points = [(start_x, start_y)]
        for i in range(8):
            x_offset = random.randint(-15, 15)
            y_offset = random.randint(20, 35)
            new_x = points[-1][0] + x_offset
            new_y = points[-1][1] + y_offset
            points.append((new_x, new_y))
        
        # Draw the fracture
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=fracture_color, width=3)
    
    # Add label
    try:
        label = "TEST X-RAY - " + ("FRACTURE" if has_fracture else "NO FRACTURE")
        draw.text((10, 10), label, fill=220)
    except:
        pass  # Skip if font not available
    
    # Save image
    img.save(filename)
    print(f"✓ Generated test image: {filename}")


def generate_test_suite():
    """
    Generate a complete suite of test images
    """
    # Create test_images folder
    os.makedirs('test_images', exist_ok=True)
    
    print("Generating test X-ray images...")
    print("=" * 50)
    
    # Generate various test cases
    test_cases = [
        ("test_images/no_fracture_1.png", False),
        ("test_images/no_fracture_2.png", False),
        ("test_images/hairline_fracture_1.png", True),
        ("test_images/hairline_fracture_2.png", True),
        ("test_images/simple_fracture_1.png", True),
        ("test_images/simple_fracture_2.png", True),
    ]
    
    for filename, has_fracture in test_cases:
        generate_test_xray(filename, has_fracture)
    
    print("=" * 50)
    print(f"✓ Generated {len(test_cases)} test images")
    print(f"✓ Images saved in 'test_images/' folder")
    print("\nYou can now use these images to test the application!")


if __name__ == "__main__":
    print("=" * 50)
    print("FractureSense AI - Test Image Generator")
    print("=" * 50)
    generate_test_suite()
