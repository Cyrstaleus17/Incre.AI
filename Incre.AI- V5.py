import cv2
import numpy as np
import glob
import os
import shutil

def analyze_image(image):
    """
    Analyzes the image to determine its quality metrics, including contrast and noise.

    Args:
        image: The input image (NumPy array).

    Returns:
        A dictionary containing various quality metrics.
    """
    # Ensure image is 8-bit for analysis
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # --- Basic Metrics ---
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    brightness = np.mean(v)
    saturation = np.mean(s)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    contrast = gray_image.std()

    # --- Noise Estimation ---
    # Estimates noise by checking the variance of the laplacian in a denoised version
    denoised_for_analysis = cv2.medianBlur(gray_image, 3)
    noise = cv2.Laplacian(denoised_for_analysis, cv2.CV_64F).var()
    noise_level = (sharpness - noise) / (sharpness + 1e-6) # Ratio of detail to noise
    
    # --- Image Type Classification (Heuristic) ---
    # Digital art tends to have fewer, more uniform colors and sharper edges.
    unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
    color_density = unique_colors / (image.shape[0] * image.shape[1])
    
    image_type = 'photograph'
    if color_density < 0.3 and sharpness > 150:
         image_type = 'digital_art'

    return {
        "brightness": brightness, 
        "saturation": saturation, 
        "sharpness": sharpness,
        "contrast": contrast,
        "noise_level": noise_level,
        "type": image_type
    }


def one_click_enhance(input_image_path, output_image_path, allow_upscaling=True, force_enhance=False):
    """
    Analyzes and enhances an image using type-specific logic (photo vs. art).

    Args:
        input_image_path (str): The path to the input image file.
        output_image_path (str): The path to save the enhanced image file.
        allow_upscaling (bool): Whether to allow automatic upscaling.
        force_enhance (bool): If True, bypasses the high-quality check.
    """
    try:
        # Step 1: Read the image, handling transparency
        image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not read image: {input_image_path}")
            return
            
        # Separate alpha channel if it exists
        alpha_channel = None
        if image.shape[2] == 4:
            alpha_channel = image[:, :, 3]
            image = image[:, :, :3]

        print("Original image loaded. Analyzing...")
        
        # --- Image Analysis ---
        metrics = analyze_image(image)
        print(f"Image Metrics: Type={metrics['type']}, Sharpness={metrics['sharpness']:.2f}, Noise={metrics['noise_level']:.2f}")

        # --- High-Quality Pre-Check ---
        is_high_quality = (metrics['sharpness'] > 250 and metrics['noise_level'] < 0.2)
        if is_high_quality and not force_enhance:
            print("Image already appears high quality. Skipping enhancements.")
            shutil.copy(input_image_path, output_image_path)
            print(f"Enhancement complete (original copied). Saved as '{output_image_path}'")
            return

        enhanced_image = image.copy()

        # --- Adaptive Upscaling ---
        height, width, _ = enhanced_image.shape
        if allow_upscaling and (height < 720 or width < 720):
            print(f"Image is low resolution. Upscaling by 2x...")
            enhanced_image = cv2.resize(enhanced_image, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)

        # --- Type-Specific Enhancement Logic ---
        if metrics['type'] == 'photograph':
            print("Applying enhancement logic for PHOTOGRAPH.")
            # Denoise photos if they are noisy
            if metrics['noise_level'] > 0.3:
                print("High noise detected. Applying photo denoiser...")
                enhanced_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
            
            # Gentle contrast for photos
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))

        else: # digital_art
            print("Applying enhancement logic for DIGITAL ART.")
            # Gentle smoothing only for very soft art
            if metrics['sharpness'] < 80:
                print("Applying edge-preserving smoothing...")
                enhanced_image = cv2.bilateralFilter(enhanced_image, d=5, sigmaColor=50, sigmaSpace=50)
            
            # More subtle contrast for art
            clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))

        # --- Universal Adjustments ---
        # Contrast (applied to L channel)
        if 50 < metrics['brightness'] < 210:
             print("Adjusting local contrast...")
             lab_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
             l, a, b = cv2.split(lab_image)
             l = clahe.apply(l)
             enhanced_image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        
        # Saturation
        if metrics['saturation'] < 180:
            print("Boosting color saturation...")
            hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s * 1.15, 0, 255).astype(np.uint8)
            enhanced_image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

        # Sharpening
        if metrics['sharpness'] < 200:
             print("Applying sharpening filter...")
             kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
             enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)

        # --- Final Output ---
        # Re-attach alpha channel if it existed
        if alpha_channel is not None:
            # Ensure alpha channel is the same size if upscaling occurred
            if alpha_channel.shape[:2] != enhanced_image.shape[:2]:
                alpha_channel = cv2.resize(alpha_channel, (enhanced_image.shape[1], enhanced_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            enhanced_image = cv2.merge([enhanced_image, alpha_channel])

        cv2.imwrite(output_image_path, enhanced_image)
        print(f"Enhancement complete! Saved as '{output_image_path}'")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    # Process all user-provided JPG, JPEG, and PNG images in the folder.
    target_files = glob.glob('*.jpg') + glob.glob('*.jpeg') + glob.glob('*.png')

    for input_file in target_files:
        filename, ext = os.path.splitext(input_file)
        
        # Avoid re-processing already enhanced images
        if "_enhanced" in filename:
            continue
            
        output_file = f"{filename}_enhanced{ext}"
        
        print(f"\n--- Processing {input_file} ---")
        one_click_enhance(input_file, output_file)

