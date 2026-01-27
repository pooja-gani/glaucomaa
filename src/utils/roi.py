import cv2
import numpy as np

def extract_disc_roi(image_rgb, crop_size=512):
    """
    Extracts a square ROI around the brightest area (likely Optic Disc).
    Returns: cropped_image, (start_y, start_x)
    """
    # Extract Red Channel (Optic Disc is brightest in Red)
    red_channel = image_rgb[:, :, 0]
    
    # Heavy Gaussian Blur to remove vessels and noise, leaving the "blob" of the disc
    # Kernel size 41 or higher is good for general localization
    blurred = cv2.GaussianBlur(red_channel, (41, 41), 0)
    
    # Find brightest spot in the blurred red channel
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
    
    # Center of ROI = maxLoc
    c_x, c_y = maxLoc
    
    h, w, _ = image_rgb.shape
    half = crop_size // 2
    
    start_x = max(0, c_x - half)
    start_y = max(0, c_y - half)
    
    end_x = min(w, start_x + crop_size)
    end_y = min(h, start_y + crop_size)
    
    # Adjust if out of bounds (keep size if possible)
    if end_x - start_x < crop_size:
        if start_x == 0:
            end_x = min(w, crop_size)
        else:
            start_x = max(0, w - crop_size)
            
    if end_y - start_y < crop_size:
        if start_y == 0:
            end_y = min(h, crop_size)
        else:
            start_y = max(0, h - crop_size)
            
    crop = image_rgb[start_y:end_y, start_x:end_x]
    
    # Ensure it's exactly crop_size x crop_size (padding if image is too small)
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size))
        
    return crop, (start_x, start_y)
