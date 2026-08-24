import cv2  # type: ignore
import numpy as np  # type: ignore

def generate_discrepancy_heatmap(image_path, actual_landmarks, discrepancies, displacement_vectors=None, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    h, w, c = image.shape
    
    heatmap = np.zeros_like(image, dtype=np.uint8)
    
    # We will draw arrows directly on a copy of the original image for clarity,
    # or on the heatmap. Let's draw arrows on a separate layer so it's sharp.
    arrow_layer = np.zeros_like(image, dtype=np.uint8)
    
    for i, lm in enumerate(actual_landmarks):
        x_px = int(lm[0] * w)
        y_px = int(lm[1] * h)
        
        disp_mm = discrepancies[i]
        
        disp = np.clip(disp_mm / 3.0, 0.0, 1.0)
        hue = int(120 * (1 - disp))
        
        color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        
        cv2.circle(heatmap, (x_px, y_px), 10, color, -1)
        
        # Draw vector field if available
        if displacement_vectors is not None:
            # displacement_vectors are 3D. We use x and y components.
            # Scale up for visibility (e.g., 5x)
            vec = displacement_vectors[i]
            dx = int(vec[0] * w * 5)
            dy = int(vec[1] * h * 5)
            
            # Draw arrow if magnitude is significant (e.g., > 1mm equivalent)
            if disp_mm > 1.0:
                cv2.arrowedLine(arrow_layer, (x_px, y_px), (x_px + dx, y_px + dy), (255, 255, 255), 1, tipLength=0.3)
        
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    
    image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    # Overlay arrow layer
    # Since arrows are white (255,255,255), we can just use bitwise_or or add
    image = cv2.add(image, arrow_layer)
        
    if output_path:
        cv2.imwrite(output_path, image)
        
    return image
