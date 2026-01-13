import cv2
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your image file
IMAGE_PATH = '132.png'

# Output text file path
OUTPUT_FILE = 'desmos_equations.txt'

# Accuracy of the vision (epsilon factor for contour approximation)
# Smaller value = More equations, Higher accuracy, More detail
# Larger value = Fewer equations, Lower accuracy, More jagged/abstract
# Recommended range: 0.001 (very detailed) to 0.05 (very simplified)
ACCURACY = 0.002 

# Invert y-axis? (Desmos y-axis goes up, images often go down)
# Set True to flip the image so it looks right side up in Desmos
FLIP_Y = True

# Scale factor to fit in Desmos view comfortably (optional)
SCALE = 0.1
# ==========================================

def image_to_desmos_equations(image_path, accuracy):
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Preprocessing
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges using Canny
    # You might need to adjust these threshold values for different images
    edges = cv2.Canny(gray, 100, 200)

    # 3. Find Contours
    # RETR_LIST gets all contours, CHAIN_APPROX_SIMPLE saves memory
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Found {len(contours)} contours. Processing...")

    all_equations = []

    for i, contour in enumerate(contours):
        # 4. Approximate Contour
        # epsilon is the maximum distance from contour to approximated contour
        epsilon = accuracy * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Convert numpy array to simple list of points
        points = approx.reshape(-1, 2)
        
        # If less than 2 points, cannot make a line
        if len(points) < 2:
            continue

        # Iterate through points to create segments
        num_points = len(points)
        for j in range(num_points):
            # Start point
            p1 = points[j]
            # End point (wrap around to 0 for the last point)
            p2 = points[(j + 1) % num_points]
            
            x1 = p1[0] * SCALE
            y1 = p1[1] * SCALE
            
            x2 = p2[0] * SCALE
            y2 = p2[1] * SCALE
            
            if FLIP_Y:
                y1 = -y1
                y2 = -y2
            
            # Calculate deltas for parametric equation: P(t) = P1 + (P2 - P1)t
            dx = x2 - x1
            dy = y2 - y1
            
            # Format: (x1 + dx*t, y1 + dy*t)
            # Using 6 decimal places as per user request example
            eq_str = f"({x1:.6f} + {dx:.6f}t, {y1:.6f} + {dy:.6f}t)"
            all_equations.append(eq_str)

    # 5. Generate Output to File
    try:
        with open(OUTPUT_FILE, 'w') as f:
            for eq in all_equations:
                f.write(eq + "\n")
        
        print("\n" + "="*50)
        print(f"SUCCESS! Equations saved to: {OUTPUT_FILE}")
        print("="*50)
        print(f"Generated {len(all_equations)} separate equations.")
        print(f"Accuracy Level: {accuracy}")
        
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    image_to_desmos_equations(IMAGE_PATH, ACCURACY)
