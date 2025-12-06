#!/usr/bin/env python3
"""
Boiler Display Icon Recognition - Desktop Calibration Version
Use this on your PC to find correct ROI coordinates using saved camera snapshots
"""

import cv2
import numpy as np
import sys
import os

# ============= CONFIGURATION =============
# Define regions of interest (ROI) - adjust these based on your display
# Format: [y_start, y_end, x_start, x_end]

# These are INITIAL estimates - you need to adjust them!
ROI_FLAME_ICON = [350, 530, 500, 540]     # Left side - flame/droplet icon
ROI_BAR_GRAPH = [350, 530, 540, 580]         # Left side - vertical bar graph
ROI_HEATING_ICON = [370, 440, 650, 730]     # Right side - radiator icon
ROI_HOT_WATER_ICON = [360, 440, 590, 650]     # Right side - tap icon

# Thresholds for icon detection
ICON_THRESHOLD = 100  # Pixels darker than this are considered "active"
ICON_ACTIVE_RATIO = 0.15  # At least 15% of ROI must be dark to be "active"
BAR_SEGMENTS = 5

# ============= DETECTION FUNCTIONS =============

def detect_icon_active(image, roi, threshold=ICON_THRESHOLD, min_ratio=ICON_ACTIVE_RATIO):
    """Detect if an icon is active by checking for dark pixels in ROI"""
    y1, y2, x1, x2 = roi

    # Validate ROI is within image bounds
    if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
        print(f"WARNING: ROI {roi} is out of image bounds {image.shape}")
        return False

    roi_image = image[y1:y2, x1:x2]

    if len(roi_image.shape) == 3:
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_image

    dark_pixels = np.sum(roi_gray < threshold)
    total_pixels = roi_gray.size
    dark_ratio = dark_pixels / total_pixels

    print(f"  Dark ratio: {dark_ratio:.3f} (threshold: {min_ratio})")
    return dark_ratio > min_ratio

def detect_flame_level(image, roi, num_segments=BAR_SEGMENTS):
    """Detect flame level from vertical bar graph"""
    y1, y2, x1, x2 = roi

    if y1 < 0 or y2 > image.shape[0] or x1 < 0 or x2 > image.shape[1]:
        print(f"WARNING: ROI {roi} is out of image bounds {image.shape}")
        return 0

    roi_image = image[y1:y2, x1:x2]

    if len(roi_image.shape) == 3:
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_image

    _, binary = cv2.threshold(roi_gray, ICON_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    height = roi_gray.shape[0]
    segment_height = height // num_segments

    active_segments = 0
    print(f"  Checking {num_segments} segments (height: {height}px, segment: {segment_height}px)")

    for i in range(num_segments):
        seg_start = height - (i + 1) * segment_height
        seg_end = height - i * segment_height
        segment = binary[seg_start:seg_end, :]

        filled_ratio = np.sum(segment > 0) / segment.size
        print(f"    Segment {i+1}: filled {filled_ratio:.3f}")

        if filled_ratio > 0.3:
            active_segments = i + 1
        else:
            break

    return active_segments

def calibrate_image(image_path):
    """Process an image and show ROI positions"""
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print('='*60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image {image_path}")
        return

    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]} (width x height)")

    # Create annotated version
    annotated = image.copy()

    # Draw ROIs with different colors
    rois = [
        (ROI_FLAME_ICON, "Flame Icon", (0, 255, 0)),      # Green
        (ROI_BAR_GRAPH, "Bar Graph", (255, 0, 0)),        # Blue
        (ROI_HEATING_ICON, "Heating", (0, 0, 255)),       # Red
        (ROI_HOT_WATER_ICON, "Hot Water", (255, 255, 0))  # Cyan
    ]

    for roi, label, color in rois:
        y1, y2, x1, x2 = roi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save annotated image
    output_path = image_path.replace('.jpg', '_annotated.jpg')
    cv2.imwrite(output_path, annotated)
    print(f"\n✓ Saved annotated image: {output_path}")

    # Test detection
    print("\nDetection Results:")
    print("-" * 40)

    print("\n1. Flame Icon:")
    flame_active = detect_icon_active(image, ROI_FLAME_ICON)
    print(f"   Result: {'ACTIVE' if flame_active else 'INACTIVE'}")

    print("\n2. Bar Graph (Flame Level):")
    flame_level = detect_flame_level(image, ROI_BAR_GRAPH)
    print(f"   Result: {flame_level}/{BAR_SEGMENTS}")

    print("\n3. Heating Icon:")
    heating_active = detect_icon_active(image, ROI_HEATING_ICON)
    print(f"   Result: {'ACTIVE' if heating_active else 'INACTIVE'}")

    print("\n4. Hot Water Icon:")
    hot_water_active = detect_icon_active(image, ROI_HOT_WATER_ICON)
    print(f"   Result: {'ACTIVE' if hot_water_active else 'INACTIVE'}")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Open the annotated image to see ROI positions")
    print("2. If ROIs are incorrect, adjust coordinates in this script:")
    print("   - ROI_FLAME_ICON = [y1, y2, x1, x2]")
    print("   - ROI_BAR_GRAPH = [y1, y2, x1, x2]")
    print("   - ROI_HEATING_ICON = [y1, y2, x1, x2]")
    print("   - ROI_HOT_WATER_ICON = [y1, y2, x1, x2]")
    print("3. Re-run calibration until detection is accurate")
    print("4. Copy final ROI values to boiler_recognition.py")

def show_pixel_finder(image_path):
    """Interactive tool to find pixel coordinates by clicking"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image {image_path}")
        return

    print("\nPixel Finder - Click on the image to get coordinates")
    print("Press 'q' to quit\n")

    coords = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: x={x}, y={y}")
            coords.append((x, y))
            # Draw a small circle at click location
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, f"({x},{y})", (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Pixel Finder', image)

            if len(coords) == 2:
                x1, y1 = coords[0]
                x2, y2 = coords[1]
                print(f"\nROI suggestion: [{min(y1,y2)}, {max(y1,y2)}, {min(x1,x2)}, {max(x1,x2)}]")
                coords.clear()

    cv2.namedWindow('Pixel Finder')
    cv2.setMouseCallback('Pixel Finder', mouse_callback)
    cv2.imshow('Pixel Finder', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Boiler Display Calibration Tool")
        print("="*60)
        print("\nUsage:")
        print("  python calibrate.py <image_file>           # Test detection")
        print("  python calibrate.py <image_file> --find    # Find coordinates")
        print("\nExample:")
        print("  python calibrate.py snapshot.jpg")
        print("  python calibrate.py snapshot.jpg --find")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    if len(sys.argv) > 2 and sys.argv[2] == '--find':
        show_pixel_finder(image_path)
    else:
        calibrate_image(image_path)
