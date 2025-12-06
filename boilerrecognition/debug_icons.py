#!/usr/bin/env python3
"""
Debug icon detection with different thresholds
"""

import cv2
import numpy as np
import sys

ROI_FLAME_ICON = [350, 530, 500, 540]
ROI_HEATING_ICON = [370, 440, 650, 730]
ROI_HOT_WATER_ICON = [360, 440, 590, 650]

def test_icon_thresholds(image, roi, name):
    """Test icon detection with various thresholds"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)

    y1, y2, x1, x2 = roi
    roi_img = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    print(f"ROI size: {gray.shape}")
    print(f"Pixel stats: min={np.min(gray)}, max={np.max(gray)}, mean={np.mean(gray):.1f}")

    # Test different darkness thresholds
    print(f"\nDark pixel detection (threshold < X):")
    for threshold in [50, 80, 100, 120, 150]:
        dark_pixels = np.sum(gray < threshold)
        dark_ratio = dark_pixels / gray.size
        print(f"  < {threshold:3d}: {dark_ratio:6.2%} dark pixels")

    # Test different minimum ratios
    print(f"\nWould detect as ACTIVE with threshold=100:")
    for min_ratio in [0.01, 0.05, 0.10, 0.15, 0.20]:
        dark_pixels = np.sum(gray < 100)
        dark_ratio = dark_pixels / gray.size
        active = dark_ratio > min_ratio
        status = "✓ YES" if active else "✗ NO"
        print(f"  min_ratio={min_ratio:.2f}: {status} (actual: {dark_ratio:.2%})")

    # Edge detection for hollow icons
    print(f"\nEdge detection:")
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    edge_ratio = edge_pixels / edges.size
    print(f"  Edge pixels: {edge_ratio:.2%}")

    # Save debug images
    cv2.imwrite(f'debug_{name}_roi.jpg', roi_img)
    cv2.imwrite(f'debug_{name}_gray.jpg', gray)
    cv2.imwrite(f'debug_{name}_edges.jpg', edges)

    # Test binary threshold
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(f'debug_{name}_binary.jpg', binary)
    print(f"\nSaved debug images: debug_{name}_*.jpg")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_icons.py <image.jpg>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Error loading {sys.argv[1]}")
        sys.exit(1)

    test_icon_thresholds(image, ROI_FLAME_ICON, "flame")
    test_icon_thresholds(image, ROI_HEATING_ICON, "heating")
    test_icon_thresholds(image, ROI_HOT_WATER_ICON, "hot_water")

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print('='*60)
    print("Based on the results above, adjust thresholds in boiler-mqtt.py")
    print("Look for the 'actual' percentages and set min_ratio slightly below those values")
