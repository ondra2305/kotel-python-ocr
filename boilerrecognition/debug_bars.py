#!/usr/bin/env python3
"""
Debug script to analyze bar graph detection
"""

import cv2
import numpy as np
import sys

# Your calibrated ROI
ROI_FLAME_ICON = [350, 530, 500, 540]     # Left side - flame/droplet icon
ROI_BAR_GRAPH = [350, 530, 540, 580]         # Left side - vertical bar graph
ROI_HEATING_ICON = [370, 440, 650, 730]     # Right side - radiator icon
ROI_HOT_WATER_ICON = [360, 440, 590, 650]     # Right side - tap icon

def analyze_bar_graph(image_path, num_segments=6):
    """Detailed analysis of bar graph"""
    print(f"\nAnalyzing: {image_path}")
    print("="*60)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load {image_path}")
        return

    y1, y2, x1, x2 = ROI_BAR_GRAPH
    roi = image[y1:y2, x1:x2]

    print(f"ROI size: {roi.shape[0]}px height x {roi.shape[1]}px width")

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Show pixel value distribution
    print(f"\nPixel intensity stats:")
    print(f"  Min: {np.min(gray)}")
    print(f"  Max: {np.max(gray)}")
    print(f"  Mean: {np.mean(gray):.1f}")
    print(f"  Median: {np.median(gray):.1f}")

    # Try different thresholds
    print(f"\nTrying different thresholds:")
    for threshold in [80, 100, 120, 150]:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        dark_ratio = np.sum(binary > 0) / binary.size
        print(f"  Threshold {threshold}: {dark_ratio:.1%} dark pixels")

    # Use threshold 100
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Analyze segments
    height = gray.shape[0]
    segment_height = height // num_segments

    print(f"\nSegment analysis ({num_segments} segments):")
    print(f"  Total height: {height}px")
    print(f"  Segment height: {segment_height}px")
    print(f"  {'Seg':<5} {'Y-range':<15} {'Filled':<10} {'Status'}")
    print("-" * 60)

    active_count = 0
    for i in range(num_segments):
        # From bottom to top
        seg_start = height - (i + 1) * segment_height
        seg_end = height - i * segment_height
        segment = binary[seg_start:seg_end, :]

        filled_ratio = np.sum(segment > 0) / segment.size
        status = "ACTIVE" if filled_ratio > 0.3 else "inactive"
        bar = "█" if filled_ratio > 0.3 else "░"

        print(f"  {i+1:<5} {seg_start:3d}-{seg_end:3d}      {filled_ratio:>6.1%}    {bar} {status}")

        if filled_ratio > 0.3:
            active_count = i + 1
        # Don't break - show all segments for analysis

    print(f"\nDetected level: {active_count}/{num_segments}")

    # Try with lower threshold
    print(f"\n--- Testing with 20% threshold instead of 30% ---")
    active_count_low = 0
    for i in range(num_segments):
        seg_start = height - (i + 1) * segment_height
        seg_end = height - i * segment_height
        segment = binary[seg_start:seg_end, :]

        filled_ratio = np.sum(segment > 0) / segment.size
        if filled_ratio > 0.2:  # Lower threshold
            active_count_low = i + 1

    print(f"With 20% threshold: {active_count_low}/{num_segments}")

    # Try with 15% threshold
    active_count_lower = 0
    for i in range(num_segments):
        seg_start = height - (i + 1) * segment_height
        seg_end = height - i * segment_height
        segment = binary[seg_start:seg_end, :]

        filled_ratio = np.sum(segment > 0) / segment.size
        if filled_ratio > 0.15:  # Even lower
            active_count_lower = i + 1

    print(f"With 15% threshold: {active_count_lower}/{num_segments}")

    # Save debug images
    output_base = image_path.replace('.jpg', '')

    # Save ROI
    cv2.imwrite(f'{output_base}_bar_roi.jpg', roi)

    # Save grayscale
    cv2.imwrite(f'{output_base}_bar_gray.jpg', gray)

    # Save binary
    cv2.imwrite(f'{output_base}_bar_binary.jpg', binary)

    # Save with segment lines
    debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for i in range(num_segments + 1):
        y = height - i * segment_height
        cv2.line(debug_img, (0, y), (debug_img.shape[1], y), (0, 255, 0), 1)
        if i < num_segments:
            seg_num = i + 1
            cv2.putText(debug_img, f"{seg_num}", (5, y + segment_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite(f'{output_base}_bar_segments.jpg', debug_img)

    print(f"\n✓ Saved debug images:")
    print(f"  {output_base}_bar_roi.jpg - Raw ROI")
    print(f"  {output_base}_bar_gray.jpg - Grayscale")
    print(f"  {output_base}_bar_binary.jpg - Thresholded")
    print(f"  {output_base}_bar_segments.jpg - Segment divisions")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_bars.py <image.jpg>")
        print("\nThis will analyze bar graph detection in detail")
        sys.exit(1)

    # Test with 5 and 6 segments
    for num_segs in [5, 6]:
        print(f"\n{'='*60}")
        print(f"Testing with {num_segs} segments")
        print('='*60)
        analyze_bar_graph(sys.argv[1], num_segs)
