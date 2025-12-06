#!/usr/bin/env python3
"""
Boiler Display Recognition - Complete Test Version
With temperature reading via ssocr
"""

import cv2
import numpy as np
import sys
import os
import glob
import subprocess
import tempfile

# ============= CALIBRATED CONFIGURATION =============
ROI_FLAME_ICON = [350, 530, 500, 540]     # Left side - flame/droplet icon
ROI_BAR_GRAPH = [350, 530, 540, 580]         # Left side - vertical bar graph
ROI_HEATING_ICON = [370, 440, 650, 730]     # Right side - radiator icon
ROI_HOT_WATER_ICON = [360, 440, 590, 650]     # Right side - tap icon
ROI_TEMPERATURE = [140, 260, 520, 650]     # Adjust this for your temperature display

ICON_THRESHOLD = 100
ICON_ACTIVE_RATIO = 0.15
BAR_SEGMENTS = 6  # Fixed: 6 bars not 5

# ============= TEMPERATURE READING =============

def read_temperature_ssocr(image, roi, debug=False):
    """Read temperature using ssocr command-line tool"""
    y1, y2, x1, x2 = roi
    temp_region = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(temp_region, cv2.COLOR_BGR2GRAY)

    # Threshold and invert (ssocr expects light digits on dark background)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        cv2.imwrite(tmp.name, thresh)

        try:
            # Call ssocr
            result = subprocess.run(
                ['ssocr', '--number-digits=2', '-d', '2', tmp.name],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                temp_str = result.stdout.strip()
                if temp_str.isdigit():
                    temp = int(temp_str)
                    if 20 <= temp <= 99:  # Sanity check
                        if debug:
                            print(f"    Raw output: '{temp_str}'")
                            print(f"    Temperature: {temp}°C")
                        return temp

            if debug:
                print(f"    ssocr failed: {result.stderr}")

        except FileNotFoundError:
            if debug:
                print("    ssocr not installed. Run: sudo apt-get install ssocr")
        except Exception as e:
            if debug:
                print(f"    Error: {e}")
        finally:
            try:
                os.unlink(tmp.name)
            except:
                pass

    return None

# ============= ICON DETECTION FUNCTIONS =============

def detect_icon_active(image, roi, threshold=ICON_THRESHOLD, min_ratio=ICON_ACTIVE_RATIO, debug=False):
    """Detect if an icon is active by checking for dark pixels"""
    y1, y2, x1, x2 = roi
    roi_image = image[y1:y2, x1:x2]

    if len(roi_image.shape) == 3:
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_image

    dark_pixels = np.sum(roi_gray < threshold)
    total_pixels = roi_gray.size
    dark_ratio = dark_pixels / total_pixels

    is_active = dark_ratio > min_ratio

    if debug:
        print(f"    Dark pixels: {dark_pixels}/{total_pixels} = {dark_ratio:.1%}")
        print(f"    Threshold: {min_ratio:.1%}")
        print(f"    Status: {'✓ ACTIVE' if is_active else '✗ INACTIVE'}")

    return is_active, dark_ratio, roi_image

def detect_flame_level(image, roi, num_segments=BAR_SEGMENTS, debug=False):
    """Detect flame level from vertical bar graph"""
    y1, y2, x1, x2 = roi
    roi_image = image[y1:y2, x1:x2]

    if len(roi_image.shape) == 3:
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi_image

    _, binary = cv2.threshold(roi_gray, ICON_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    height = roi_gray.shape[0]
    segment_height = height // num_segments

    active_segments = 0
    segment_ratios = []

    for i in range(num_segments):
        seg_start = height - (i + 1) * segment_height
        seg_end = height - i * segment_height
        segment = binary[seg_start:seg_end, :]

        filled_ratio = np.sum(segment > 0) / segment.size
        segment_ratios.append(filled_ratio)

        if filled_ratio > 0.3:
            active_segments = i + 1
        else:
            break

    if debug:
        print(f"    Bar graph height: {height}px, segment: {segment_height}px")
        for i, ratio in enumerate(segment_ratios):
            status = "█" if ratio > 0.3 else "░"
            print(f"    Segment {i+1}: {status} {ratio:.1%}")
        print(f"    Level: {active_segments}/{num_segments}")

    return active_segments, segment_ratios, roi_image

def process_boiler_display(image, debug=False):
    """Process the entire boiler display"""
    results = {
        'flame_active': False,
        'flame_level': 0,
        'heating_active': False,
        'hot_water_active': False,
        'temperature': None,
        'debug_data': {}
    }

    if debug:
        print("\n" + "="*60)
        print("DETECTION RESULTS")
        print("="*60)

    # 1. Flame icon
    if debug:
        print("\n1. FLAME ICON")
        print("-" * 40)
    flame_active, flame_ratio, flame_roi = detect_icon_active(image, ROI_FLAME_ICON, debug=debug)
    results['flame_active'] = flame_active
    results['debug_data']['flame_ratio'] = flame_ratio
    results['debug_data']['flame_roi'] = flame_roi

    # 2. Bar graph
    if debug:
        print("\n2. BAR GRAPH (Flame Level)")
        print("-" * 40)
    flame_level, segment_ratios, bar_roi = detect_flame_level(image, ROI_BAR_GRAPH, debug=debug)
    results['flame_level'] = flame_level
    results['debug_data']['segment_ratios'] = segment_ratios
    results['debug_data']['bar_roi'] = bar_roi

    # 3. Heating icon
    if debug:
        print("\n3. HEATING ICON")
        print("-" * 40)
    heating_active, heating_ratio, heating_roi = detect_icon_active(image, ROI_HEATING_ICON, debug=debug)
    results['heating_active'] = heating_active
    results['debug_data']['heating_ratio'] = heating_ratio
    results['debug_data']['heating_roi'] = heating_roi

    # 4. Hot water icon
    if debug:
        print("\n4. HOT WATER ICON")
        print("-" * 40)
    hot_water_active, hw_ratio, hw_roi = detect_icon_active(image, ROI_HOT_WATER_ICON, debug=debug)
    results['hot_water_active'] = hot_water_active
    results['debug_data']['hw_ratio'] = hw_ratio
    results['debug_data']['hw_roi'] = hw_roi

    # 5. Temperature
    if debug:
        print("\n5. TEMPERATURE (Seven-Segment OCR)")
        print("-" * 40)
    temperature = read_temperature_ssocr(image, ROI_TEMPERATURE, debug=debug)
    results['temperature'] = temperature

    return results

def create_annotated_image(image, results):
    """Create annotated version showing all detections"""
    annotated = image.copy()

    COLOR_ACTIVE = (0, 255, 0)
    COLOR_INACTIVE = (0, 0, 255)

    rois = [
        (ROI_FLAME_ICON, f"Flame: {'ON' if results['flame_active'] else 'OFF'}",
         COLOR_ACTIVE if results['flame_active'] else COLOR_INACTIVE),
        (ROI_BAR_GRAPH, f"Level: {results['flame_level']}/6", (255, 0, 0)),
        (ROI_HEATING_ICON, f"Heating: {'ON' if results['heating_active'] else 'OFF'}",
         COLOR_ACTIVE if results['heating_active'] else COLOR_INACTIVE),
        (ROI_HOT_WATER_ICON, f"Hot Water: {'ON' if results['hot_water_active'] else 'OFF'}",
         COLOR_ACTIVE if results['hot_water_active'] else COLOR_INACTIVE),
        (ROI_TEMPERATURE, f"Temp: {results['temperature'] if results['temperature'] else '?'}°C",
         COLOR_ACTIVE if results['temperature'] else COLOR_INACTIVE)
    ]

    for roi, label, color in rois:
        y1, y2, x1, x2 = roi
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated, (x1, y1-label_size[1]-10),
                     (x1+label_size[0]+5, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Summary overlay
    summary_y = 30
    summary_texts = [
        f"Temperature: {results['temperature'] if results['temperature'] else '?'}°C",
        f"Flame: {'ON' if results['flame_active'] else 'OFF'}",
        f"Level: {results['flame_level']}/6",
        f"Heating: {'ON' if results['heating_active'] else 'OFF'}",
        f"Hot Water: {'ON' if results['hot_water_active'] else 'OFF'}"
    ]

    for i, text in enumerate(summary_texts):
        cv2.putText(annotated, text, (10, summary_y + i*30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return annotated

def process_single_image(image_path, save_annotated=True, debug=True):
    """Process a single image"""
    print(f"\n{'='*70}")
    print(f"Processing: {os.path.basename(image_path)}")
    print('='*70)

    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image {image_path}")
        return None

    results = process_boiler_display(image, debug=debug)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    temp_display = f"{results['temperature']}°C" if results['temperature'] else "NOT DETECTED"
    print(f"Temperature:       {temp_display}")
    print(f"Flame Active:      {'✓ YES' if results['flame_active'] else '✗ NO'}")
    print(f"Flame Level:       {results['flame_level']}/6 {'█' * results['flame_level']}{'░' * (6-results['flame_level'])}")
    print(f"Heating Active:    {'✓ YES' if results['heating_active'] else '✗ NO'}")
    print(f"Hot Water Active:  {'✓ YES' if results['hot_water_active'] else '✗ NO'}")

    if save_annotated:
        annotated = create_annotated_image(image, results)
        output_path = image_path.replace('.jpg', '_detected.jpg')
        cv2.imwrite(output_path, annotated)
        print(f"\n✓ Saved annotated image: {os.path.basename(output_path)}")

    return results

def process_multiple_images(pattern, save_annotated=True):
    """Process multiple images"""
    files = glob.glob(pattern)

    if not files:
        print(f"No files found matching: {pattern}")
        return

    print(f"\nFound {len(files)} image(s) to process")

    all_results = []
    for image_path in sorted(files):
        results = process_single_image(image_path, save_annotated=save_annotated, debug=False)
        if results:
            results['filename'] = os.path.basename(image_path)
            all_results.append(results)

    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        print(f"{'Filename':<40} {'Temp':<8} {'Flame':<8} {'Level':<8} {'Heat':<8} {'Water':<8}")
        print("-" * 80)
        for r in all_results:
            temp_str = f"{r['temperature']}°C" if r['temperature'] else "?"
            print(f"{r['filename']:<40} "
                  f"{temp_str:<8} "
                  f"{'ON' if r['flame_active'] else 'OFF':<8} "
                  f"{r['flame_level']}/6{'':<5} "
                  f"{'ON' if r['heating_active'] else 'OFF':<8} "
                  f"{'ON' if r['hot_water_active'] else 'OFF':<8}")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Boiler Display Recognition - Complete Test Tool")
        print("="*60)
        print("\nUsage:")
        print("  python test_complete.py <image_file>")
        print("  python test_complete.py <pattern>")
        print("  python test_complete.py <image> --no-save")
        print("\nExamples:")
        print("  python test_complete.py snapshot.jpg")
        print("  python test_complete.py snapshot*.jpg")
        print("  python test_complete.py *.jpg")
        print("\nNote: Requires 'ssocr' for temperature reading")
        print("      Install with: sudo apt-get install ssocr")
        sys.exit(1)

    pattern = sys.argv[1]
    save_annotated = '--no-save' not in sys.argv

    files = glob.glob(pattern)

    if len(files) == 0:
        print(f"ERROR: No files found matching: {pattern}")
        sys.exit(1)
    elif len(files) == 1:
        process_single_image(files[0], save_annotated=save_annotated, debug=True)
    else:
        process_multiple_images(pattern, save_annotated=save_annotated)

if __name__ == "__main__":
    main()
