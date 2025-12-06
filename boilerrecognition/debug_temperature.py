#!/usr/bin/env python3
"""
Debug temperature detection - Simplified to working methods only
Tests only binary and adaptive thresholds at 80 and 100
"""

import cv2
import numpy as np
import subprocess
import tempfile
import os
import sys

ROI_TEMPERATURE = [140, 260, 520, 650]

def test_ssocr_methods(image, roi):
    """Test only the methods that produce valid results"""

    y1, y2, x1, x2 = roi
    temp_roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(temp_roi, cv2.COLOR_BGR2GRAY)

    print(f"Temperature ROI: {gray.shape}")
    print(f"Pixel stats: min={np.min(gray)}, max={np.max(gray)}, mean={np.mean(gray):.1f}, std={np.std(gray):.1f}")
    print()

    # Save original
    cv2.imwrite('debug_temp_original.jpg', temp_roi)
    cv2.imwrite('debug_temp_gray.jpg', gray)

    # Only test methods that work: binary and adaptive at 80 and 100
    methods = [
        # Binary methods - normal (light bg, dark digits)
        ("binary_inv_80", lambda g: cv2.threshold(g, 80, 255, cv2.THRESH_BINARY_INV)[1]),
        ("binary_inv_100", lambda g: cv2.threshold(g, 100, 255, cv2.THRESH_BINARY_INV)[1]),

        # Binary methods - inverted (dark bg, light digits)
        ("binary_80", lambda g: cv2.threshold(g, 80, 255, cv2.THRESH_BINARY)[1]),
        ("binary_100", lambda g: cv2.threshold(g, 100, 255, cv2.THRESH_BINARY)[1]),

        ("binary_inv_60", lambda g: cv2.threshold(g, 60, 255, cv2.THRESH_BINARY_INV)[1]),
        ("binary_inv_120", lambda g: cv2.threshold(g, 120, 255, cv2.THRESH_BINARY_INV)[1]),

        # Binary methods - inverted (dark bg, light digits)
        ("binary_60", lambda g: cv2.threshold(g, 60, 255, cv2.THRESH_BINARY)[1]),
        ("binary_120", lambda g: cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)[1]),

        # Adaptive methods - normal
        ("adaptive_mean_inv_80", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                                   cv2.THRESH_BINARY_INV, 11, 2)),
        ("adaptive_gaussian_inv_80", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                       cv2.THRESH_BINARY_INV, 11, 2)),

        # Adaptive methods - inverted
        ("adaptive_mean_80", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                               cv2.THRESH_BINARY, 11, 2)),
        ("adaptive_gaussian_80", lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                   cv2.THRESH_BINARY, 11, 2)),
    ]

    results = []

    print("Testing working methods:")
    print("="*80)

    for method_name, preprocess_func in methods:
        try:
            # Apply preprocessing
            processed = preprocess_func(gray)

            # Save debug image
            cv2.imwrite(f'debug_temp_{method_name}.jpg', processed)

            # Run ssocr
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                cv2.imwrite(tmp.name, processed)

                try:
                    result = subprocess.run(
                        ['ssocr', '--number-digits=2', '-d', '2', tmp.name],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if result.returncode == 0:
                        temp_str = result.stdout.strip()
                        if temp_str.isdigit() and len(temp_str) == 2:
                            temp = int(temp_str)
                            valid = "✓" if 30 <= temp <= 90 else "✗"
                            results.append((method_name, temp, valid))
                            print(f"{valid} {method_name:30s} → {temp}°C")
                        else:
                            print(f"✗ {method_name:30s} → Invalid: '{temp_str}'")
                    else:
                        stderr = result.stderr.strip()[:50]
                        print(f"✗ {method_name:30s} → ssocr failed: {stderr}")

                except subprocess.TimeoutExpired:
                    print(f"✗ {method_name:30s} → Timeout")
                except FileNotFoundError:
                    print(f"✗ ssocr not installed")
                    return
                finally:
                    try:
                        os.unlink(tmp.name)
                    except:
                        pass

        except Exception as e:
            print(f"✗ {method_name:30s} → Error: {e}")

    print("="*80)
    print()

    if results:
        # Find most common valid result
        valid_results = [r for r in results if r[2] == "✓"]
        if valid_results:
            temps = [r[1] for r in valid_results]
            most_common = max(set(temps), key=temps.count)
            count = temps.count(most_common)

            print(f"✅ CONSENSUS: {most_common}°C ({count}/{len(valid_results)} methods agree)")
            print()
            print("Best methods for this image:")
            for method, temp, valid in valid_results:
                if temp == most_common:
                    print(f"  • {method}")

            print()
            print("Recommendation:")
            best_methods = [m for m, t, v in valid_results if t == most_common]

            # Categorize by type
            binary_inv = [m for m in best_methods if m.startswith('binary_inv')]
            binary = [m for m in best_methods if m.startswith('binary_') and not m.startswith('binary_inv')]
            adaptive = [m for m in best_methods if m.startswith('adaptive')]

            if len(binary_inv) >= 2:
                print("  → Flame OFF (no backlight) - use binary_inv methods")
            elif len(binary) >= 2:
                print("  → Flame ON (orange backlight) - use binary methods")
            elif len(adaptive) >= 2:
                print("  → Use adaptive methods")
            else:
                print("  → Mixed results - try multiple methods with consensus")

        else:
            print("❌ No valid results found!")
            print()
            print("Troubleshooting:")
            print("  1. Check flame state - is backlight ON (orange) or OFF?")
            print("  2. Verify ROI coordinates are correct")
            print("  3. Check if digits are clearly visible in debug_temp_gray.jpg")
            print("  4. Try adjusting ROI or lighting")
    else:
        print("❌ No results at all")
        print("  → ssocr may not be working properly")

    print()
    print(f"Saved {len(methods) + 2} debug images: debug_temp_*.jpg")
    print("Inspect these images to see which preprocessing works best")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_temperature.py <image.jpg>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1])
    if image is None:
        print(f"Error loading {sys.argv[1]}")
        sys.exit(1)

    test_ssocr_methods(image, ROI_TEMPERATURE)
