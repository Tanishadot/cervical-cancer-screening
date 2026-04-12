#!/usr/bin/env python3
"""
Terminal Example of Swin Transformer Visualization
Shows what the feature visualization looks like in ASCII format
"""

import numpy as np

def show_example_visualization():
    """Show example of Swin Transformer visualization in terminal."""
    
    print("\n" + "="*80)
    print("🧠 SWIN TRANSFORMER FEATURE VISUALIZATION EXAMPLE")
    print("="*80)
    
    # Example feature map (simulated)
    print(f"\n📊 Sample Information:")
    print(f"  Ground Truth: Abnormal")
    print(f"  Prediction: Abnormal")
    print(f"  Confidence: 0.9234")
    
    print(f"\n🔥 Feature Activation Map:")
    print(f"  Shape: (224, 224)")
    print(f"  Min: 0.0000")
    print(f"  Max: 1.0000")
    print(f"  Mean: 0.3421")
    
    # ASCII representation of feature map
    print(f"\n📈 ASCII Feature Map (7x7 grid):")
    print("  " + "-"*30)
    
    # Simulated feature map with high activation in center
    feature_map = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2],
        [0.3, 0.6, 0.8, 0.9, 0.8, 0.6, 0.3],
        [0.4, 0.8, 0.9, 1.0, 0.9, 0.8, 0.4],
        [0.3, 0.6, 0.8, 0.9, 0.8, 0.6, 0.3],
        [0.2, 0.4, 0.6, 0.8, 0.6, 0.4, 0.2],
        [0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1]
    ])
    
    # ASCII characters for different intensity levels
    ascii_chars = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@']
    
    for i in range(7):
        row = "  |"
        for j in range(7):
            intensity = feature_map[i, j]
            char_idx = min(int(intensity * len(ascii_chars)), len(ascii_chars) - 1)
            row += ascii_chars[char_idx] * 2
        row += "|"
        print(row)
    
    print("  " + "-"*30)
    
    # Highlight regions
    print(f"\n🎯 High Activation Regions:")
    print(f"  Found 9 high-activation regions")
    print(f"  Region 1: Position (3, 3) - Intensity: 1.000")
    print(f"  Region 2: Position (2, 3) - Intensity: 0.900")
    print(f"  Region 3: Position (3, 2) - Intensity: 0.900")
    
    print(f"\n🔍 Model Interpretation:")
    print("  ✅ Correct Prediction")
    print("  📈 Model confidence is high and prediction matches ground truth")
    print("  🎯 Model focuses on central regions (likely cell nuclei)")
    
    print(f"\n🏥 Clinical Relevance:")
    print("  📍 Feature map shows regions model considers important")
    print("  🔬 Bright areas indicate cell nuclei or abnormal features")
    print("  👨‍⚕️  Clinicians can review these regions for validation")
    print("  🧪 High activation in center suggests abnormal cell morphology")
    
    print("\n" + "="*80)
    
    # Show another example
    print(f"\n📊 Sample Information (Example 2):")
    print(f"  Ground Truth: Normal")
    print(f"  Prediction: Normal")
    print(f"  Confidence: 0.8756")
    
    print(f"\n📈 ASCII Feature Map (7x7 grid):")
    print("  " + "-"*30)
    
    # Different pattern for normal cells
    normal_map = np.array([
        [0.2, 0.3, 0.4, 0.3, 0.4, 0.3, 0.2],
        [0.3, 0.5, 0.6, 0.5, 0.6, 0.5, 0.3],
        [0.4, 0.6, 0.7, 0.6, 0.7, 0.6, 0.4],
        [0.3, 0.5, 0.6, 0.5, 0.6, 0.5, 0.3],
        [0.4, 0.6, 0.7, 0.6, 0.7, 0.6, 0.4],
        [0.3, 0.5, 0.6, 0.5, 0.6, 0.5, 0.3],
        [0.2, 0.3, 0.4, 0.3, 0.4, 0.3, 0.2]
    ])
    
    for i in range(7):
        row = "  |"
        for j in range(7):
            intensity = normal_map[i, j]
            char_idx = min(int(intensity * len(ascii_chars)), len(ascii_chars) - 1)
            row += ascii_chars[char_idx] * 2
        row += "|"
        print(row)
    
    print("  " + "-"*30)
    
    print(f"\n🔍 Model Interpretation:")
    print("  ✅ Correct Prediction")
    print("  📈 Model shows distributed activation pattern")
    print("  🎯 Lower intensity suggests normal cell morphology")
    
    print("\n" + "="*80)
    print("🎉 TERMINAL VISUALIZATION EXAMPLE COMPLETED")
    print("="*80)
    
    print(f"\n💡 What this shows:")
    print(f"  • Bright regions (@, %, #) = high model attention")
    print(f"  • Dark regions (., :, -) = low model attention")
    print(f"  • Center focus = cell nuclei region (important for classification)")
    print(f"  • Different patterns for normal vs abnormal cells")
    
    print(f"\n🚀 For interactive visualizations:")
    print(f"  streamlit run dashboard_swin_xai.py")

if __name__ == "__main__":
    show_example_visualization()
