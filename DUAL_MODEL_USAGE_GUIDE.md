# Dual-Model Medical Image Classification System

## Overview

This system combines CNN (EfficientNet) and Swin Transformer models with feature fusion and comprehensive interpretability visualizations for medical image classification.

## 🎯 Key Features

### **STEP 1: FEATURE EXTRACTION**
- **CNN Features**: Extract from last convolutional layer with Grad-CAM heatmaps
- **Swin Features**: Extract patch embeddings with attention maps
- **Feature Vectors**: CNN (1280-dim) + Swin (768-dim) = 2048-dim fused

### **STEP 2: FEATURE FUSION**
- **Concatenation**: `[cnn_features, swin_features]`
- **MLP Classifier**: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear
- **Attention Weights**: Automatic learning of CNN vs Transformer importance

### **STEP 3: DECISION LAYER**
- **Final Prediction**: Class (0=Normal, 1=Abnormal)
- **Confidence Score**: Softmax probability
- **Model Contributions**: CNN weight vs Transformer weight

### **STEP 4: INTERPRETABILITY OUTPUTS**
- **CNN Grad-CAM**: Heatmap showing local feature attention
- **Swin Attention**: Global patch importance visualization
- **Feature Comparison**: Side-by-side model contributions
- **Fusion Insights**: How models combine for decision

### **STEP 5: DASHBOARD SECTIONS**
1. **Prediction Results**: Final classification and confidence
2. **CNN Insights**: Grad-CAM heatmap and overlay
3. **Transformer Insights**: Attention map and overlay
4. **Feature Comparison**: Model weights and dimensions
5. **Fusion Insight**: Combined feature analysis

## 🚀 Quick Start

### 1. Train Models (if not already done)
```bash
# Train CNN model
python main.py --binary --model efficientnet_b0

# Train Swin model
python main.py --binary --model swin_transformer
```

### 2. Test the Pipeline
```bash
# Run comprehensive tests
python test_dual_model_pipeline.py
```

### 3. Launch Dashboard
```bash
# Start the dual-model dashboard
streamlit run dual_model_dashboard.py
```

## 📁 Required Files

```
outputs/
├── models/
│   └── best_model.pth                 # CNN (EfficientNet) model
└── cross_dataset_training/models/
    └── best_swin_model.pth           # Swin Transformer model
```

## 🎮 Dashboard Usage

### Upload/Select Image
- **Upload**: Use file uploader for custom images
- **Dataset**: Select from pre-loaded medical images

### Analysis Results
1. **Original Image**: Input medical image
2. **CNN Analysis**: 
   - Grad-CAM heatmap (local patterns)
   - Overlay on original image
   - Explanation of focus regions
3. **Transformer Analysis**:
   - Attention map (global relationships)
   - Overlay on original image
   - Explanation of patch importance
4. **Feature Fusion**:
   - Model contribution weights
   - Fused feature vector visualization
   - Decision process explanation

### Export Results
- Download analysis results as JSON
- Save visualizations for medical reports

## 🔧 Technical Details

### Feature Extraction
```python
# CNN features (1280-dim)
cnn_features = extract_cnn_features(image)  # Local patterns

# Swin features (768-dim) 
swin_features = extract_swin_features(image)  # Global context

# Fusion (2048-dim)
fused_features = torch.cat([cnn_features, swin_features], dim=1)
```

### MLP Architecture
```
Input (2048) → Linear(512) → ReLU → Dropout(0.3)
           → Linear(256) → ReLU → Dropout(0.3) 
           → Linear(2) → Softmax
```

### Attention Weights
- **CNN Weight**: How much the fusion relies on local features
- **Swin Weight**: How much the fusion relies on global features
- **Dynamic**: Learned per image during inference

## 🧪 Testing

### Run All Tests
```bash
python test_dual_model_pipeline.py
```

### Test Components Separately
```bash
# Test feature extraction
python -c "from dual_model_classifier import FeatureExtractor; print('Feature extractor OK')"

# Test feature fusion
python -c "from dual_model_classifier import FeatureFusion; print('Feature fusion OK')"

# Test complete system
python -c "from dual_model_classifier import DualModelClassifier; print('Dual model OK')"
```

## 📊 Expected Outputs

### API Response Format
```json
{
  "prediction": 0,
  "confidence": 0.853,
  "probabilities": [0.853, 0.147],
  "cnn_features_shape": [1, 1280],
  "swin_features_shape": [1, 768],
  "fused_features_shape": [1, 2048],
  "cnn_prediction": 0,
  "swin_prediction": 1,
  "cnn_weight": 0.629,
  "swin_weight": 0.371,
  "gradcam_heatmap": [[...]],  // 224x224 numpy array
  "attention_map": [[...]],     // 224x224 numpy array
  "explanation": {
    "cnn_focus": "CNN focuses on upper-right regions with high local patterns",
    "swin_focus": "Transformer shows focused attention on specific patches",
    "fusion_insight": "Fusion relies more on CNN features (0.63 vs 0.37)"
  }
}
```

## 🏥 Medical Interpretation

### For Doctors/Clinicians
1. **CNN Grad-CAM**: Shows *what* local patterns the model sees
   - Red areas = high local feature activation
   - Useful for identifying specific cellular structures

2. **Swin Attention**: Shows *where* global relationships matter
   - Bright patches = important spatial regions
   - Useful for understanding context and relationships

3. **Fusion Weights**: Shows *how* models combine
   - CNN > 0.6: Local patterns dominate decision
   - Swin > 0.6: Global context dominates decision
   - Balanced (~0.5): Both contribute equally

### Clinical Decision Support
- **High Confidence (>0.9)**: Strong model agreement
- **Medium Confidence (0.7-0.9)**: Consider both visualizations
- **Low Confidence (<0.7)**: Review both heatmaps carefully

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   ❌ CNN model not found at outputs/models/best_model.pth
   ```
   **Solution**: Train the CNN model first or check file paths

2. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use CPU or reduce batch size

3. **Grad-CAM Layer Not Found**
   ```
   ValueError: Target layer 'backbone.conv_head' not found
   ```
   **Solution**: Check model architecture with `find_cnn_layers.py`

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test with dummy data
classifier = DualModelClassifier(...)
dummy_input = torch.randn(1, 3, 224, 224)
result = classifier.predict(dummy_input)
print(result)
```

## 📈 Performance

### Expected Metrics
- **Inference Time**: ~200ms per image (CPU)
- **Memory Usage**: ~2GB RAM
- **Accuracy**: Depends on trained models
- **Interpretability**: Full visual explanations

### Optimization Tips
1. **GPU Acceleration**: Use CUDA for faster inference
2. **Batch Processing**: Process multiple images together
3. **Model Pruning**: Reduce model size for deployment
4. **Caching**: Cache preprocessed features

## 🔬 Research Applications

### Medical Research
- **Feature Analysis**: Study what features each model learns
- **Error Analysis**: Understand failure modes
- **Domain Adaptation**: Test on different medical datasets
- **Clinical Validation**: Compare with expert annotations

### Extensions
- **More Models**: Add ResNet, Vision Transformer, etc.
- **3D Medical Images**: Extend to CT/MRI volumes
- **Multi-Class**: Support more than binary classification
- **Temporal Analysis**: Handle video/sequence data

## 📞 Support

### Files Created
- `dual_model_classifier.py`: Core classification system
- `dual_model_dashboard.py`: Streamlit dashboard
- `test_dual_model_pipeline.py`: Comprehensive tests
- `find_cnn_layers.py`: Model debugging utility

### Dependencies
- PyTorch >= 1.9
- Streamlit >= 1.25
- Plotly >= 5.0
- OpenCV >= 4.5
- TIMM >= 0.6

---

## 🎉 Success!

Your dual-model medical image classification system is now ready! 

**Next Steps:**
1. ✅ Train your models (if not done)
2. ✅ Run tests to verify everything works
3. ✅ Launch the dashboard
4. ✅ Start analyzing medical images with dual-model insights

The system provides comprehensive interpretability to help medical professionals understand AI decisions, combining the strengths of both CNN (local features) and Transformer (global context) models.
