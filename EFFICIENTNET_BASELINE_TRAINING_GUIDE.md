# EfficientNet-B0 Baseline Training Guide

## 🎯 Objective

Train an EfficientNet-B0 baseline model on SIPaKMeD dataset (CROPPED-only) and evaluate cross-dataset generalization on Herlev dataset.

## 📊 Dataset Status

### ✅ Validated and Ready
- **Training Data**: SIPaKMeD (CROPPED-only) - 4,049 samples
- **Validation Data**: SIPaKMeD split (15%) - ~607 samples  
- **Test Data**: Herlev (cross-dataset) - ~917 samples
- **Duplicate Rate**: 0.00% (perfect)
- **Data Leakage**: 0 samples (perfect separation)
- **Class Balance**: 1.06 imbalance ratio (excellent)
- **Classification**: Binary (NORMAL vs ABNORMAL)

### 🏷️ Label Mapping (Binary)
- **NORMAL**: Superficial-Intermediate, Parabasal
- **ABNORMAL**: Koilocytotic, Metaplastic, Dyskeratotic

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

### 2. Run Training (Binary Classification)
```bash
python train_efficientnet_baseline.py --mode binary
```

### 3. Run Training (Multiclass Classification)
```bash
python train_efficientnet_baseline.py --mode multiclass
```

### 4. Custom Parameters
```bash
python train_efficientnet_baseline.py --mode binary --epochs 30 --batch-size 16 --lr 0.0001
```

---

## ⚙️ Configuration

### Model Configuration
- **Architecture**: EfficientNet-B0 (pretrained)
- **Dropout**: 0.3
- **Output Classes**: 2 (binary) or 5 (multiclass)
- **Backbone**: Frozen initially, trainable classifier

### Training Configuration
- **Epochs**: 25 (with early stopping)
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Weight Decay**: 1e-5
- **Optimizer**: Adam
- **Early Stopping**: Patience = 5

### Data Augmentation (Training)
- **Resize**: 256x256 → RandomCrop 224x224
- **Horizontal Flip**: p=0.5
- **Vertical Flip**: p=0.5
- **Random Rotation**: ±15°
- **Color Jitter**: brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
- **Normalization**: ImageNet mean/std

### Validation/Test Transforms
- **Resize**: 224x224
- **Normalization**: ImageNet mean/std

---

## 📈 Expected Results

### Training Metrics
- **Validation Accuracy**: Expected > 85%
- **Training Stability**: Smooth convergence with early stopping
- **Class Balance**: Good balance due to CROPPED-only dataset

### Cross-Dataset Generalization
- **Herlev Test Accuracy**: Expected 60-80% (domain gap expected)
- **F1-Score**: Primary metric for generalization assessment
- **Generalization Gap**: < 20% indicates good transfer learning

### Performance Indicators
- **Excellent**: Gap < 10%, F1 > 0.8
- **Good**: Gap < 20%, F1 > 0.7
- **Poor**: Gap > 20%, F1 < 0.7

---

## 📁 Output Structure

```
outputs/efficientnet_baseline/
├── models/
│   └── best_model.pth              # Best checkpoint
├── plots/
│   ├── training_curves.png         # Loss/Accuracy curves
│   ├── confusion_matrix_validation.png
│   └── confusion_matrix_herlev.png
├── logs/
│   └── efficientnet_baseline.log   # Training logs
├── results.json                    # Complete results
└── training_setup_summary.json     # Setup validation
```

---

## 🔍 Monitoring and Debugging

### Training Progress
- **Loss Curves**: Should decrease steadily
- **Accuracy Curves**: Should increase and plateau
- **Learning Rate**: Adaptive reduction on plateau
- **Early Stopping**: Prevents overfitting

### Common Issues
1. **Overfitting**: High train accuracy, low val accuracy
   - Solution: Increase dropout, reduce epochs, add augmentation
   
2. **Underfitting**: Low accuracy on both train and val
   - Solution: Increase learning rate, reduce regularization
   
3. **Poor Generalization**: Large gap between val and test
   - Solution: More diverse augmentation, domain adaptation

### Debug Mode
```bash
python train_efficientnet_baseline.py --debug-dataset --mode binary
```

---

## 📊 Evaluation Metrics

### Primary Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across classes
- **Recall**: Weighted recall across classes
- **F1-Score**: Harmonic mean of precision and recall

### Secondary Metrics
- **AUC**: Area under ROC curve (binary only)
- **Confusion Matrix**: Per-class performance
- **Classification Report**: Detailed per-class metrics

### Cross-Dataset Analysis
- **Generalization Gap**: Val Acc - Test Acc
- **Domain Adaptation**: Performance drop analysis
- **Class-wise Performance**: Identify weak classes

---

## 🎯 Expected Training Timeline

### Phase 1: Setup (5 minutes)
- Dataset validation
- Configuration loading
- Model initialization

### Phase 2: Training (30-60 minutes)
- 25 epochs with early stopping
- ~2-3 minutes per epoch
- Automatic checkpointing

### Phase 3: Evaluation (5 minutes)
- Validation evaluation
- Herlev cross-dataset testing
- Results generation and plotting

### Total Time: ~45-70 minutes

---

## 🔧 Advanced Configuration

### Custom Config File
```yaml
# custom_config.yaml
dataset:
  classification_mode: "multiclass"
  use_cropped_only: true

training:
  num_epochs: 50
  batch_size: 16
  learning_rate: 0.00005

model:
  dropout_rate: 0.5
```

```bash
python train_efficientnet_baseline.py --config custom_config.yaml
```

### Hyperparameter Tuning
```bash
# High learning rate for faster convergence
python train_efficientnet_baseline.py --lr 0.0005

# Smaller batch size for memory efficiency
python train_efficientnet_baseline.py --batch-size 16

# More epochs for complex datasets
python train_efficientnet_baseline.py --epochs 50
```

---

## 🚨 Important Notes

### Data Safety
- ✅ Dataset guard validation runs automatically
- ✅ No data leakage between SIPaKMeD and Herlev
- ✅ CROPPED-only ensures high-quality images
- ✅ Proper train/val/test separation

### Model Safety
- ✅ Pretrained weights for better convergence
- ✅ Gradient clipping prevents explosion
- ✅ Early stopping prevents overfitting
- ✅ Class weighting handles imbalance

### Reproducibility
- ✅ Fixed random seeds (configurable)
- ✅ Deterministic data loading
- ✅ Detailed logging and checkpoints
- ✅ Complete configuration tracking

---

## 🎉 Success Criteria

### Training Success
- ✅ Validation accuracy > 85%
- ✅ Stable training curves
- ✅ No overfitting signs
- ✅ Early stopping triggered appropriately

### Generalization Success
- ✅ Herlev test accuracy > 60%
- ✅ F1-score > 0.7
- ✅ Generalization gap < 20%
- ✅ Reasonable confusion matrix

### Baseline Achievement
- ✅ Establishes strong performance baseline
- ✅ Provides comparison for hybrid models
- ✅ Validates cross-dataset approach
- ✅ Demonstrates preprocessing effectiveness

---

## 🔄 Next Steps

### After Baseline Training
1. **Analyze Results**: Review training curves and metrics
2. **Compare with Hybrid**: Use baseline for comparison
3. **Error Analysis**: Examine misclassified samples
4. **Domain Adaptation**: Consider if gap is too large

### Model Improvements
1. **Architecture**: Try ResNet50, Swin Transformer
2. **Augmentation**: Add more diverse transforms
3. **Regularization**: Adjust dropout and weight decay
4. **Ensemble**: Combine multiple models

### Research Directions
1. **Domain Adaptation**: Address SIPaKMeD→Herlev gap
2. **Multi-dataset**: Train on combined datasets
3. **Semi-supervised**: Use unlabeled data
4. **XAI Integration**: Add explainability features

---

## 📞 Troubleshooting

### Common Errors
1. **CUDA Out of Memory**: Reduce batch size
2. **Dataset Not Found**: Check dataset paths
3. **Permission Errors**: Check file permissions
4. **Import Errors**: Install missing dependencies

### Getting Help
- Check logs: `outputs/efficientnet_baseline/efficientnet_baseline.log`
- Review configuration: `outputs/efficientnet_baseline/training_setup_summary.json`
- Validate dataset: Run `python setup_efficientnet_baseline.py`

---

*This guide provides everything needed to train a strong EfficientNet-B0 baseline for cervical cancer classification with proper cross-dataset evaluation.*
