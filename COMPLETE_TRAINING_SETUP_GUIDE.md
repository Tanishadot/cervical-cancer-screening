# Complete End-to-End EfficientNet-B0 Training Setup Guide

## 🎯 Objective

Set up complete environment, validate dataset, and train EfficientNet-B0 to achieve 85-95% validation accuracy with strong cross-dataset generalization.

---

## 🚀 QUICK START (For Systems with Working PyTorch)

### Option 1: Automated Pipeline
```bash
python setup_and_train.py
```

### Option 2: Manual Steps
```bash
# 1. Install dependencies
pip install torch torchvision torchaudio numpy pandas matplotlib seaborn scikit-learn pillow opencv-python tqdm pyyaml

# 2. Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 3. Run dataset validation
python -c "from utils.dataset_guard import run_dataset_guard; print(run_dataset_guard({'dataset': {'root_dir': 'datasets', 'train_dataset': 'sipakmed', 'classification_mode': 'binary', 'use_cropped_only': True}, 'paths': {'logs': 'outputs/logs'}})"

# 4. Run training
python train_efficientnet_optimized.py --mode binary
```

---

## ✅ STEP 0: INSTALL DEPENDENCIES

### Required Packages
```bash
# Core ML framework
pip install torch torchvision torchaudio

# Data processing
pip install numpy pandas

# Visualization
pip install matplotlib seaborn

# Machine learning
pip install scikit-learn

# Image processing
pip install pillow opencv-python

# Utilities
pip install tqdm pyyaml
```

### GPU Check
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

**Expected Output:**
- `CUDA Available: True` (GPU available)
- `CUDA Available: False` (CPU only)

---

## ✅ STEP 1: VERIFY ENVIRONMENT

### Python Version Check
```bash
python --version
```
**Required:** Python ≥ 3.9

### PyTorch Installation Check
```bash
python -c "
import torch
import torchvision
print('PyTorch Version:', torch.__version__)
print('TorchVision Version:', torchvision.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Count:', torch.cuda.device_count())
    print('GPU Name:', torch.cuda.get_device_name(0))
"
```

### Expected Output
```
PyTorch Version: 2.x.x
TorchVision Version: 0.x.x
CUDA Available: True
GPU Count: 1
GPU Name: [Your GPU Name]
```

---

## ✅ STEP 2: RUN DATASET GUARD

### Execute Dataset Validation
```bash
python -c "
from utils.dataset_guard import run_dataset_guard
config = {
    'dataset': {
        'root_dir': 'datasets',
        'train_dataset': 'sipakmed',
        'classification_mode': 'binary',
        'use_cropped_only': True,
        'run_validation': True,
        'validation_strict_mode': True
    },
    'paths': {'logs': 'outputs/logs'}
}
results = run_dataset_guard(config)
print('Guard Status:', results['guard_status'])
print('Action:', results['action'])
print('Message:', results['message'])
"
```

### Expected Validation Results
```
🛡️ DATASET GUARD: PASS - PROCEED
✅ Dataset validation passed

Dataset Statistics:
- Total Samples: 4049
- Duplicate Percentage: 0.00%
- Imbalance Ratio: 1.06
- CROPPED Percentage: 100.0%
```

**IF VALIDATION FAILS:** Training will be automatically stopped.

---

## ✅ STEP 3: DATASET CONFIGURATION

### Training Setup
```yaml
dataset:
  root_dir: "datasets"
  train_dataset: "sipakmed"
  classification_mode: "binary"
  use_cropped_only: true

splits:
  train: "SIPaKMeD (70%)"
  validation: "SIPaKMeD (15%)"
  test: "Herlev (ONLY)"
```

### Data Flow
1. **Training**: SIPaKMeD CROPPED images (70%)
2. **Validation**: SIPaKMeD CROPPED images (15%)
3. **Testing**: Herlev images (100% - cross-dataset)

---

## ✅ STEP 4: PREPROCESSING PIPELINE

### Training Transforms
```python
transforms.Compose([
    transforms.Resize((256, 256)),           # Larger initial size
    transforms.CenterCrop(224),               # Consistent focus
    transforms.RandomHorizontalFlip(p=0.5),   # Medical symmetry
    transforms.RandomVerticalFlip(p=0.5),     # Cell orientation
    transforms.RandomRotation(15),              # Rotation invariance
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Scale invariance
    transforms.ColorJitter(                     # Color variation
        brightness=0.2, contrast=0.2, 
        saturation=0.2, hue=0.1
    ),
    GaussianBlur(p=0.2, sigma_range=(0.1, 1.5)),  # Medical noise
    transforms.ToTensor(),
    transforms.Normalize(                         # ImageNet standard
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Validation/Test Transforms
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

## ✅ STEP 5: MODEL SETUP

### EfficientNet-B0 Configuration
```python
import torchvision.models as models

model = models.efficientnet_b0(pretrained=True)

# Enhanced classifier
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),                    # Higher dropout
    nn.Linear(num_features, 512),         # Intermediate layer
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),                    # Additional dropout
    nn.Linear(512, 2)                   # Binary output
)
```

### Model Specifications
- **Architecture**: EfficientNet-B0
- **Pretrained**: ImageNet weights
- **Dropout**: 0.4 (regularization)
- **Output Classes**: 2 (binary classification)
- **Parameters**: ~5M total, ~1M trainable initially

---

## ✅ STEP 6: TRAINING STRATEGY

### Phase 1: Frozen Backbone (Epochs 1-5)
```python
# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

# Train only classifier
optimizer = Adam(model.parameters(), lr=1e-4)
```

### Phase 2: Progressive Unfreezing (Epochs 6-30)
```python
# Unfreeze entire model
for param in model.parameters():
    param.requires_grad = True

# Reduce learning rate
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.5  # lr = 5e-5
```

---

## ✅ STEP 7: TRAINING CONFIGURATION

### Complete Training Setup
```python
# Loss function with label smoothing
criterion = CrossEntropyLoss(label_smoothing=0.1)

# Optimizer
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

# Training parameters
epochs = 30
batch_size = 16
early_stopping_patience = 5
```

---

## ✅ STEP 8: REGULARIZATION

### Regularization Techniques
1. **Dropout**: 0.4 (prevents overfitting)
2. **Label Smoothing**: 0.1 (prevents overconfidence)
3. **Weight Decay**: 1e-5 (L2 regularization)
4. **Gradient Clipping**: 1.0 (prevents explosion)
5. **Data Augmentation**: Diverse transforms
6. **Early Stopping**: Prevents overtraining

---

## ✅ STEP 9: LOGGING & SAVING

### Metrics Tracked
```python
metrics = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'val_f1': [],        # Primary medical metric
    'val_recall': [],     # Important for diagnosis
    'learning_rate': []
}
```

### Saved Outputs
```
outputs/efficientnet_end2end/
├── models/
│   └── best_model.pth              # Best checkpoint
├── plots/
│   ├── training_curves.png         # Loss/accuracy curves
│   ├── confusion_matrix_validation.png
│   └── confusion_matrix_herlev.png
├── logs/
│   └── training.log               # Detailed logs
├── results.json                    # Complete results
└── pipeline_report.json            # End-to-end report
```

---

## ✅ STEP 10: EARLY STOPPING

### Implementation
```python
best_val_acc = 0.0
epochs_without_improvement = 0

if val_acc > best_val_acc:
    best_val_acc = val_acc
    torch.save(model.state_dict(), 'best_model.pth')
    epochs_without_improvement = 0
else:
    epochs_without_improvement += 1

if epochs_without_improvement >= early_stopping_patience:
    print("Early stopping triggered")
    break
```

---

## ✅ STEP 11: FINAL EVALUATION

### Cross-Dataset Evaluation
```python
# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on Herlev (cross-dataset)
test_results = evaluate_model(model, herlev_loader)

# Report metrics
print(f"Herlev Accuracy: {test_results['accuracy']:.4f}")
print(f"Herlev F1-Score: {test_results['f1_score']:.4f}")
print(f"Generalization Gap: {val_acc - test_results['accuracy']:.4f}")
```

---

## 🎯 EXPECTED RESULTS

### Training Performance
- **Validation Accuracy**: 85-95% (target achieved)
- **F1-Score**: 0.85-0.95 (excellent for medical)
- **Training Stability**: Smooth convergence curves

### Cross-Dataset Generalization
- **Herlev Accuracy**: 65-80% (good domain transfer)
- **Generalization Gap**: <20% (acceptable)
- **F1-Score**: 0.70-0.85 (strong generalization)

### Success Indicators
```
🎯 ACHIEVED TARGET: 87.3% accuracy (≥85%)
🏆 ACHIEVED EXCELLENCE: 91.2% accuracy (≥90%)
✅ EXCELLENT generalization (gap < 10%)
```

---

## 🚨 TROUBLESHOOTING

### Common Issues

#### PyTorch DLL Error (Windows)
```
Error: [WinError 1114] A dynamic link library (DLL) initialization routine failed
```
**Solution:**
```bash
# Install CPU version first
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install CUDA version if needed
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA Out of Memory
```
Error: CUDA out of memory
```
**Solution:**
```bash
# Reduce batch size
python train_efficientnet_optimized.py --mode binary --batch-size 8
```

#### Dataset Validation Failure
```
Error: Dataset validation failed
```
**Solution:**
```bash
# Check dataset structure
python -c "from utils.dataset_guard import DatasetGuard; guard = DatasetGuard({'dataset': {'root_dir': 'datasets'}}); guard.print_debug_info('sipakmed')"
```

#### Poor Performance
```
Validation accuracy < 80%
```
**Solution:**
```bash
# Increase training epochs
python train_efficientnet_optimized.py --mode binary --epochs 40

# Adjust hyperparameters
python train_efficientnet_optimized.py --mode binary --dropout 0.5 --label-smoothing 0.05
```

---

## 🚀 EXECUTION COMMANDS

### Full Pipeline (Recommended)
```bash
# Automated setup and training
python setup_and_train.py
```

### Manual Training
```bash
# Binary classification
python train_efficientnet_optimized.py --mode binary

# Multiclass classification
python train_efficientnet_optimized.py --mode multiclass

# Custom parameters
python train_efficientnet_optimized.py --mode binary --epochs 35 --batch-size 16 --dropout 0.5
```

### With Configuration File
```bash
python train_efficientnet_optimized.py --config configs/efficientnet_optimized_config.yaml
```

---

## 📊 MONITORING TRAINING

### Real-time Monitoring
```bash
# Watch training logs
tail -f efficientnet_optimized.log

# Monitor GPU usage (if available)
nvidia-smi -l 1
```

### Key Metrics to Watch
1. **Training Loss**: Should decrease steadily
2. **Validation Accuracy**: Should increase to 85%+
3. **F1-Score**: Should reach 0.85+
4. **Generalization Gap**: Should remain <20%

### Achievement Notifications
```
🎯 ACHIEVED TARGET: Validation accuracy ≥85%
🏆 ACHIEVED EXCELLENCE: Validation accuracy ≥90%
✅ EXCELLENT generalization: Gap <10%
```

---

## 🎉 SUCCESS CRITERIA

### Training Success
- ✅ Validation accuracy: 85-95%
- ✅ F1-score: 0.85-0.95
- ✅ Stable training curves
- ✅ No severe overfitting

### Generalization Success
- ✅ Herlev accuracy: 65-80%
- ✅ Generalization gap: <20%
- ✅ Cross-dataset F1: 0.70-0.85

### Pipeline Success
- ✅ All dependencies installed
- ✅ Dataset validation passed
- ✅ Training completed without errors
- ✅ Results saved and documented

---

## 🔄 NEXT STEPS

### After Successful Training
1. **Analyze Results**: Review all metrics and plots
2. **Compare with Baseline**: Validate improvement claims
3. **Error Analysis**: Examine misclassified samples
4. **Prepare for Hybrid**: Use optimized baseline for comparison
5. **Clinical Validation**: If applicable, validate with medical experts

### Model Deployment
1. **Export Model**: Convert to deployment format
2. **Create Inference Script**: For new image prediction
3. **Performance Testing**: Benchmark on various hardware
4. **Documentation**: Complete training and deployment guide

---

*This complete setup ensures reliable, reproducible training of EfficientNet-B0 with target achievement of 85-95% validation accuracy and strong cross-dataset generalization.*
