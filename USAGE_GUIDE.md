# Usage Guide for Cervical Cancer Classification Pipeline

## 🚀 Quick Start

### 1. Environment Setup
**Automated Setup (Recommended):**
```bash
python setup_environment.py
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Verify setup
python demo_dataset.py
```

### 2. Dataset Preparation
The datasets should already be organized as follows:

```
datasets/
├── sipakmed/
│   └── archive/
│       ├── im_Superficial-Intermediate/
│       │   └── im_Superficial-Intermediate/  # Images here
│       ├── im_Parabasal/
│       │   └── im_Parabasal/  # Images here
│       ├── im_Koilocytotic/
│       │   └── im_Koilocytotic/  # Images here
│       ├── im_Metaplastic/
│       │   └── im_Metaplastic/  # Images here
│       └── im_Dyskeratotic/
│           └── im_Dyskeratotic/  # Images here
└── herlev/
    └── archive (1)/
        ├── normal_superficiel/     # Images here
        ├── normal_intermediate/    # Images here
        ├── normal_columnar/        # Images here
        ├── light_dysplastic/       # Images here
        ├── moderate_dysplastic/    # Images here
        ├── severe_dysplastic/      # Images here
        └── carcinoma_in_situ/      # Images here
```

**Dataset Statistics:**
- **SIPaKMeD**: 966 images (5 classes)
- **Herlev**: 1,834 images (7 classes)
- **Total**: 2,800 images

### 3. Verify Dataset Loading
```bash
# Test dataset loading and get statistics
python demo_dataset.py

# Test with PyTorch integration
python test_dataset.py
```

### 3. Run the Pipeline

#### Complete Pipeline (Recommended)
```bash
python main.py
```

#### Specific Phases
```bash
# Training only
python main.py --mode train

# Evaluation only
python main.py --mode eval

# XAI analysis only
python main.py --mode xai
```

## 🎯 Experiment Configurations

### Experiment A: Binary Classification
```bash
# Binary classification (Normal vs Abnormal)
python main.py --binary

# Or with specific model
python main.py --binary --model efficientnet_b0
```

### Experiment B: Multi-class Classification
```bash
# Multi-class classification
python main.py --multiclass

# With specific model and epochs
python main.py --multiclass --model resnet50 --epochs 50
```

## ⚙️ Configuration

### Edit Configuration File
Modify `configs/config.yaml` to customize:

```yaml
# Model selection
model:
  architecture: "efficientnet_b0"  # Options: efficientnet_b0, resnet50, swin_transformer

# Training parameters
training:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 32

# Dataset settings
dataset:
  classification_mode: "binary"  # Options: binary, multiclass
  train_dataset: "sipakmed"
  test_dataset: "herlev"
```

### Command Line Overrides
```bash
# Override any configuration parameter
python main.py --model swin_transformer --epochs 200 --batch-size 64 --lr 0.0001
```

## 📊 Available Models

| Model | Parameters | Memory | Speed | Accuracy |
|-------|-------------|---------|-------|----------|
| EfficientNet-B0 | ~5M | Low | Fast | High |
| ResNet50 | ~25M | Medium | Medium | High |
| Swin Transformer | ~28M | High | Slow | Very High |

## 🔍 Advanced Features

### Cross-Dataset Evaluation
The pipeline automatically evaluates cross-dataset generalization:
- Train on SIPaKMeD → Test on Herlev
- Train on Herlev → Test on SIPaKMeD

### Explainable AI (XAI)
Enable XAI analysis with multiple methods:
- Grad-CAM
- Grad-CAM++
- Attention Rollout (for Swin Transformer)

### Data Augmentation
Strong augmentations for staining variations:
- Color jitter and shifts
- Contrast and brightness adjustments
- Geometric transformations
- Elastic deformations

## 📈 Output Analysis

### Training Outputs
```
outputs/
├── models/
│   └── best_model.pth
├── logs/
│   ├── experiment_history.json
│   └── test_results.json
└── visualizations/
    ├── training_curves.png
    ├── confusion_matrix.png
    └── roc_curves.png
```

### XAI Outputs
```
outputs/xai/
├── sample_0001.png
├── sample_0002.png
└── xai_results.json
```

### Cross-Dataset Results
```
outputs/cross_dataset/
├── sipakmed_to_herlev_results.json
├── herlev_to_sipakmed_results.json
└── summary_comparison.png
```

## 🎛️ Performance Optimization

### GPU Usage
```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0
python main.py

# Mixed precision training (enabled by default)
# Set in config: training.use_amp: true
```

### Memory Optimization
```bash
# Reduce batch size if OOM
python main.py --batch-size 16

# Use gradient accumulation
# Set in config: advanced.gradient_accumulation_steps: 4
```

### Speed Optimization
```bash
# Increase workers for data loading
# Set in config: dataset.num_workers: 8

# Use smaller image size
# Set in config: augmentation.image_size: 128
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python main.py --batch-size 16

# Enable gradient accumulation
# Edit config.yaml:
# advanced:
#   gradient_accumulation_steps: 4
```

#### Dataset Not Found
```bash
# Check dataset structure
ls -la datasets/sipakmed/
ls -la datasets/herlev/

# Update paths in config.yaml
dataset:
  root_dir: "/path/to/your/datasets"
```

#### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"
```

### Performance Issues

#### Slow Training
- Increase `num_workers` in config
- Enable mixed precision (`use_amp: true`)
- Use smaller image size

#### Poor Accuracy
- Try different model architectures
- Increase training epochs
- Adjust learning rate
- Enable stronger augmentations

#### Overfitting
- Increase dropout rate
- Add more data augmentation
- Use early stopping (enabled by default)
- Reduce model complexity

## 📚 Example Workflows

### Research Workflow
```bash
# 1. Binary classification with EfficientNet
python main.py --binary --model efficientnet_b0 --epochs 100

# 2. Multi-class classification with ResNet
python main.py --multiclass --model resnet50 --epochs 150

# 3. Transformer-based approach
python main.py --binary --model swin_transformer --epochs 200 --lr 0.0001
```

### Production Workflow
```bash
# Train best model
python main.py --binary --model efficientnet_b0 --epochs 200

# Evaluate on external dataset
python main.py --mode eval --config configs/production_config.yaml

# Generate XAI visualizations
python main.py --mode xai --xai-num-samples 100
```

### Comparison Study
```bash
# Train all models
for model in efficientnet_b0 resnet50 swin_transformer; do
    python main.py --model $model --epochs 100 --experiment-name "study_$model"
done

# Compare results
python scripts/compare_results.py outputs/study_*/test_results.json
```

## 📖 Advanced Configuration

### Custom Augmentations
```yaml
augmentation:
  custom_augmentations:
    - name: "GaussianBlur"
      params:
        blur_limit: 7
        p: 0.3
    - name: "GridDistortion"
      params:
        num_steps: 5
        distort_limit: 0.3
        p: 0.5
```

### Advanced Training
```yaml
advanced:
  gradient_clipping:
    enabled: true
    max_norm: 1.0
  
  ema:
    enabled: true
    decay: 0.999
  
  checkpointing:
    save_top_k: 5
    monitor: "val_f1_macro"
```

### XAI Configuration
```yaml
xai:
  methods:
    - "grad_cam"
    - "grad_cam_plus"
    - "attention_rollout"
  
  target_layers:
    - "backbone.features.8"  # For EfficientNet
    - "backbone.layer4"      # For ResNet
    - "backbone.stages.3"   # For Swin
  
  save_individual_visualizations: true
```

## 🎓 Best Practices

1. **Data Preparation**
   - Ensure balanced datasets
   - Apply consistent preprocessing
   - Use stain normalization for cross-dataset evaluation

2. **Model Selection**
   - Start with EfficientNet-B0 for baseline
   - Use ResNet50 for balanced performance
   - Try Swin Transformer for best accuracy

3. **Training**
   - Use early stopping to prevent overfitting
   - Monitor validation metrics closely
   - Experiment with learning rates

4. **Evaluation**
   - Always test on external datasets
   - Use multiple metrics (accuracy, F1, AUC)
   - Generate confusion matrices

5. **XAI Analysis**
   - Visualize attention maps for model understanding
   - Compare different XAI methods
   - Use visualizations for clinical validation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration documentation
3. Examine the log files in `outputs/logs/`
4. Create an issue on the project repository

---

**Happy Research! 🧬🔬**
