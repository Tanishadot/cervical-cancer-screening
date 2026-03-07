# Cervical Cancer Classification Research Pipeline

A comprehensive deep learning research pipeline for cervical cancer screening using Pap smear cell datasets. This project implements state-of-the-art computer vision techniques for automated cervical cancer detection with explainable AI capabilities.

## 🎯 Project Overview

This pipeline supports two main experiments:

- **Experiment A**: Binary classification (Normal vs Abnormal)
- **Experiment B**: Multi-class classification with original labels

## 📊 Datasets

### Supported Datasets
1. **SIPaKMeD Dataset** - Training dataset with 5 classes
2. **Herlev Dataset** - External evaluation dataset with 7 classes

### Class Mapping

#### SIPaKMeD Classes
- Superficial-Intermediate
- Parabasal
- Koilocytotic
- Metaplastic
- Dyskeratotic

#### Herlev Classes
- Normal superficial
- Normal intermediate
- Normal columnar
- Mild dysplasia
- Moderate dysplasia
- Severe dysplasia
- Carcinoma in situ

#### Binary Classification Mapping
- **NORMAL**: Superficial-Intermediate, Parabasal, Normal superficial, Normal intermediate, Normal columnar
- **ABNORMAL**: Koilocytotic, Metaplastic, Dyskeratotic, Mild dysplasia, Moderate dysplasia, Severe dysplasia, Carcinoma in situ

## 🚀 Features

### Model Architectures
- **EfficientNet-B0** - Efficient and accurate
- **ResNet50** - Classic deep residual network
- **Swin Transformer** - State-of-the-art vision transformer

### Advanced Preprocessing
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Color normalization
- Macenko stain normalization
- Advanced artifact removal

### Data Augmentation
- Strong color augmentations for staining variations
- Geometric transformations
- Elastic transforms
- Test-time augmentation support

### Training Features
- Transfer learning with pretrained weights
- Early stopping with patience
- Learning rate scheduling
- Class weighting for imbalanced datasets
- Mixed precision training
- Gradient clipping

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- AUC-ROC curves
- Confusion matrices
- Cross-dataset generalization assessment

### Explainable AI (XAI)
- Grad-CAM visualizations
- Grad-CAM++ enhanced visualizations
- Attention Rollout for Swin Transformer
- Comprehensive heatmap overlays

## 📁 Project Structure

```
project/
├── datasets/
│   ├── sipakmed/
│   └── herlev/
├── models/
│   └── model_factory.py
├── preprocessing/
│   ├── image_preprocessing.py
│   ├── transforms.py
│   └── stain_normalization.py
├── training/
│   ├── trainer.py
│   └── cross_dataset_eval.py
├── xai/
│   └── grad_cam.py
├── utils/
│   ├── label_mapping.py
│   ├── config_manager.py
│   └── visualization.py
├── configs/
│   └── config.yaml
├── outputs/
│   ├── models/
│   ├── visualizations/
│   ├── xai/
│   └── logs/
└── main.py
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Automated Setup (Recommended)
1. Clone the repository
2. Run the automated setup script:
```bash
python setup_environment.py
```

This will:
- Create a virtual environment (`venv/`)
- Install CPU-compatible PyTorch
- Install all other dependencies
- Verify the setup works

### Manual Setup
If automated setup fails, follow these steps:

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install PyTorch (CPU version):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

5. Verify setup:
```bash
python demo_dataset.py
```

### Dataset Setup
The datasets should already be in the `datasets/` folder:
- SIPaKMeD: 966 images (5 classes)
- Herlev: 1,834 images (7 classes)

If you need to download datasets:
```bash
python setup_dataset.py
```

### Dependencies
- torch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- albumentations >= 1.3.0
- tqdm >= 4.64.0
- seaborn >= 0.11.0
- grad-cam >= 1.4.0
- einops >= 0.6.0
- PyYAML >= 6.0
- kaggle >= 1.5.0

### Dataset Setup
See [DATASET_SETUP_GUIDE.md](DATASET_SETUP_GUIDE.md) for detailed instructions on downloading and preparing datasets.

**Quick Setup:**
```bash
# Test configuration
python test_kaggle_setup.py

# Download datasets
python setup_dataset.py

# Verify structure
python setup_dataset.py --verify
```

## 🎮 Usage

### Basic Usage
```bash
# Run complete pipeline
python main.py

# Run specific phases
python main.py --mode train
python main.py --mode eval
python main.py --mode xai

# Binary classification
python main.py --binary

# Multi-class classification
python main.py --multiclass

# Specify model
python main.py --model efficientnet_b0
python main.py --model resnet50
python main.py --model swin_transformer
```

### Advanced Usage
```bash
# Custom configuration
python main.py --config configs/custom_config.yaml

# Override parameters
python main.py --epochs 50 --batch-size 64 --lr 0.0001

# Set random seed
python main.py --seed 123
```

### Configuration
Edit `configs/config.yaml` to customize:
- Model architecture and hyperparameters
- Training settings
- Data augmentation parameters
- Evaluation configurations
- XAI methods

## 📈 Results and Outputs

The pipeline generates comprehensive outputs:

### Model Outputs
- Trained model weights (`outputs/models/`)
- Training checkpoints
- Best model based on validation metrics

### Visualizations
- Training curves (loss, accuracy, F1-score)
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Class distribution plots

### XAI Outputs
- Grad-CAM heatmaps
- Grad-CAM++ enhanced visualizations
- Attention rollouts (for Swin Transformer)
- Overlay visualizations on original images

### Evaluation Reports
- Cross-dataset performance metrics
- Per-class analysis
- Generalization assessment
- Comprehensive JSON reports

## 🔬 Research Applications

### Medical Research
- Automated cervical cancer screening
- Comparative analysis of deep learning architectures
- Cross-dataset generalization studies
- Explainable AI for medical diagnosis

### Technical Features
- Reproducible research with seed control
- Comprehensive logging and monitoring
- Flexible configuration system
- Modular and extensible design

## 📊 Performance Metrics

The pipeline tracks comprehensive metrics:

### Classification Metrics
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-score (macro and weighted)
- AUC-ROC

### Cross-Dataset Evaluation
- Generalization performance
- Domain adaptation analysis
- Per-class transferability

### XAI Analysis
- Attention map quality
- Feature importance visualization
- Model interpretability assessment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{cervical_cancer_pipeline,
  title={Deep Learning Pipeline for Cervical Cancer Classification},
  author={Your Name},
  year={2024},
  description={Comprehensive deep learning pipeline for cervical cancer screening with XAI}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Enable gradient accumulation
   - Use mixed precision training

2. **Dataset Not Found**
   - Check dataset paths in config
   - Ensure proper directory structure
   - Verify dataset integrity

3. **Grad-CAM Issues**
   - Ensure target layers are accessible
   - Check model compatibility
   - Verify input tensor dimensions

### Performance Tips

1. **Training Speed**
   - Use mixed precision (`use_amp: true`)
   - Optimize data loading (`num_workers`)
   - Enable GPU memory optimization

2. **Model Performance**
   - Experiment with different architectures
   - Tune hyperparameters
   - Use strong data augmentation
   - Apply stain normalization

3. **Generalization**
   - Use cross-dataset evaluation
   - Apply domain adaptation techniques
   - Use ensemble methods

## 🔮 Future Enhancements

- [ ] Additional model architectures (Vision Transformers, CNN-Transformer hybrids)
- [ ] Advanced stain normalization techniques
- [ ] Semi-supervised learning capabilities
- [ ] Active learning for annotation efficiency
- [ ] Deployment optimization for clinical use
- [ ] Integration with digital pathology systems

## 📧 Contact

For questions, issues, or collaborations:
- Create an issue on GitHub
- Contact: [your-email@example.com]

---

**Disclaimer**: This research pipeline is for academic and research purposes. For clinical applications, ensure proper validation and regulatory compliance.
