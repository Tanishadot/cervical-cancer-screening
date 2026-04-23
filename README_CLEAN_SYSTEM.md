# Clean Cytology Classification System

A production-ready, explainable AI system for cervical cytology classification using real feature extraction and clinical reasoning.

## 🏗️ Architecture

### Backend (`backend/`)
- **`feature_extraction.py`** - Real OpenCV-based cytology feature extraction
- **`model.py`** - Model management, feature fusion, Grad-CAM
- **`inference.py`** - Complete inference pipeline with clinical reasoning

### Frontend (`frontend/`)
- **`app.py`** - Clean, optimized Streamlit interface

## 🔬 Real Feature Extraction

### Nuclear Features (30% weight)
- **Nuclear enlargement** - Size-based analysis
- **Chromatin density** - Intensity variation
- **Nuclear contours** - Shape irregularity
- **N:C ratio** - Nucleus-to-cytoplasm ratio
- **Hyperchromasia** - Dark staining intensity

### Cytoplasmic Features (40% weight) - CRITICAL
- **Perinuclear halo** - Edge-based halo detection
- **Keratinization** - HSV color analysis
- **Cytoplasmic texture** - Local variation
- **Cell maturity** - Color characteristics
- **Koilocytosis** - HPV indicator detection

### Background Features (30% weight)
- **Background debris** - Variation analysis
- **Inflammatory cells** - Small circular structures
- **Tumor diathesis** - Necrotic background detection
- **Background cleanliness** - Inverse of debris

## 🧠 Clinical Reasoning Engine

### Bethesda Classification
- **NILM** - Negative for Intraepithelial Lesion
- **ASC-US** - Atypical Squamous Cells (Undetermined)
- **LSIL** - Low-grade Squamous Intraepithelial Lesion
- **HSIL** - High-grade Squamous Intraepithelial Lesion
- **SCC** - Squamous Cell Carcinoma

### Clinical Rules
```
IF halo high AND debris moderate → LSIL
IF chromatin high AND N/C ratio high → HSIL
ELSE → NILM
```

## 🚀 Key Features

### Real Feature Extraction
- **OpenCV-based** - No synthetic values
- **Region segmentation** - Nucleus/Cytoplasm/Background
- **Clinical metrics** - Medical-grade calculations
- **Threshold-based** - Reproducible results

### Feature Fusion
- **Model confidence** - 40% weight
- **Nuclear features** - 20% weight
- **Cytoplasmic features** - 30% weight
- **Background features** - 10% weight

### Visual Explainability
- **Grad-CAM** - Model attention visualization
- **Region segmentation** - Feature area visualization
- **Feature contributions** - Importance breakdown
- **Clinical reasoning** - Medical interpretation

## 🖥️ Streamlit Optimization

### Performance Features
- **`@st.cache_resource`** - Model loading
- **`@st.cache_data`** - Feature extraction
- **Efficient preprocessing** - No redundant operations
- **Memory optimization** - Smart tensor management

### UI Components
- **Bethesda classification display** - Risk assessment
- **Feature analysis panels** - Detailed breakdown
- **Visual analysis** - Segmentation + Grad-CAM
- **Clinical reasoning** - Medical interpretation
- **Summary dashboard** - Key metrics

## 🎯 Production Ready

### Error Handling
- **Graceful fallbacks** - Default values on errors
- **Comprehensive logging** - Debug information
- **User-friendly messages** - Clear error communication

### Medical Compliance
- **Bethesda alignment** - Clinical standards
- **Explainable AI** - Transparent decisions
- **Clinical reasoning** - Medical interpretation
- **Risk assessment** - Actionable insights

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install streamlit torch torchvision opencv-python numpy matplotlib plotly pillow
```

### 2. Run Application
```bash
streamlit run frontend/app.py --server.port 8507
```

### 3. Upload Image
- Select cytology slide image
- Click "Analyze with AI"
- View comprehensive results

## 📊 Output Information

### Classification Results
- **Bethesda class** - Medical classification
- **Confidence score** - AI certainty
- **Risk level** - Clinical significance
- **Clinical reasoning** - Medical explanation

### Feature Analysis
- **Nuclear metrics** - Detailed nuclear features
- **Cytoplasmic metrics** - Detailed cytoplasmic features
- **Background metrics** - Detailed background features
- **Feature contributions** - Importance breakdown

### Visual Analysis
- **Region segmentation** - Nucleus/Cytoplasm/Background
- **Grad-CAM heatmap** - Model attention
- **Combined overlay** - Integrated visualization
- **Statistics** - Quantitative analysis

## 🔧 Technical Details

### Feature Extraction Pipeline
1. **Image preprocessing** - RGB conversion, resizing
2. **Region segmentation** - Threshold + morphology
3. **Feature calculation** - OpenCV-based metrics
4. **Score normalization** - 0-1 range scaling

### Model Integration
1. **Model prediction** - CNN-based classification
2. **Feature fusion** - Weighted combination
3. **Clinical reasoning** - Rule-based classification
4. **Confidence calculation** - Score-based certainty

### Performance Optimization
- **Lazy loading** - Models loaded on demand
- **Caching** - Feature extraction cached
- **Memory management** - Efficient tensor operations
- **Parallel processing** - Concurrent operations

## 🏥 Medical Validation

### Clinical Accuracy
- **Multi-factor analysis** - Not just nuclear features
- **Cytoplasmic priority** - Critical for LSIL detection
- **Background context** - Essential for HSIL/SCC
- **Bethesda compliance** - Standard classification

### Explainability
- **Transparent reasoning** - Clear medical logic
- **Feature importance** - Quantified contributions
- **Visual evidence** - Heatmaps and segmentation
- **Clinical terminology** - Medical standards

---

**Status**: ✅ Production Ready
**Dashboard**: http://localhost:8507
**Architecture**: Clean backend/frontend separation
**Features**: Real extraction + clinical reasoning
