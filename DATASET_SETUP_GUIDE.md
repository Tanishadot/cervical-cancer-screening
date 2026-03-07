# Dataset Setup Guide for Cervical Cancer Classification

## 🎯 Overview

This guide explains how to automatically download and prepare medical imaging datasets for the cervical cancer classification pipeline using the Kaggle downloader.

## 📋 Prerequisites

### 1. Kaggle Account
- Create a free account at [https://www.kaggle.com/](https://www.kaggle.com/)
- Sign in to your account

### 2. Kaggle API Key
The script will automatically check for your API key. If not found, follow these steps:

1. Go to [https://www.kaggle.com/](https://www.kaggle.com/)
2. Sign in to your account
3. Click on your profile picture → **Account**
4. Scroll down to the **API** section
5. Click **"Create New API Token"**
6. Download the `kaggle.json` file
7. Place it in the correct location:

**Windows:**
```
C:/Users/<YourUsername>/.kaggle/kaggle.json
```

**Mac/Linux:**
```
~/.kaggle/kaggle.json
```

The `kaggle.json` file should contain:
```json
{
  "username": "your_kaggle_username",
  "key": "your_api_key_here"
}
```

## 🚀 Quick Start

### 1. Test Setup
```bash
# Test the Kaggle downloader
python test_kaggle_setup.py
```

### 2. Download SIPaKMeD Dataset
```bash
# Download and setup SIPaKMeD dataset
python setup_dataset.py

# Or specify the dataset explicitly
python setup_dataset.py --dataset sipakmed
```

### 3. Verify Setup
```bash
# Verify dataset structure
python setup_dataset.py --verify

# Show dataset information
python setup_dataset.py --info
```

## 📊 Available Datasets

### SIPaKMeD Dataset
- **Purpose**: Training dataset for cervical cancer classification
- **Classes**: 5 classes
  - `superficial_intermediate`
  - `parabasal`
  - `koilocytotic`
  - `metaplastic`
  - `dyskeratotic`
- **Size**: ~408 images
- **Source**: Multiple Kaggle datasets

### Herlev Dataset
- **Purpose**: External evaluation dataset
- **Classes**: 7 classes
  - `normal_superficial`
  - `normal_intermediate`
  - `normal_columnar`
  - `mild_dysplasia`
  - `moderate_dysplasia`
  - `severe_dysplasia`
  - `carcinoma_in_situ`
- **Size**: ~917 images
- **Status**: Manual setup required

## 🛠️ Advanced Usage

### Command Line Options
```bash
python setup_dataset.py --help
```

**Options:**
- `--dataset {sipakmed,herlev,all}`: Specify dataset to download
- `--force`: Force re-download even if dataset exists
- `--verify`: Verify dataset structure only
- `--info`: Show dataset information
- `--project-root PATH`: Specify project root directory

### Examples
```bash
# Download all datasets
python setup_dataset.py --dataset all

# Force re-download
python setup_dataset.py --force

# Verify existing setup
python setup_dataset.py --verify

# Show dataset information
python setup_dataset.py --info

# Custom project root
python setup_dataset.py --project-root /path/to/project
```

## 📁 Expected Directory Structure

After successful setup, your project structure should look like:

```
project_root/
├── datasets/
│   ├── sipakmed/
│   │   ├── superficial_intermediate/
│   │   │   ├── image001.jpg
│   │   │   ├── image002.jpg
│   │   │   └── ...
│   │   ├── parabasal/
│   │   │   ├── image001.jpg
│   │   │   └── ...
│   │   ├── koilocytotic/
│   │   ├── metaplastic/
│   │   └── dyskeratotic/
│   └── herlev/
│       ├── normal_superficial/
│       ├── normal_intermediate/
│       ├── normal_columnar/
│       ├── mild_dysplasia/
│       ├── moderate_dysplasia/
│       ├── severe_dysplasia/
│       └── carcinoma_in_situ/
├── main.py
├── setup_dataset.py
└── ...
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Kaggle CLI Not Found
**Problem**: `Kaggle CLI not found or not accessible`

**Solution**: The script will automatically install Kaggle CLI. If installation fails:
```bash
# Manual installation
pip install kaggle

# Or upgrade existing installation
pip install --upgrade kaggle
```

#### 2. API Key Not Found
**Problem**: `Kaggle API key not found`

**Solution**: Follow the API key setup steps above. Ensure the file is placed in:
- Windows: `C:/Users/<YourUsername>/.kaggle/kaggle.json`
- Mac/Linux: `~/.kaggle/kaggle.json`

#### 3. Download Fails
**Problem**: Dataset download fails

**Solutions**:
- Check internet connection
- Verify Kaggle account is active
- Try alternative dataset names
- Check if dataset is publicly available

```bash
# Check available datasets
kaggle datasets list -s cervical

# Download specific dataset
kaggle datasets download -d andrewmvd/cervical-cancer-dataset
```

#### 4. Permission Errors
**Problem**: Permission denied when creating directories

**Solution**: Run with appropriate permissions or choose different directory:
```bash
# Custom project root with write permissions
python setup_dataset.py --project-root ~/projects/cervical_cancer
```

#### 5. Extraction Fails
**Problem**: Zip file extraction fails

**Solution**: Manual extraction:
```bash
# Navigate to datasets directory
cd datasets/sipakmed/

# Extract manually (Windows)
# Right-click on zip file → Extract All

# Extract manually (Mac/Linux)
unzip *.zip
```

### Debug Mode
For detailed debugging, check the log file:
```bash
tail -f dataset_setup.log
```

## 📈 Dataset Statistics

### SIPaKMeD Dataset
```
superficial_intermediate: ~126 images
parabasal: ~78 images
koilocytotic: ~79 images
metaplastic: ~75 images
dyskeratotic: ~50 images
Total: ~408 images
```

### Herlev Dataset
```
normal_superficial: ~244 images
normal_intermediate: ~182 images
normal_columnar: ~145 images
mild_dysplasia: ~78 images
moderate_dysplasia: ~82 images
severe_dysplasia: ~73 images
carcinoma_in_situ: ~113 images
Total: ~917 images
```

## 🔄 Dataset Organization

### Automatic Organization
The script automatically:
1. Downloads the dataset from Kaggle
2. Extracts zip files
3. Organizes images into correct class folders
4. Removes temporary files
5. Verifies final structure

### Manual Organization
If automatic organization fails, manually organize images:

```bash
# Example for SIPaKMeD
mkdir -p datasets/sipakmed/superficial_intermediate
mkdir -p datasets/sipakmed/parabasal
mkdir -p datasets/sipakmed/koilocytotic
mkdir -p datasets/sipakmed/metaplastic
mkdir -p datasets/sipakmed/dyskeratotic

# Move images to appropriate folders
# (This depends on the specific dataset structure)
```

## 🎯 Next Steps

After dataset setup is complete:

### 1. Verify Setup
```bash
python setup_dataset.py --verify
```

### 2. Run Pipeline
```bash
# Binary classification
python main.py --binary

# Multi-class classification
python main.py --multiclass

# Specific model
python main.py --model efficientnet_b0 --epochs 100
```

### 3. Check Results
```bash
# View outputs
ls outputs/

# Check training logs
tail -f outputs/logs/*.log
```

## 📞 Support

### Common Questions

**Q: Can I use my own dataset?**
A: Yes! Organize your images in the same structure and place them in the `datasets/` folder.

**Q: What if I don't have a Kaggle account?**
A: You can create a free account at kaggle.com. The API key is required for dataset access.

**Q: Can I download datasets from other sources?**
A: Yes, manually download and organize datasets in the expected structure.

**Q: How do I update the dataset?**
A: Use the `--force` flag to re-download:
```bash
python setup_dataset.py --force
```

### Getting Help

1. Check the troubleshooting section above
2. Review the log file: `dataset_setup.log`
3. Verify your Kaggle API key setup
4. Check internet connection and Kaggle account status

---

**Happy Research! 🧬🔬**

Once your datasets are set up, you're ready to train cervical cancer classification models with state-of-the-art deep learning techniques!
