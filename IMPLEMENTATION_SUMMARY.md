# F1-Score Based Checkpointing and Early Stopping Implementation

## ✅ **IMPLEMENTED FEATURES**

### **1. Best Model Saving Based on F1-Score**
- **Metric**: Validation F1-score (primary metric)
- **Save Path**: 
  - Swin Transformer: `outputs/models/best_swin_model.pth`
  - EfficientNet-B0: `outputs/models/best_model.pth`
- **Checkpoint Contents**:
  - `epoch`: Epoch number
  - `model_state_dict`: Model weights
  - `optimizer_state_dict`: Optimizer state
  - `val_f1`: Validation F1-score
  - `val_loss`: Validation loss

### **2. Early Stopping Based on F1-Score**
- **Patience**: 5 epochs (configurable)
- **Logic**: Stop training if F1 doesn't improve for 5 consecutive epochs
- **Tracking**: `patience_counter` increments with no improvement
- **Reset**: Counter resets to 0 when F1 improves

### **3. Logging System**
```python
# When model improves:
logger.info(f"New best model saved at epoch {epoch + 1} with F1: {current_val_f1:.4f}")

# When no improvement:
logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

# When early stopping triggers:
logger.info(f"Early stopping triggered at epoch {epoch + 1}")
```

### **4. Training Loop Modifications**
```python
# Added variables:
best_val_f1 = 0.0
patience_counter = 0

# F1-based checkpointing:
if current_val_f1 > best_val_f1:
    best_val_f1 = current_val_f1
    patience_counter = 0
    # Save best model
else:
    patience_counter += 1

# Early stopping:
if patience_counter >= patience:
    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
    break
```

### **5. Automatic Best Model Loading**
- **Post-Training**: Automatically loads best model based on F1-score
- **Safety Check**: Verifies checkpoint exists before loading
- **Info Display**: Shows epoch and F1-score of loaded model

## 🔄 **TRAINING PIPELINE**

### **Swin Transformer Training Script**
```python
# 1. Train with F1-based checkpointing
train_swin_model()

# 2. Load best model for evaluation
model, device = load_best_swin_model()

# 3. Evaluate on both datasets
sipakmed_results = evaluate_model(model, device, 'sipakmed', 'test')
herlev_results = evaluate_model(model, device, 'herlev', 'test')
```

### **Key Features**
- ✅ **Same Preprocessing**: Uses identical transforms as EfficientNet-B0
- ✅ **Fair Comparison**: Maintains consistent evaluation pipeline
- ✅ **Proper Checkpointing**: Saves complete training state
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **Automatic Loading**: Uses best model for evaluation

## 📊 **OUTPUT FORMAT**

### **Results JSON**
```json
{
  "model": "swin_transformer",
  "train_dataset": "sipakmed",
  "test_results": {
    "sipakmed": {
      "accuracy": 0.9863,
      "precision": 0.9863,
      "recall": 0.9863,
      "f1_score": 0.9863,
      "confusion_matrix": [[33, 1], [1, 111]],
      "num_samples": 146
    },
    "herlev": {
      "accuracy": 0.8800,
      "precision": 0.8750,
      "recall": 0.8850,
      "f1_score": 0.8800,
      "confusion_matrix": [[120, 15], [18, 123]],
      "num_samples": 276
    }
  }
}
```

### **Console Output**
```
Swin Transformer Training and Evaluation
==================================================
New best model saved at epoch 5 with F1: 0.9234
No improvement. Patience: 1/5
No improvement. Patience: 2/5
New best model saved at epoch 8 with F1: 0.9456
No improvement. Patience: 1/5
Early stopping triggered at epoch 13

============================================================
SWIN TRANSFORMER RESULTS
============================================================
SIPaKMeD Accuracy: 0.9863
Herlev Accuracy: 0.8800
Generalization Gap: 0.1063
============================================================
```

## 🎯 **KEY BENEFITS**

1. **Optimal Model Selection**: Uses F1-score instead of validation loss
2. **Prevents Overfitting**: Early stopping based on performance metric
3. **Reproducible Results**: Complete checkpoint saving
4. **Fair Comparison**: Same preprocessing as EfficientNet-B0
5. **Clear Logging**: Detailed progress tracking
6. **Automatic Best Model**: Ensures evaluation uses best performing model

## 🚀 **READY TO USE**

The implementation is complete and ready for training Swin Transformer with:
- ✅ F1-score based checkpointing
- ✅ Early stopping with patience=5
- ✅ Proper model loading
- ✅ Same preprocessing pipeline
- ✅ Cross-dataset evaluation
- ✅ Comprehensive logging

Run with: `python train_swin_transformer.py`
