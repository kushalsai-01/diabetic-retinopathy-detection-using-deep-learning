# Multimodal Diabetic Retinopathy Detection - Implementation Plan

## Overview
Converting the DR detection system to a **multimodal** approach combining:
1. **Fundus images** (retina photos)
2. **Clinical/Patient data** (age, diabetes duration, HbA1c, blood pressure, etc.)

## Why Multimodal?
- 5 DR stages alone may not be sufficient for accurate diagnosis
- Clinical factors significantly influence DR progression
- Combining image + patient data improves prediction accuracy
- Real-world clinical decision-making uses both modalities

---

## 📊 Phase 1: Dataset Structure

### Required Patient Data (Tabular Features)
```
- age: Patient age (years)
- gender: Male/Female (0/1)
- diabetes_duration: Years since diabetes diagnosis
- hba1c: Glycated hemoglobin level (%)
- blood_pressure_systolic: Systolic BP (mmHg)
- blood_pressure_diastolic: Diastolic BP (mmHg)
- bmi: Body Mass Index
- smoking_status: 0=Never, 1=Former, 2=Current
- insulin_treatment: 0=No, 1=Yes
```

### CSV Structure Example
```csv
image_path,diagnosis,age,gender,diabetes_duration,hba1c,bp_sys,bp_dia,bmi,smoking,insulin
img_001.jpg,2,54,1,8,7.2,140,90,28.5,0,1
img_002.jpg,0,45,0,3,6.1,120,80,24.2,0,0
```

---

## 🔄 Phase 2: Enhanced Augmentation Strategy

### Medical Image-Specific Augmentations
```python
OPTIMAL_AUGMENTATIONS = {
    "rotation": ±15°,          # Conservative for fundus images
    "horizontal_flip": 0.5,     # Eyes can appear flipped
    "vertical_flip": 0.5,       # Valid for fundus
    "brightness": ±0.15,        # Lighting variations
    "contrast": ±0.15,          # Camera/quality differences
    "color_jitter": mild,       # Fundus color variations
    "gaussian_blur": 0.2,       # Out-of-focus images
    "clahe": 0.3,              # Enhance retinal features
    "crop_scale": 0.9-1.0,     # Mild zooming
}
```

### Why ±15° instead of ±5°?
- Fundus cameras can have slight misalignment
- ±5° is too conservative, won't create enough variation
- ±15° is medically validated for retinal imaging
- ±180° (current) is too aggressive and unrealistic

---

## 🏗️ Phase 3: Multimodal Architecture

### Model Structure
```
Input 1: Fundus Image (3, 448, 448)
    ↓
[CNN/ViT Backbone]
    ↓
Image Features (2048-dim)
    
Input 2: Clinical Data (9 features) [OPTIONAL]
    ↓
[MLP with Dropout]
    ↓
Clinical Features (128-dim)

[Fusion Layer]
    ↓
Combined Features
    ↓
[Classifier Head]
    ↓
5 DR Classes
```

### Key Innovation: Optional Tabular Input
- During **training**: Use both image + clinical data
- Apply **50% tabular dropout** randomly during training
- Model learns to work with OR without clinical data
- During **inference**: Works with image only OR image + clinical

---

## 🎯 Phase 4: Training Strategy

### 4.1 Data Augmentation
- Generate **3-5x more training samples** via augmentation
- Use online augmentation during training (efficient)
- Total effective dataset: Original × Augmentation factor

### 4.2 Multimodal Training
```python
Training Loop:
1. Load image + clinical data
2. 50% chance: Set clinical data to zeros (simulate missing)
3. Forward pass with both modalities
4. Model learns to rely primarily on image
5. Clinical data becomes supplementary information
```

### 4.3 Loss Function
- Use **Focal Loss** or **Class-Balanced Loss**
- Handle class imbalance (DR stages distribution)
- Optional: Add auxiliary loss for feature consistency

---

## 📝 Implementation Checklist

### ✅ Step 1: Prepare Dataset
- [ ] Collect/generate patient clinical data
- [ ] Create CSV with image_path + diagnosis + clinical features
- [ ] Validate data completeness and ranges
- [ ] Split: Train/Val/Test (80/10/10)

### ✅ Step 2: Update Code
- [ ] Modify `dataset.py`: Add tabular feature loading
- [ ] Update `transforms.py`: Better augmentation (±15°)
- [ ] Create `multimodal_model.py`: Fusion architecture
- [ ] Update `datamodule.py`: Pass tabular features
- [ ] Modify `train.py`: Handle multimodal inputs

### ✅ Step 3: Training
- [ ] Train multimodal model with dropout
- [ ] Monitor: Loss, accuracy, quadratic kappa
- [ ] Validate on val set (with/without clinical data)
- [ ] Save best checkpoint

### ✅ Step 4: Inference
- [ ] Test with full data (image + clinical)
- [ ] Test with image only (clinical = zeros)
- [ ] Compare performance differences
- [ ] Deploy model with flexible input

---

## 💡 Expected Improvements

### With Multimodal Approach
- **+5-10% accuracy** with full data
- **Maintains baseline** with image only
- Better generalization on diverse patient populations
- Clinically interpretable predictions

### With Better Augmentation
- **2-3x effective dataset size**
- Reduced overfitting
- Better robustness to image quality variations
- Improved performance on edge cases

---

## 🚀 Quick Start

1. **Prepare data**: Create CSV with clinical features
2. **Run setup**: `python scripts/setup_multimodal.py`
3. **Train**: `python src/training/train.py --config configs/multimodal.yaml`
4. **Inference**: 
   - With clinical: `python predict.py --image img.jpg --clinical data.json`
   - Image only: `python predict.py --image img.jpg`

---

## 📚 References
- Gulshan et al. (2016): Development and Validation of Deep Learning Algorithm for DR
- Ting et al. (2017): Deep learning in ophthalmology
- Multimodal medical imaging: Survey and best practices

---

**Next Steps**: Implement Phase 1 - Update dataset and dataloader code
