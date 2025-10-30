# Kaggle Dataset Integration Guide

## 🏥 Recommended Datasets for MI Modeling

### 1. MIT-BIH Arrhythmia Database
- **Kaggle Search**: "MIT-BIH Arrhythmia Database"
- **Size**: ~40MB
- **Content**: 48 half-hour ECG recordings from 47 subjects
- **Features**: Various arrhythmias including MI patterns
- **Format**: CSV, MAT files
- **Best For**: Basic arrhythmia detection and MI pattern recognition

### 2. ECG Heartbeat Categorization Dataset
- **Kaggle Search**: "ECG Heartbeat Categorization"
- **Size**: ~100MB
- **Content**: 109,446 ECG heartbeat samples
- **Categories**: 
  - Normal (N)
  - Supraventricular (S)
  - Ventricular (V)
  - Fusion (F)
  - Unknown (Q)
- **Format**: CSV with preprocessed features
- **Best For**: Classification model training and validation

### 3. PTB-XL ECG Database
- **Kaggle Search**: "PTB-XL ECG Database"
- **Size**: ~5GB
- **Content**: 21,837 clinical 12-lead ECG recordings
- **Features**: Comprehensive cardiac conditions including MI
- **Format**: CSV, executables
- **Best For**: Comprehensive cardiac modeling and validation

### 4. Heart Failure Prediction Dataset
- **Kaggle Search**: "Heart Failure Prediction"
- **Size**: ~10KB
- **Content**: Clinical features + outcomes
- **Use**: For validation of your model's predictions

## 📥 How to Download and Use

### Step 1: Install Kaggle API
```bash
pip install kaggle
```

### Step 2: Setup API Credentials
1. Go to Kaggle Account → API → Create New API Token
2. Download `kaggle.json`
3. Place in `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Step 3: Download Dataset
```bash
# Example: Download MIT-BIH dataset
kaggle datasets download -d shayanfazeli/heartbeat

# Example: Download PTB-XL dataset
kaggle datasets download -d bashayer/ecg-dataset

# Extract the dataset
unzip heartbeat.zip -d ./datasets/
```

### Step 4: Convert to Your Format
```python
import pandas as pd
import numpy as np

# Load Kaggle dataset
df = pd.read_csv('datasets/mitbih_train.csv')

# Convert to your system's expected format
# Your system expects: time, lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF, lead_V1, lead_V2, lead_V3, lead_V4, lead_V5, lead_V6
def convert_kaggle_to_system_format(kaggle_data, sample_rate=360):
    """
    Convert Kaggle ECG data to your system's format
    """
    # Create time vector
    time = np.arange(len(kaggle_data)) / sample_rate
    
    # If single lead, replicate to 12 leads
    if kaggle_data.shape[1] == 1:
        # Replicate single lead to all 12 leads with slight variations
        leads = np.zeros((len(kaggle_data), 12))
        for i in range(12):
            leads[:, i] = kaggle_data.iloc[:, 0] * (0.8 + 0.4 * np.random.random())
    else:
        # If multiple leads, pad or truncate to 12
        leads = np.zeros((len(kaggle_data), 12))
        min_leads = min(12, kaggle_data.shape[1])
        leads[:, :min_leads] = kaggle_data.iloc[:, :min_leads]
    
    # Create DataFrame in your system's format
    system_df = pd.DataFrame({
        'time': time,
        'lead_I': leads[:, 0],
        'lead_II': leads[:, 1],
        'lead_III': leads[:, 2],
        'lead_aVR': leads[:, 3],
        'lead_aVL': leads[:, 4],
        'lead_aVF': leads[:, 5],
        'lead_V1': leads[:, 6],
        'lead_V2': leads[:, 7],
        'lead_V3': leads[:, 8],
        'lead_V4': leads[:, 9],
        'lead_V5': leads[:, 10],
        'lead_V6': leads[:, 11]
    })
    
    return system_df

# Convert and save
converted_data = convert_kaggle_to_system_format(df)
converted_data.to_csv('datasets/converted_kaggle_data.csv', index=False)
```

## 🧪 Testing Your System with Kaggle Data

### 1. Upload to WebAssembly Interface
- Use the converted CSV files in your web interface
- Test with different categories (Normal, MI, Arrhythmia)

### 2. Validate Against Known Labels
```python
# Compare your model's predictions with Kaggle labels
def validate_predictions(your_predictions, kaggle_labels):
    """
    Validate your model's predictions against Kaggle ground truth
    """
    from sklearn.metrics import accuracy_score, classification_report
    
    accuracy = accuracy_score(kaggle_labels, your_predictions)
    report = classification_report(kaggle_labels, your_predictions)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(report)
    
    return accuracy, report
```

### 3. Performance Benchmarking
- Compare your model's performance against published results
- Use cross-validation with Kaggle datasets
- Test edge cases and rare conditions

## 📊 Expected Results

### MIT-BIH Dataset:
- **Normal**: Should show stable membrane potential
- **MI**: Should show elevated ST segments, T-wave inversions
- **Arrhythmia**: Should show irregular patterns

### ECG Heartbeat Categorization:
- **Normal (N)**: 90%+ accuracy expected
- **Ventricular (V)**: 85%+ accuracy expected
- **Supraventricular (S)**: 80%+ accuracy expected

## 🔍 Quality Assurance

### Data Validation Checklist:
- [ ] Sample rate consistency (360 Hz for most datasets)
- [ ] Lead count (12-lead vs single-lead)
- [ ] Time duration (30 seconds to 30 minutes)
- [ ] Signal quality (noise levels, artifacts)
- [ ] Label accuracy (clinical annotations)

### Preprocessing Steps:
1. **Filtering**: Remove baseline drift, high-frequency noise
2. **Normalization**: Scale to ±1mV range
3. **Segmentation**: Extract heartbeat cycles
4. **Feature Extraction**: R-peak detection, ST-segment analysis

## 🚀 Advanced Usage

### Custom Dataset Creation:
```python
def create_mi_simulation_dataset(n_samples=1000, duration=30):
    """
    Create synthetic MI dataset for testing
    """
    from your_mi_model import MIModel
    
    model = MIModel()
    dataset = []
    
    for i in range(n_samples):
        # Simulate different MI conditions
        mi_type = np.random.choice(['anterior', 'inferior', 'lateral', 'normal'])
        severity = np.random.uniform(0.1, 0.9)
        
        # Generate ECG data
        ecg_data = model.simulate_mi(mi_type, severity, duration)
        dataset.append({
            'data': ecg_data,
            'label': mi_type,
            'severity': severity
        })
    
    return dataset
```

## 📈 Performance Metrics

### Expected Performance on Kaggle Datasets:
- **MIT-BIH**: 95%+ accuracy for basic arrhythmia detection
- **PTB-XL**: 90%+ accuracy for MI detection
- **ECG Heartbeat**: 85%+ accuracy for multi-class classification

### Benchmarking Against Literature:
- Compare with published results from IEEE/ACM papers
- Use standard evaluation metrics (sensitivity, specificity, F1-score)
- Report performance on held-out test sets

## 🔗 Useful Resources

### Kaggle Notebooks:
- Search for "ECG analysis" notebooks
- Look for "cardiac arrhythmia detection" examples
- Find "MI prediction" implementations

### Academic References:
- MIT-BIH Database: Moody & Mark, 2001
- PTB-XL Database: Wagner et al., 2020
- ECG Classification: Rajpurkar et al., 2017

### Tools and Libraries:
- **wfdb**: For reading MIT-BIH format files
- **scipy**: For signal processing
- **scikit-learn**: For machine learning evaluation
- **matplotlib**: For visualization

## ⚠️ Important Notes

1. **Data Privacy**: Ensure compliance with data usage agreements
2. **Clinical Validation**: Always validate with clinical experts
3. **Reproducibility**: Document all preprocessing steps
4. **Version Control**: Keep track of dataset versions and modifications
5. **Quality Control**: Regularly check for data drift and quality issues




