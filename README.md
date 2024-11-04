# 🛡️ Phishing URL Detection Using BERT

## 🎯 About
This project implements a machine learning model for detecting phishing URLs using a fine-tuned BERT model. The system analyzes URLs to determine whether they are legitimate or potentially malicious phishing attempts. Built on the `bert-base-uncased` architecture from Google, this model achieves high accuracy in distinguishing between safe and unsafe URLs.

## ⭐ Key Features
- 🎯 Binary classification of URLs (Safe/Not Safe)
- 🤖 Fine-tuned BERT model with frozen base layers
- 📈 High performance metrics (ROC-AUC and Accuracy)
- 🔌 Easy-to-use interface for URL classification
- ⚡ Efficient preprocessing pipeline
- 💪 Built-in confidence scoring

## 🔧 Technical Details

### 🏗️ Model Architecture
- 🔮 Base Model: `google-bert/bert-base-uncased`
- 🎯 Classification Head: Binary classification (Safe/Not Safe)
- 🛠️ Optimization: Base model layers frozen except for pooling layers
- 📝 Training Strategy: Fine-tuning approach with selective layer freezing

### 📊 Dataset
🗂️ The model is trained on the "shawhin/phishing-site-classification" dataset from Hugging Face, which contains labeled examples of both legitimate and phishing URLs.

### 📈 Training Metrics
#### 🏆 Final Training Outputs
- 🔄 Total Training Steps: 2,630
- 📉 Final Training Loss: 0.3581
- ⏱️ Training Runtime: 1018.05 seconds
- 🚀 Training Throughput: 20.63 samples/second
- ⚡ Training Steps/Second: 2.58
- 💻 Total FLOPS: 7.07e14

#### 📊 Epoch-wise Performance
| Epoch | Training Loss | Validation Loss | Accuracy | AUC     |
|-------|---------------|-----------------|----------|---------|
| 1     | 0.4940       | 0.3772          | 0.8160   | 0.9130  |
| 2     | 0.4058       | 0.3677          | 0.8400   | 0.9340  |
| 3     | 0.3687       | 0.3245          | 0.8560   | 0.9350  |
| 4     | 0.3460       | 0.4032          | 0.8380   | 0.9440  |
| 5     | 0.3501       | 0.3123          | 0.8710   | 0.9450  |
| 6     | 0.3522       | 0.2904          | 0.8620   | 0.9500  |
| 7     | 0.3217       | 0.3069          | 0.8620   | 0.9470  |
| 8     | 0.3109       | 0.2940          | 0.8640   | 0.9490  |
| 9     | 0.3214       | 0.2852          | 0.8730   | 0.9500  |
| 10    | 0.3097       | 0.2974          | 0.8710   | 0.9510  |

### 🎯 Performance Highlights
- 🏆 Best Accuracy: 0.8730 (Epoch 9)
- 🌟 Best AUC Score: 0.9510 (Epoch 10)
- 📉 Lowest Validation Loss: 0.2852 (Epoch 9)
- 🎯 Final Model Metrics:
  - ✅ Accuracy: 0.8710
  - 📈 AUC: 0.9510
  - 📉 Validation Loss: 0.2974

### ⚙️ Training Configuration
- 📊 Learning Rate: 2e-4
- 📦 Batch Size: 8
- 🔄 Training Epochs: 10
- 📋 Evaluation Strategy: Per epoch
- ✨ Best Model Selection: Enabled

## 🚀 Usage

### 📥 Installation Requirements
```bash
pip install transformers datasets evaluate numpy torch accelerate
```

### 💻 Basic Usage
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("bert-phishing-classifier")
tokenizer = AutoTokenizer.from_pretrained("bert-phishing-classifier")

# Classify a URL
url = "http://example.com"
label, confidence = classify_url(url)
print(f"Classification: {label}, Confidence: {confidence:.2f}%")
```

### 🔄 Model Training Process
1. 📥 Dataset loading and preprocessing
2. 🎯 Model initialization with frozen layers
3. ⚙️ Training configuration setup
4. 🏃 Model training with evaluation
5. 💾 Model saving for future use

## 📝 Implementation Notes
- 🔍 Custom preprocessing function for URL tokenization
- 🛑 Early stopping and best model saving
- 📦 Efficient batch processing with data collation
- 📊 Comprehensive evaluation metrics

## 🚀 Future Improvements
- 🌍 Multi-language support
- 🔍 More sophisticated URL preprocessing techniques
- 🎯 Regular expression-based features
- 📈 Training dataset expansion
- 🤝 Ensemble methods implementation

## 📜 License
[Specify your license here]

## 👥 Contributors
[Add contributor information here]

## 📚 Citation
If you use this model in your research, please cite:
```
[Add citation information here]
```