# 🎯 FashionMNIST Classifier

<div align="center">
  
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

**🔥 High Performance Neural Network | 🎯 90%+ Validation Accuracy | 📊 89.8% Test Accuracy**

</div>

---

## 👨‍💻 Project Overview

This project implements a **state-of-the-art multi-class image classifier** for the FashionMNIST dataset using artificial neural networks (ANNs) in PyTorch. The model achieves impressive performance with **90%+ validation accuracy** and **89.8% test accuracy**.

**Author:** Luyanda  
**Email:** MBLLUY007@myuct.ac.za

---

## 🚀 Key Features

- ✅ **High-Performance ANN** with variable depth and dropout regularization
- ✅ **Advanced Training Pipeline** with early stopping and Adam optimization
- ✅ **Interactive Classification** via command-line interface
- ✅ **Robust Data Preprocessing** with normalization and augmentation
- ✅ **Professional Code Structure** with modular design
- ✅ **Cross-Platform Compatibility** with automated setup

---

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| **Validation Accuracy** | **90%+** |
| **Test Accuracy** | **89.8%** |
| **Loss Function** | CrossEntropy |
| **Optimizer** | Adam |

---

## 📁 Project Structure

```
FashionMNIST-Classifier/
├── 📄 classifier.py        # Main neural network implementation
├── 📄 requirements.txt     # Python dependencies
├── 📄 Makefile            # Automated setup and execution
└── 📄 README.md           # Project documentation
```

### 🔧 File Descriptions

#### `classifier.py`
The core implementation featuring:
- **Data Loading & Preprocessing** - Efficient FashionMNIST dataset handling
- **Neural Network Architecture** - Configurable ANN with dropout layers
- **Training Loop** - Advanced training with early stopping
- **Evaluation Pipeline** - Comprehensive performance assessment
- **Interactive Interface** - Real-time image classification

#### `requirements.txt`
Essential Python packages:
- `torch` - Deep learning framework
- `torchvision` - Computer vision utilities
- `numpy` - Numerical computing
- `matplotlib` - Data visualization

#### `Makefile`
Convenient automation commands:
- `make install` - One-click dependency installation
- `make run` - Quick project execution

---

## 🛠️ Quick Start

### Option 1: Using Makefile (Recommended)
```bash
make install
make run
```

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
python classifier.py
```

---

## 🎮 Interactive Usage

Once running, the classifier provides an intuitive interface:

```bash
🎯 FashionMNIST Classifier Ready!
📊 Test Accuracy: 89.8%

Please enter a filepath: ./images/dress.jpg
🔍 Analyzing image...
✅ Classifier: Dress (Confidence: 94.2%)

Please enter a filepath: exit
👋 Thank you for using FashionMNIST Classifier!
```

---

## 🎯 Supported Fashion Categories

The classifier can identify **10 different fashion items**:

| Category | Examples |
|----------|----------|
| 👕 T-shirt/top | Casual wear, blouses |
| 👖 Trouser | Jeans, pants, leggings |
| 👗 Dress | Formal, casual dresses |
| 🧥 Coat | Jackets, outerwear |
| 👡 Sandal | Open-toe footwear |
| 👔 Shirt | Formal, button-up shirts |
| 👟 Sneaker | Athletic shoes |
| 👜 Bag | Handbags, purses |
| 🥾 Ankle boot | Short boots |
| 🩱 Pullover | Sweaters, hoodies |

---

## 💡 Technical Highlights

- **🧠 Neural Architecture**: Multi-layer perceptron with dropout regularization
- **📈 Optimization**: Adam optimizer with learning rate scheduling
- **🎯 Early Stopping**: Prevents overfitting and saves training time
- **🔄 Data Augmentation**: Enhanced training with transformations
- **📊 Evaluation Metrics**: Comprehensive accuracy and loss tracking

---

## 📋 Requirements

- **Python 3.7+**
- **CUDA-capable GPU** (optional, for faster training)
- **28x28 grayscale images** for custom classification

---

## 🌟 Why This Project Stands Out

- **Production-Ready Code**: Clean, modular, and well-documented
- **High Accuracy**: Exceeds 89% test accuracy benchmark
- **User-Friendly**: Interactive command-line interface
- **Scalable Architecture**: Easy to extend and modify
- **Professional Setup**: Automated installation and execution

---

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

---

## 📧 Contact

**Luyanda**  
📧 MBLLUY007@myuct.ac.za

---

<div align="center">
  
**⭐ Star this repo if you found it helpful!**

</div>