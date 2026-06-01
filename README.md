# COVID-19-Classifaction
"A Medical Image Classification system using Deep Learning (CNN) to detect COVID-19, Normal, and Viral Pneumonia from X-Ray scans, featuring a live Streamlit web application."
# 🦠 AI-Powered COVID-19 & Lung Infection Detection System

An end-to-end Deep Learning project designed to classify Chest X-Ray images into three medical categories: **COVID-19**, **Normal**, and **Viral Pneumonia**. The repository includes a comparative CNN analysis notebook and a fully interactive **Streamlit** web application for real-time patient diagnosis.

---

## 🚀 Project Overview & Objectives
Medical imaging datasets are often limited in size, posing a high risk of model overfitting. This project implements and benchmarks two distinct Custom Convolutional Neural Network (CNN) architectures to evaluate performance under limited resource constraints:
1. **Basic CNN Model**: Utilizing standard Conv2D and MaxPooling layers without regularization.
2. **Regularized CNN Model**: Integrating `BatchNormalization` and `Dropout` layers to stabilize gradients and prevent overfitting.

---

## 📊 Dataset Detail
* **Source**: [COVID-19 Image Dataset (Kaggle)](https://kaggle.com)
* **Image Input Resolution**: `224 x 224` pixels (RGB)
* **Batch Size**: `16`
* **Target Classes**: `COVID-19`, `Normal`, `Viral Pneumonia`

---

## 🔬 Model Architectures & Comparison
Both networks were trained under identical hyperparameters (Adam Optimizer, Learning Rate: 0.001, Epochs: 15) using a `DataFrameIterator` pipeline:

### 🏛️ Model 1: Basic CNN
* Pure feature extraction via Conv2D blocks.
* Rapidly prone to memorization (overfitting) on smaller medical datasets.

### 🛡️ Model 2: Advanced Regularized CNN
* Added `BatchNormalization` after convolutions to accelerate early convergence.
* Embedded `Dropout` (up to 0.5) in deep dense layers to enforce robust feature generalization.
* **Verdict for Resource-Constrained Environments**: Model 2 is heavily preferred. It requires negligible parameter overhead while ensuring stable, realistic validation accuracy on unseen test profiles.

---

## 🖥️ Interactive Web Application (Streamlit)
To bridge the gap between AI code and clinical utility, this project includes a lightweight frontend app built with **Streamlit**. 
* **Live Inference**: Allows medical staff to drag and drop any Chest X-Ray scan.
* **Dynamic Probability Plots**: Displays interactive bar charts highlighting probability distribution scores across all labels.
* **Statistical Insights**: Integrates a sidebar report derived from the validation Confusion Matrix.

---

## 🛠️ Installation & Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd YOUR_REPOSITORY_NAME
   ```

2. **Install requirements:**
   ```bash
   pip install streamlit tensorflow pandas numpy pillow matplotlib seaborn scikit-learn
   ```

3. **Train & Save the Model:**
   Run the python/notebook code to train the regularized model. Ensure it saves the final weights as `covid_model.keras`.

4. **Launch the Web App:**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Key Technical Implementation Details
* Handled the Keras `DataFrameIterator` class-mapping using dynamic index binding (`len(train_generator.class_indices)`).
* Automated evaluation matrices mapping loss and accuracy coefficients across all training pipelines.
