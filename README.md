# 📊 DataForge — Data Preprocessing Studio

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python"/>
  <img src="https://img.shields.io/badge/GUI-Tkinter-green"/>
  <img src="https://img.shields.io/badge/Data-Pandas-yellow?logo=pandas"/>
  <img src="https://img.shields.io/badge/Math-NumPy-blue?logo=numpy"/>
  <img src="https://img.shields.io/badge/ML-Scikit--learn-orange?logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Charts-Matplotlib-red"/>
  <img src="https://img.shields.io/badge/Status-Active-success"/>
  <img src="https://img.shields.io/badge/License-Educational-lightgrey"/>
</p>

---

## 🚀 Overview

**DataForge** is a modern desktop-based application designed to simplify **data preprocessing, transformation, analysis, and visualization**.

Built with Python, it transforms complex data workflows into an **interactive GUI pipeline**, making it ideal for students, beginners, and developers working with datasets.

---

## 🖥️ UI Preview

### 🔹 Main Dashboard
![Dashboard](screenshots/dashboard.png)

### 🔹 Data Cleaning
![Cleaning](screenshots/cleaning.png)

### 🔹 Normalization
![Normalization](screenshots/normalization.png)

### 🔹 Visualization
![Visualization](screenshots/visualization.png)

### 🔹 Data Reduction (PCA / LDA)
![Reduction](screenshots/reduction.png)

---

## ✨ Features

### 🧹 Data Cleaning Pipeline
- Detects and replaces null-like values (`N/A`, `null`, `-`, etc.)
- Removes empty rows and duplicate entries
- Converts numeric-like strings into numeric format
- Handles missing values:
  - Median for numerical columns
  - Mode / `"Unknown"` for categorical columns
- Removes outliers using **IQR (Interquartile Range)**

---

### 📐 Data Transformation
- Min-Max Normalization
- Scales numeric data to `[0,1]`
- Handles constant columns safely

---

### 📉 Data Reduction
- **PCA (Principal Component Analysis)**
- **LDA (Linear Discriminant Analysis)**

---

### 📈 Data Visualization
- Statistical summary:
  - Mean, Median, Mode, Std Dev
- Charts:
  - Bar graph
  - Box plot
  - Histogram
- Embedded visualization inside GUI

---

### 📂 File Handling
- Upload CSV datasets
- Preview original, cleaned, and normalized data
- Export cleaned dataset

---

## 🔄 Workflow

```text
Upload Data → Clean Data → Normalize → Reduce → Visualize → Export

