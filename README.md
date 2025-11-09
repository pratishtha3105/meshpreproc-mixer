# 🧠 3D Mesh Preprocessing and Quantization Pipeline

## 📘 Overview
This project implements a complete **3D mesh preprocessing system** using Python.  
It performs normalization, quantization, reconstruction, and error evaluation on 3D object meshes (`.obj` files).  
The pipeline is modular and automatically processes all meshes in the `data/` directory.

---

## 🧱 Project Structure

mesh-preproc/
├── data/ # Input .obj mesh files
├── outputs/ # Processed results, plots, and summary CSV
├── src/ # Source code
│ ├── load.py # Mesh loading and vertex extraction
│ ├── normalize.py # Min-Max and Unit-Sphere normalization
│ ├── quantize.py # Quantization and dequantization
│ ├── reconstruct.py # Reconstruction from quantized data
│ ├── metrics.py # MSE/MAE computation and visualization
│ └── run_all.py # Full automation for all meshes
└── venv/ # Virtual environment

---

## ⚙️ Setup Instructions

### 1️⃣ Clone or copy the project
```bash
cd path/to/your/folder
---
### 2️⃣ Create a virtual environment
```bash
python -m venv venv
---
### 3️⃣ Activate the virtual environment
```bash
venv\Scripts\activate
---
### 4️⃣ Install dependencies
```bash
pip install numpy matplotlib tqdm trimesh pandas
---
### ▶️ Running the Project
To execute the full preprocessing pipeline for all .obj files:
```bash
python src/run_all.py

This will:

Load all meshes from data/

Apply normalization (Min–Max & Unit-Sphere)

Quantize and reconstruct

Compute MSE/MAE errors

Generate bar plots and a summary CSV in outputs/
----

📊 Outputs
| Folder/File           | Description                                          |
| --------------------- | ---------------------------------------------------- |
| `outputs/*.npy`       | Normalized, quantized, and reconstructed vertex data |
| `outputs/plots/`      | MSE per-axis error bar charts                        |
| `outputs/summary.csv` | Per-mesh MSE and MAE results for both methods        |
---
🧠 Key Concepts Used

Normalization: Scales mesh vertices using Min–Max and Unit Sphere methods

Quantization: Converts continuous coordinates into discrete bins for compression

Reconstruction: Restores original coordinates from quantized data

Error Metrics: Measures loss using Mean Squared Error (MSE) and Mean Absolute Error (MAE)

Visualization: Error plots per axis using Matplotlib

---
