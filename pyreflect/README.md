# **NR-SCFT-ML: Neutron Reflectivity SCFT Machine Learning**
A Python package for Neutron Reflectivity data analysis using Machine Learning.
Including **Chi parameters prediction** from **SLD profile data** and **SLD profile prediction from NR curves**.

## **Overview**
**NR-SCFT-ML** is a PyPI package for efficient processing and analysis of Neutron Reflectivity (NR) data. It provides a streamlined pipeline for:

- Preprocessing NR datasets  
- Predicting SLD Profile by training a CNN
- Predicting **Chi** parameters by training a combined Autoencoder and MLP model.

## **Prerequest**

Conda  
Python >=3.10

## **Quick Start Guide**  

Prepare the Package Environment on HPC or Your Local Machine (Large Memory Required). The `setup.sh` creates a Jupyter kernel environment named `PyreflectEnvironment`.

run in cell:
```code
!curl -fsSLo setup.sh https://raw.githubusercontent.com/williamQyq/pyreflect/main/setup.sh
!bash setup.sh
```

To install from TestPyPI, run:
```bash
%pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyreflect==1.3.1
```

### 🧑🏻‍💻Example Notebooks ### 
To learn how to use it, check the example notebooks in `pyreflect/example_notebooks` or watch the tutorial video. Click below:

<a href="https://youtu.be/cc8xeLhOXDo">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" width="50" height="50" />
</a>

### **1️⃣ Initialize Configuration**
To initialize the default configuration for model path, hyperparameters, and data reading paths, run:

```bash
python -m pyreflect init --force
```
Use the --force flag if the configuration has already been created and you want to overwrite it.

### **2️⃣ Run Training & Prediction**
Open `settings.yml` and update the file paths for your SLD profile data and Chi parameters

### **2️⃣.1️⃣ Training datasets and Train model 
Generate training datasets using the example in [example_notebook_autoencoder.ipynb](examples/example_notebook_autoencoder.ipynb).  
The memory usage should be properly managed by controlling the number of curves generated for training, as it will consume a large amount of memory.  

Train the model using the example in [example_notebook_SLD_prediction.ipynb](examples/example_notebook_SLD_prediction.ipynb).

### **2️⃣.2️⃣ A simple Experimental Datasets - from real world lab 
[datasets](datasets) contains processed experimental NR, its manual fit SLD profile data, and AE denoised experimental NR data. 

### **3️⃣ Run interaction chi parameters prediction**
```bash
python -m pyreflect run --enable-chi-prediction
```

### **4️⃣ Run sld profile prediction from nr curves(or import package lib in notebook)**
```bash
python -m pyreflect run --enable-sld-prediction
```

## Credits

This project builds on work by:
- Brian Qu ([NR-SLD-CNN](https://github.com/BBQ591/NR-SLD-CNN))
- Dr. Rajeev Kumar
- Prof. Miguel Fuentes-Cabrera [NR-SCFT-ML](https://github.com/miguel-fc/NR-SCFT-ML)

## Author

- Yuqing Qiao (William) – Maintainer and developer of this PyPI package
