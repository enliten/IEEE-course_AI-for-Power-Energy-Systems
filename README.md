# IEEE-course---AI-for-Power-Energy-Systems

This repository provides source code examples and Jupyter notebooks used in the online course named "*[AI for Power and Energy Systems: Applications, Challenges, and Opportunities](https://xplorestaging.ieee.org/courses/details/EDP818)*" by [Dr. Fran Li](https://www.eecs.utk.edu/people/fangxing-fran-li/) at the University of Tennessee.
The goal is to help participants run and explore the code during or after the course with minimal setup.

## What in the repository

This repository currently contains the following Jupyter notebooks:

### **`DNN_sin.ipynb`**

  Demonstrates the use of **Deep Neural Networks (DNN)** to approximate a sine function.
  *Course context:* Introduces fundamental concepts of deep learning and regression modeling using TensorFlow/Keras.

### **`DNN_template.ipynb`**

Implements a **feed-forward Deep Neural Network (DNN)** for generic one-dimensional regression tasks such as function fitting or load-curve prediction.It provides multiple **function modes** to generate datasets, including:

- `"sin"` – Sine wave function
- `"cos"` – Cosine wave function
- `"exp"` – Exponential function
- `"log"` – Logarithmic function
- `"poly3"` – Cubic polynomial function
- `"custom"` – User-defined function

Each option can be selected in the *Section 0: Settings* block.
The notebook supports configurable hyperparameters such as `EPOCHS`, `HIDDEN_LAYERS`, and `TRAIN_RATIO`, and provides an end-to-end workflow for model training, testing, and visualization.

### **`dnn_load_curve/` Folder**

This folder contains two key files:

- **`dnn_load_curve.ipynb`***(Available at: [GitHub Link](https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems/blob/main/dnn_load_curve/dnn_load_curve.ipynb))*Demonstrates how to train a DNN on real-world load data to perform short-term load curve prediction.
- **`1min_load_linear.xlsx`**
  *(Available at: [GitHub Link](https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems/blob/main/dnn_load_curve/1min_load_linear.xlsx))*
  This Excel file contains one-minute interval load data used as the dataset for the notebook.

**Note**: Uploading the Excel File in Google Colab: To run the notebook in **Google Colab**, upload the Excel file named **`1min_load_linear.xlsx`** directly into the Colab working directory.


### **`CNN_for_5_bus_system.ipynb`**

  Implements a **Convolutional Neural Network (CNN)** for analyzing a **5-bus power system**.
  *Course context:* Applies deep learning techniques to power system datasets, showing how CNNs can capture grid topology and operational features.

### **`DRL_BESS.ipynb`**

Introduces **Deep Reinforcement Learning (DRL)** through a simple battery energy storage control problem.
The notebook uses a small **Deep Q-Network (DQN)** agent to learn when to charge, discharge, or idle under a synthetic 24-hour load profile and illustrative time-of-use electricity price.
*Course context:* Connects reinforcement learning concepts such as state, action, reward, policy, and Q-values to power-system energy management.

### **`PINN_Frequency.ipynb`**

Introduces **Physics-Informed Neural Networks (PINNs)** through a linearized power-system swing-equation problem.
The notebook compares an ordinary neural network with a PINN that uses sparse synthetic rotor-angle samples plus the physical residuals of `d delta/dt = omega` and `M d omega/dt + D omega + Ks delta = Delta P`.
*Course context:* Shows how governing equations, initial conditions, and sparse data can be combined in neural-network training for electromechanical oscillations.

## How to Use

We provide two options depending on your background and environment:

### 1. Easiest Option – Run on Google Colab (No Installation Required)

- Upload the `.ipynb` notebook file(s) from this repository to [Google Colab](https://colab.research.google.com/).
- Simply open the file in Colab and click **Run All**.
- No local installation or configuration is required. This is the recommended method for first-time users.

### 2. Advanced Option – Run Locally with Micromamba

If you already have Micromamba or Mamba installed:

1. Clone this repository or download the files.
2. Create the course environment:

   ```bash
   micromamba env create -f environment.yml
   micromamba activate ai-course-pes
   python -m ipykernel install --user --name ai-course-pes --display-name "Python (ai-course-pes)"
   ```
3. Launch Jupyter:

   ```bash
   jupyter lab
   ```
4. Open a notebook and select the **Python (ai-course-pes)** kernel.

This environment includes TensorFlow/Keras, PyTorch, pandapower, PuLP/CBC, pandas, openpyxl, matplotlib, and Jupyter tools used by the course notebooks.

## License

[MIT License](LICENSE) – Feel free to use and modify for learning and research purposes.
