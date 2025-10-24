# IEEE-course---AI-for-Power-Energy-Systems
This repository provides source code examples and Jupyter notebooks used in the online course named "*AI for Power &amp; Energy Systems: Computation Acceleration in Modeling and Control*" by [Dr. Fran Li](https://www.eecs.utk.edu/people/fangxing-fran-li/) at the University of Tennessee.  
The goal is to help participants run and explore the code during or after the course with minimal setup.  
## What in the repository

This repository currently contains two Jupyter notebooks:

  ### **`DNN_sin.ipynb`**  
  Demonstrates the use of **Deep Neural Networks (DNN)** to approximate a sine function.  
  *Course context:* Introduces fundamental concepts of deep learning and regression modeling using TensorFlow/Keras.

---

  ### **`CNN_for_5_bus_system.ipynb`**  
  Implements a **Convolutional Neural Network (CNN)** for analyzing a **5-bus power system**.  
  *Course context:* Applies deep learning techniques to power system datasets, showing how CNNs can capture grid topology and operational features.

---

  ### **`DNN_template.ipynb`**
Implements a **feed-forward Deep Neural Network (DNN)** for generic one-dimensional regression tasks such as function fitting or load-curve prediction.  
It provides multiple **function modes** to generate datasets, including:

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
- **`dnn_load_curve.ipynb`**  
  *(Available at: [GitHub Link](https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems/blob/main/dnn_load_curve/dnn_load_curve.ipynb))*  
  Demonstrates how to train a DNN on real-world load data to perform short-term load curve prediction.

- **`1min_load_linear.xlsx`**  
  *(Available at: [GitHub Link](https://github.com/enliten/IEEE-course_AI-for-Power-Energy-Systems/blob/main/dnn_load_curve/1min_load_linear.xlsx))*  
  This Excel file contains one-minute interval load data used as the dataset for the notebook.


**Note**: Uploading the Excel File in Google Colab: To run the notebook in **Google Colab**, upload the Excel file named **`1min_load_linear.xlsx`** directly into the Colab working directory.  
  <img width="500" alt="2c3421b3e75c8a976866475d1e92f029" src="https://github.com/user-attachments/assets/1607271c-d89d-4b0d-99c9-381073c021b1" />


## How to Use

We provide two options depending on your background and environment:

### 1. Easiest Option – Run on Google Colab (No Installation Required)
- Upload the `.ipynb` notebook file(s) from this repository to [Google Colab](https://colab.research.google.com/).
- Simply open the file in Colab and click **Run All**.  
- No local installation or configuration is required. This is the recommended method for first-time users.  

### 2. Advanced Option – Run Locally with Python
If you already have Python and Jupyter installed on your computer:
1. Clone this repository or download the `.ipynb` files.  
2. Make sure your Python environment includes the required packages:  

   ```
   numpy
   pandas
   matplotlib
   tensorflow
   pandapower
   jupyterlab
   ```  
3. Launch Jupyter Notebook or Jupyter Lab.  
4. Open the `.ipynb` file and run the cells.  

This option is more flexible for users with prior Python experience.  

## License
[MIT License](LICENSE) – Feel free to use and modify for learning and research purposes.
