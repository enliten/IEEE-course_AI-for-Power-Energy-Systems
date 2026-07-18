# AI for Power and Energy Systems: Hands-on Examples

This repository hosts the hands-on Jupyter notebooks used in an IEEE course and an IEEE PES webinar series on applying AI to power and energy systems; every module runs locally or on Google Colab with minimal setup.

**IEEE course.** Modules 01-03 accompany the online course "*[AI for Power and Energy Systems: Applications, Challenges, and Opportunities](https://xplorestaging.ieee.org/courses/details/EDP818)*" by [Dr. Fran Li](https://www.eecs.utk.edu/people/fangxing-fran-li/) at the University of Tennessee, which introduces the underlying AI concepts.

**IEEE PES webinar.** Modules 01-03 are also presented in Part I of the IEEE PES webinar "*AI for Power System Applications: Hands-on Examples, Selected Applications, and Perspectives*" ([video](https://resourcecenter.ieee-pes.org/education/webinars/pes_ed_web_aip1_061026), DOI [10.17023/1r81-nn94](https://dx.doi.org/10.17023/1r81-nn94); [slides](https://resourcecenter.ieee-pes.org/education/webinars/pes_ed_web_aip1_061026_sld), DOI [10.17023/p8cc-re33](https://dx.doi.org/10.17023/p8cc-re33)); modules 04-06 will follow in Part II.

## Getting started

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

## Module overview

Each module pairs a core AI concept with a representative power-system task. Click a module name to jump to its details and file links below.

<table>
<thead>
<tr>
<th>Tier</th>
<th>Module</th>
<th>Core AI concept</th>
<th>Power-system task</th>
<th>System / data</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2" valign="middle"><strong>Tier 1</strong><br>(Foundational)</td>
<td><a href="#01-dnn-function-fitting">01 DNN function fitting</a></td>
<td>Nonlinear regression, MLP training</td>
<td>Curve / function approximation</td>
<td>Analytic y = f(x)</td>
</tr>
<tr>
<td><a href="#02-dnn-load-curve-fitting">02 DNN load curve fitting</a></td>
<td>Data-driven regression, train/test split</td>
<td>Short-term load-curve prediction</td>
<td>1-min load data</td>
</tr>
<tr>
<td valign="middle"><strong>Tier 2</strong><br>(Domain-coupled)</td>
<td><a href="#03-cnn-for-5-bus-system">03 CNN for 5 bus system</a></td>
<td>1-D convolution, multi-output regression</td>
<td>Power-flow surrogate (voltages, line flows)</td>
<td>PJM 5-bus, Monte-Carlo PF</td>
</tr>
<tr>
<td rowspan="3" valign="middle"><strong>Tier 3</strong><br>(Frontier)</td>
<td><a href="#04-dnn-assisted-optimization">04 DNN Assisted Optimization</a></td>
<td>Learned constraint with exact MILP encoding</td>
<td>Security-constrained decision making</td>
<td>Synthetic feasibility set</td>
</tr>
<tr>
<td><a href="#05-drl-bess">05 DRL BESS</a></td>
<td>Deep Q-network (DQN)</td>
<td>Battery charge/discharge scheduling</td>
<td>24-h load and ToU price</td>
</tr>
<tr>
<td><a href="#06-pinn-frequency">06 PINN Frequency</a></td>
<td>Physics-informed training</td>
<td>Rotor-angle / frequency dynamics</td>
<td>Linearized swing equation</td>
</tr>
</tbody>
</table>

## Module details

Each folder and file below links directly into the repository.

### 01 DNN function fitting

Folder: [`01_DNN_function_fitting/`](01_DNN_function_fitting)

**[`DNN_sin.ipynb`](01_DNN_function_fitting/DNN_sin.ipynb)** — A **Deep Neural Network (DNN)** approximating a sine function; a first look at deep-learning regression with TensorFlow/Keras.

**[`DNN_template.ipynb`](01_DNN_function_fitting/DNN_template.ipynb)** — A reusable **feed-forward DNN** for 1-D regression. Selectable function modes generate the dataset:

- `"sin"` – Sine wave function
- `"cos"` – Cosine wave function
- `"exp"` – Exponential function
- `"log"` – Logarithmic function
- `"poly3"` – Cubic polynomial function
- `"custom"` – User-defined function

Set the mode and hyperparameters (`EPOCHS`, `HIDDEN_LAYERS`, `TRAIN_RATIO`) in the initial settings block.

### 02 DNN load curve fitting

Folder: [`02_DNN_load_curve_fitting/`](02_DNN_load_curve_fitting)

**[`DNN_load_curve.ipynb`](02_DNN_load_curve_fitting/DNN_load_curve.ipynb)** — Demonstrates how to train a DNN on real-world load data to perform short-term load-curve prediction.

**[`1min_load_linear.xlsx`](02_DNN_load_curve_fitting/1min_load_linear.xlsx)** — Input data: a one-minute-interval load profile.

**Note (Google Colab):** upload `1min_load_linear.xlsx` directly into the Colab working directory before running the notebook.


### 03 CNN for 5 bus system

Folder: [`03_CNN_for_5_bus_system/`](03_CNN_for_5_bus_system)

**[`CNN_for_5_bus_system.ipynb`](03_CNN_for_5_bus_system/CNN_for_5_bus_system.ipynb)** — A **Convolutional Neural Network (CNN)** on a **5-bus power system**, showing how CNNs capture grid topology and operational features.

### 04 DNN Assisted Optimization

Folder: [`04_DNN_Assisted_Optimization/`](04_DNN_Assisted_Optimization)

**[`DNN_Assisted_Opti.ipynb`](04_DNN_Assisted_Optimization/DNN_Assisted_Opti.ipynb)** — Embeds a trained ReLU MLP into an optimization model: the network approximates a nonlinear constraint and is reformulated exactly as **mixed-integer linear programming (MILP)** constraints, bridging machine learning and optimization.

### 05 DRL BESS

Folder: [`05_DRL_BESS/`](05_DRL_BESS)

**[`DRL_BESS.ipynb`](05_DRL_BESS/DRL_BESS.ipynb)** — A small **Deep Q-Network (DQN)** agent that learns to charge, discharge, or idle a battery under a synthetic 24-hour load and time-of-use price; an introduction to **Deep Reinforcement Learning (DRL)** for energy management.

### 06 PINN Frequency

Folder: [`06_PINN_Frequency/`](06_PINN_Frequency)

**[`PINN_Frequency.ipynb`](06_PINN_Frequency/PINN_Frequency.ipynb)** — A **Physics-Informed Neural Network (PINN)** for the linearized swing equation, compared against an ordinary NN. It combines sparse rotor-angle samples with the physical residuals $d\delta/dt = \omega$ and $M\,d\omega/dt + D\omega + K_s\delta = \Delta P$.

## License

[MIT License](LICENSE) – Feel free to use and modify for learning and research purposes.
