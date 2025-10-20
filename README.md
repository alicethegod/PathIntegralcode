# Code for: A Path Integral Field Theory for the Emergence of Meaning in Artificial Intelligence

This repository contains the official source code and numerical experiment scripts for the paper, "A Path Integral Field Theory for the Emergence of Meaning in Artificial Intelligence." Our work proposes a first-principles physical theory, based on statistical field theory and the path integral formalism, to provide a unified explanation for the learning mechanisms of AI.

The scripts provided here allow for the full replication of the key experimental results presented in the paper and its supplemental material, including:
1.  **D-Scaling as a Cognitive Phase Transition** in a Vision Transformer.
2.  **Grounding Duality** validation in a simple MLP.
3.  **Physical Characterization of Learning States** (Underfitting, Good Fit, Overfitting).

## Repository Structure

This repository is organized as follows:

-   `VIT-D.ipynb`: The main Jupyter Notebook for running the **D-Scaling experiment** on a Vision Transformer.
-   `VIT_D_logic.py`: The core logic module for the ViT model, theory analyzer, and training task, imported by `VIT-D.ipynb`.
-   `Path Integral Duality Experiment Script .ipynb`: A Jupyter Notebook to replicate the **Grounding Duality experiment**.
-   `NN Fitting and Cognitive Theory Analysis.ipynb`: A Jupyter Notebook to replicate the experiment on **characterizing learning states**.
-   `optimized_publication_grade_plotting_script.py`: A Python script with a simple GUI to generate publication-quality plots from the CSV data produced by the D-Scaling experiment.
-   `requirements.txt`: A list of all necessary Python packages.

## Setup and Installation

To run these experiments, you will need a Python 3 environment. We recommend using a virtual environment.

**1. Clone the repository:**
```bash
git clone https://github.com/alicethegod/PathIntegralcode
cd PathIntegralcode
```


**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows
# venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install the required packages:**
```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
torch
torchvision
numpy
matplotlib
seaborn
networkx
scipy
tqdm
pandas
scikit-learn
```

## How to Run the Experiments

All main experiments can be run directly from their respective Jupyter Notebooks.

### Experiment 1: D-Scaling as a Cognitive Phase Transition

This is the main experiment demonstrating that neural scaling laws can be interpreted as a critical phenomenon.

**To run:**
1. Open and run the `VIT-D.ipynb` notebook.
2.  The script will train multiple Vision Transformer models on subsets of the Fashion-MNIST dataset of varying sizes.
3.  After training, it will perform the thermodynamic analysis and save a results plot (`.png`) and a data file (`.csv`) to a `results_final` directory.

**To generate a publication-grade plot from the saved CSV:**
Run the plotting script from your terminal:
```bash
python optimized_publication_grade_plotting_script.py
```
A GUI window will pop up, allowing you to select the `.csv` file you wish to plot.

### Experiment 2: Grounding Duality

This experiment validates that the physical quantities (U and S) show consistent dynamics whether calculated from input (forward) or output (backward) grounding constraints.

**To run:**
1.  Open and run the `Path Integral Duality Experiment Script .ipynb` notebook.
2.  The script will train a simple MLP on the `make_moons` dataset and log the evolution of U and S over time.
3.  The final cells will generate a 2x2 plot showing the U-S linear relationship and their dynamics over epochs.

### Experiment 3: Physical States of Learning

This experiment demonstrates that underfitting, good fitting, and overfitting correspond to distinct and predictable physical states in the (U, S) phase space.

**To run:**
1.  Open and run the `NN Fitting and Cognitive Theory Analysis.ipynb` notebook.
2.  The script will train three different MLP models to induce the three fitting states.
3.  The final cell will generate a plot comparing the fitting curves and their corresponding final U and S values.

## Theoretical Background: The `PathIntegralAnalyzer`

The core of the numerical validation is the `PathIntegralAnalyzer` (or `TheoryAnalyzer`) class implemented in the scripts. This class performs the following key functions:
1.  **Graph Transformation**: It transforms a trained PyTorch model's linear layers into a weighted, directed graph where edge weights correspond to path energies.
2.  **Path Integration**: It finds all possible paths from a given hidden node to the "grounding" nodes (either the input or output layer).
3.  **Calculation of Physical Quantities**: For each node, it computes:
    -   **Cognitive Internal Energy (U)**: The average energy over all grounding paths, proxied by the harmonic mean of path costs. It represents the concept's intrinsic complexity.
    -   **Cognitive Entropy (S)**: The Shannon entropy of the path importance distribution. It represents the concept's representational diversity and robustness.
    -   **Cognitive Free Energy (F)**: Derived from the partition function ($Z$, the sum of all path importances). It represents the concept's structural stability and accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
