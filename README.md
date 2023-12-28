# Error Bounds in Variational Quantum Algorithms

This repository provides the code for the numerical simulations of our paper [K. Ito, W. Mizukami, and K. Fujii, "Universal noise-precision relations in variational quantum algorithms," arXiv:2106.03390 (2021)](https://arxiv.org/abs/2106.03390).

The purpose of this repository is to provide the code and instructions for reproducing the results of our paper and to provide a python module to compute rough upper and lower bounds of the minimization error in VQAs, which may be used for an easy check of the achievable precision via a noisy quantum circuit.

## Files

- [`Error_bounds.ipynb`](https://github.com/kosukeitos/VQA_Error_Bounds/blob/main/Error_bounds.ipynb): a Jupyter notebook that provides a code tutorial for how to compute and use our bounds.
- `error_bounds.py`: a python module for computing the upper and lower bounds of the minimization error in VQAs.
- `ALT_circuit.py`: a python module that defines functions to produce quantum circuits used in our simulations

## Usage

To use the code, you can either run the notebook [`Error_bounds.ipynb`](https://github.com/kosukeitos/VQA_Error_Bounds/blob/main/Error_bounds.ipynb) or import the `error_bounds` module into your own code. The notebook provides a step-by-step tutorial for computing and using our bounds, while the `error_bounds` module contains the functions for computing the bounds.

In `error_bounds.py`, we define two functions `rough_LB` and `rough_UB` for computing rough lower bounds and upper bounds, respectively. 
Please read the source file or the notebook for the details of using these functions.
To use these functions, import them from `error_bounds.py`:

```python
from error_bounds import rough_LB, rough_UB
```

## Requirements

Requirements for `Error_bounds.ipynb`:

- Python              3.8.8
- matplotlib          3.3.4
- numpy               1.20.1
- qulacs              0.5.3
- quspin              0.3.6
- scipy               1.6.2
- IPython             7.22.0
- jupyter_client      6.1.12
- jupyter_core        4.7.1
- jupyterlab          3.0.14
- notebook            6.3.0

Requirements for `error_bounds.py`:

- Python              3.8.8
- numpy               1.20.1

Requirements for `ALT_circuit.py`:

- Python              3.8.8
- numpy               1.20.1
- qulacs              0.5.3

## Citation

If you use this code in your research, please cite our paper:

K. Ito, W. Mizukami, and K. Fujii, "Universal noise-precision relations in variational quantum algorithms," Phys. Rev. Research 5, 023025 (2023).
