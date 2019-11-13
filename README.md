# HyperKG: Hyperbolic Knowledge Graph Embeddings for Knowledge Base Completion
This repository contains our implementation of the [HyperKG: Hyperbolic Knowledge Graph Embeddings for Knowledge Base Completion](https://arxiv.org/abs/1908.04895).

# License #

This code is partially based on code from the following repositories:
* [OpenKE-PyTorch (old)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old))
* [poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)

Every source code file written exclusively by the author of this repo is licensed under Apache License Version 2.0. For more information, please refer to the [license](LICENSE).

# Instructions for running:
* Prerequisites : 
    * Python, C, C++.
    * Python Libraries: NumPy, SciPy, pytorch, pickle.
    
* Run the code:
    1. Compile the C, C++ code using: `sh make.sh `
    2. To analyze **HyperKG**'s performance on a dataset, please run:
        ```
        python example_train_poincare.py
        ```
        All parameters/hyperparameters can be altered by directly modifying the [example_train_poincare.py](example_train_poincare.py) file.

# Saved Models:
The folder [res/saved_models](res/saved_models) contains saved models for the experiments **WN18RR** and **FB15k-237**.
	
# Contact:
* prodromos DOT kolyvakis AT epfl DOT ch