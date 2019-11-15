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
    3. __Known issues__:
    	There is a portability issue with the original C code provided by [OpenKE-PyTorch (old)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)). As a quick workaround, I added a Datatype control variable in the original **Config Class**. If a segmentation fault occurs after Step 2, then this command `con.set_int_type('int64')` should be commented out in both [example_train_poincare.py](example_train_poincare.py) and [example_test_poincare.py](example_test_poincare.py) files.

# Saved Models:
The folder [res/saved_models](res/saved_models) contains saved models for the experiments **WN18RR** and **FB15k-237**.
	
# Contact:
* prodromos DOT kolyvakis AT epfl DOT ch