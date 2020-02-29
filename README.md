# Group 4 Final project
More details about the project can be seen in [Proposal.pdf](Proposal.pdf).

## Introduction
Two main components:
* AegisEngine: Intel SGX + CUDA AES algorithm implementation
* PytorchAegis: Pytorch binding for AegisEngine

Usage:
1. Build AegisEngine in PreRelease mode
2. Copy the dll and lib files to PytorchAegis
3. Install PytorchAegis with `setup.py`

See `PytorchAegis/test_run.py` for the interface usage.

