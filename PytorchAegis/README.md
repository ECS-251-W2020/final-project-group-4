# Final project 'PytorchAegis' demo

## Dependencies

- Windows OS x64
- Intel CPU compatible with SGX
- Visual Studio 2017 community
- Cuda 10.1
- Python 3.7.2
- Pytorch 1.4.0 stable
- Some Python libraries are not listed here

## Usage:

1. Build [AegisEngine](../AegisEngine/) in PreRelease mode
2. Copy the dll and lib all 3 files to [PytorchAegis](./)
3. Install PytorchAegis with [setup.py](./setup.py)

See [test_run.py](./test_run.py) for the interface usage for testing the demo.

## Clarification

- The demo here is to make sure we can run the whole thing. But some thing needed to make it clear. Practically, we do not touch host memory with the exact data. Here in the `test_run.py`, we load a image from disk to host memory and then copy to GPU memory, however, we won't do that since the training data in main memory from user should be encrpyted.