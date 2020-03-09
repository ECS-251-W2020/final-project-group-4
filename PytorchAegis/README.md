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
   * Make sure the Windows [SDK of Intel SGX](https://software.intel.com/en-us/sgx/sdk) with version 2.X 2.6.100.2 is installed
   * Open [AegisEngin.sln](../AegisEngine/AegisEngine.sln) in Visual Studio 2017
   * Build two projects in prerelease mode under window x64
2. Copy the dll and lib all 3 files to [PytorchAegis](./)
   * `AegisEngine/x64/Prerelease/AegisEngine.dll`
   * `AegisEngine/x64/Prerelease/AegisEngine.lib`
   * `AegisEngine/x64/Prerelease/StorageEnclave.signed.dll`
3. Install PytorchAegis with [setup.py](./setup.py)
        
        > python setup.py install

See [test_run.py](./test_run.py) for the interface usage for testing the demo.

## Clarification

- The demo here is to make sure we can run the whole thing. But some thing needed to make it clear. Practically, we do not touch host memory with the exact data. Here in the [test_run.py](./test_run.py), we load a image from disk to host memory and then copy to GPU memory, however, we won't do that since the training data in main memory from user should be encrpyted.