#### Collect ROCm Environment Parameters

This script outputs system environment info pertaining to ROCm

Its inspired by PyTorch's own environment collection tool:
https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py

##### Invocation

Run it with `python collect_rocm_env.py`.

##### Expected Output

The expected output would be something like:
<code>
root@prj47-rack-40:/collect_rocm_env# python3.6 collect_rocm_env.py
        
Collecting environment information...

PyTorch version: 1.1.0a0+1e42720

OS: Ubuntu 16.04.5 LTS

Kernel: 4.20.0-rc3-kfd-compute-roc-master-9702

VBIOS version:

        Total GPUs:4
        
        GPU[1]          : VBIOS version: 113-D0513100-004
        
        GPU[2]          : VBIOS version: 113-D0513100-004
        
        GPU[3]          : VBIOS version: 113-D0513100-004
        
        GPU[4]          : VBIOS version: 113-D0513100-004
        

ROCm version: : 2.2.31

MIOpen version: 1.7.1

