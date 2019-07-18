#### Collect ROCm Environment Parameters

This script outputs system environment info pertaining to ROCm and application middleware.

Its inspired by PyTorch's own environment collection tool:
https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py

##### Invocation

Run it like so: `python collect_rocm_env.py`.

##### Expected Output

The expected output would be something like:

<code>  
Collecting environment information...
        
PyTorch version: 1.1.0a0+17232fb

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

Large Bar status:

        23:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Vega 10 [Radeon Instinct MI25] (rev 01)
                Region 0: Memory at 1a000000000 (64-bit, prefetchable) [size=16G]
                Large Bar Enabled

        26:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Vega 10 [Radeon Instinct MI25] (rev 01)
                Region 0: Memory at 19800000000 (64-bit, prefetchable) [size=16G]
                Large Bar Enabled

        63:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Vega 10 [Radeon Instinct MI25] (rev 01)
                Region 0: Memory at 12c00000000 (64-bit, prefetchable) [size=16G]
                Large Bar Enabled

        66:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Vega 10 [Radeon Instinct MI25] (rev 01)
                Region 0: Memory at 12400000000 (64-bit, prefetchable) [size=16G]
                Large Bar Enabled
</code>

#### Dependency

The tool needs the unix command `lspci` to gather Large Bar memory status. 
`lspci` can be easily installed with `apt install pciutils`
