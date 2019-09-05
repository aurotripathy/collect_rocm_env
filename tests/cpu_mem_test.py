import subprocess
import sys
import os
from pudb import set_trace
from io import StringIO

def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    output = output.decode("utf-8").split('\n')
    
    rc = p.returncode
    return rc, output[4], output[7], output[8], output[12], output[21] 

attributes = run('cat /proc/cpuinfo')
for attribute in attributes[1:]:
    print(attribute)

