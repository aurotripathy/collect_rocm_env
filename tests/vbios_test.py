import subprocess
import subprocess
import sys
import os
import re
from pudb import set_trace

def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    return rc, output.strip(), err.strip()


def run_and_parse_first_match(run_lambda, command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, command_out, _ = run_lambda(command)
    print('commmand out')
    print(command_out)
    if rc != 0:
        return None
    match = re.search(regex, command_out)
    if match is None:
        return None
    lines = match.group(1).split('\n')
    total_gpus = len(lines) - 4
    print('Total GPUs:', total_gpus)
    set_trace()

    print("needed lines")
  
    # print(lines[2 : 2 + total_gpus])
    return [line for line in lines[2 : 2 + total_gpus]]
    # return match

run_lambda = run
out = run_and_parse_first_match(run_lambda,
                                '/opt/rocm/bin/rocm-smi -v',
                                r'((?s).*)')

print(out)

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
print color.GREEN + 'Hello World !' + color.END
