import subprocess
import subprocess
import sys
import os
import re

def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    return rc, output.strip(), err.strip()


def run_and_parse_first_match(run_lambda, command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    # set_trace()
    return match.group(1)

run_lambda = run
out = run_and_parse_first_match(run_lambda,
                                '/opt/rocm/bin/rocm-smi -v',
                                r'(^(GPU(.*)))+')

print(out)

