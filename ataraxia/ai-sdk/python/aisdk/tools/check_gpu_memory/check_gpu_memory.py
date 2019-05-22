import subprocess
import re

line_regex = re.compile(r'^\|\s+\d+\s+(\d+)\s+C\s+\S+\s+(\d+)MiB\s+\|')
for line in subprocess.check_output('nvidia-smi').split('\n'):
    if 'python' in line:
        r = line_regex.match(line)
        assert r
        pid = r.group(1)
        mem = r.group(2)
        print('-----')
        print(line)
        print(subprocess.check_output(['ps', '--pid', str(pid), '-u']))
        print('-----\n')
