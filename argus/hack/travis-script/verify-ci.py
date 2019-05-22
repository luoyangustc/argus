# coding=utf8
import subprocess
import re

# 这个包只有atnet用，其它的地方不应该用
# atnet client, atborad, qavactl, atcensor
# 这个rpc库error解析和其它base里面的不兼容，误用会导致error解析都是空的
# 如果仍然有使用该 rpc 库的需求，要把 out +1
out = int(subprocess.check_output(
    "git grep qiniu.com/ava/atnet/api/rpc -- '*.go' | wc -l", shell=True).decode('utf8').strip())
assert out == 6, "bad rpc lib:{}".format(out)
