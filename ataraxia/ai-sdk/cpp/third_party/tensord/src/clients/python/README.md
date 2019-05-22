
# 代码生成

```bash
python -m grpc_tools.protoc \
       -Isrc/servers/grpc \
       --python_out=src/clients/python \
       --grpc_python_out=src/clients/python \
       src/servers/grpc/tensord.proto
```
