```
1. 环境安装

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

2. 启动模型

# 进入openpi环境
source .venv/bin/activate
# 将launch_policy.py的17行改成存模型检查点的地址，启动policy
cd examples/franka
python launch_policy.py

```