cd /home/zejun/CAP-A2GN/data/dataset_b

# 看 GPU 是否空(chendong 跑完了的话现在 4 张 A100 都空)
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# Smoke:5 条,验证 pipeline + 模型下载
python scripts/03_estimate_depth.py \
    --data_dir outputs/data \
    --config configs/default.yaml \
    --device cuda \
    --limit 5 2>&1 | tee step3_smoke.log

# 看一个 depth 结果(范围合理性 + 形状)
python -c "
import numpy as np
from pathlib import Path
d_files = list(Path('outputs/data').glob('*/depth.npz'))
print(f'depth.npz files so far: {len(d_files)}')
data = np.load(d_files[0])
print(f'\nfirst file: {d_files[0]}')
print(f'  keys: {list(data.keys())}')
print(f'  depth shape: {data[\"depth\"].shape}')
print(f'  depth min/median/max: {data[\"depth\"].min():.3f} / {np.median(data[\"depth\"]):.3f} / {data[\"depth\"].max():.3f} m')
print(f'  is_metric: {bool(data[\"is_metric\"])}')
print(f'  model: {str(data[\"model\"])}')
print(f'  variant: {str(data[\"variant\"])}')
"