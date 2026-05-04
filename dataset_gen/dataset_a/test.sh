python scripts/04_init_gs_from_first_frame.py \
    --data_dir outputs/data \
    --config configs/default.yaml \
    --backend mesh \
    --num_workers 1 \
    --device cuda \
    --limit 5 2>&1 | tee step4_smoke.log

# 验证 5 个 ply 都生成了
find outputs/data -name "init_gs.ply" | wc -l
ls -la outputs/data/$(ls outputs/data | head -1)/init_gs.ply

# 看一个 ply 的点数(应该 ~50000)
python -c "
import numpy as np
from pathlib import Path
# 用最简单方式读 ply header
p = list(Path('outputs/data').glob('*/init_gs.ply'))[0]
print('file:', p)
print('size:', p.stat().st_size, 'bytes')
with open(p, 'rb') as f:
    header = f.read(2048).decode('ascii', errors='ignore')
    print(header[:500])
"

# ─── C. Step 4 全量 (16 worker, ~5-10 min) ───
WORLD_GPU_COUNT=4 \
nohup python scripts/04_init_gs_from_first_frame.py \
    --data_dir outputs/data \
    --config configs/default.yaml \
    --backend mesh \
    --num_workers 16 \
    --device cuda > step4_full.log 2>&1 &

echo "Step 4 PID: $!"

# 监控
tail -f step4_full.log

# ─── D. 完成后验证 ───
find outputs/data -name "init_gs.ply" | wc -l    # 期待 3238
du -sh outputs/data/                              # 看占用增加多少
grep -E "ok=|skipped|fail|Done\." step4_full.log | tail -5