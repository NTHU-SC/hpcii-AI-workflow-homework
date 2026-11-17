# HPC AI Workflow 作業

這份作業要你們寫一個 judge，判斷 `heat` 程式的候選版本是否正確、是否真的比較快。

- [`ASSIGNMENT.md`](ASSIGNMENT.md)：作業內容、時程、繳交項目
- [`RULES.md`](RULES.md)：候選和 judge 的規則

## 內容

```
baseline/            baseline 候選：heat.c、build.sh、run.sh
inputs/default.in    預設輸入
reference/default.out  baseline 對預設輸入的輸出（512×512 個 little-endian float64，二進位檔）
reference/default.png  上面那個輸出畫成的圖（用 tools/show.py 產生）
tools/show.py        看輸出檔：印出摘要、畫成 PNG、比較兩個輸出
```

## 先跑一次 baseline

```bash
bash baseline/build.sh
srun -w <你們的節點> -c 8 --hint=nomultithread \
    bash baseline/run.sh inputs/default.in out.bin
cmp out.bin reference/default.out && echo same
```

## 看輸出檔

輸出檔是二進位格式，文字編輯器打不開。用 `tools/show.py` 看（art 兩台都已經裝好 NumPy 和 matplotlib）：

```bash
python3 tools/show.py reference/default.out              # 印出摘要，並畫成 reference/default.png
python3 tools/show.py out.bin --diff reference/default.out   # 比較兩個輸出：最大誤差在哪裡，差異畫成圖
```

圖的下緣是板子的第 0 列（下邊），左緣是第 0 行（左邊）。`--diff` 畫的是 log10 |A − B|，白色代表完全相同。

自己寫程式讀的話：

```python
import numpy as np
T = np.fromfile("reference/default.out", dtype="<f8").reshape(512, 512)
# T[j, i]：j 是列（0 = 下邊），i 是行（0 = 左邊）
```
