# rag_minimal: 不用框架的最小 RAG

目的: 先懂原理, 再用框架. 不用 LangChain, 只用 numpy + sentence-transformers + Anthropic SDK,
被追問「這一步在做什麼」時都答得出來.

資料集是公開的 Ubuntu `linux-oem-7.0` kernel changelog (18 個版本, 3,588 個 chunk), 題庫 28 題.
學習日誌: [Week 1](https://chinyakao.github.io/rag_minimal/) ·
[Week 2–3](https://chinyakao.github.io/rag_minimal/week2-3.html) ·
[CPU / GPU / NPU 筆記](https://chinyakao.github.io/rag_minimal/notes/npu.html)

## 安裝與執行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_oem_changelog.py      # 下載資料到 data/oem-7.0/ (每個版本一個檔案)
python rag.py index                        # 建索引
python rag.py ask "Which version fixed the upside-down front camera?" --no-llm   # 只看檢索, 不需 API key
cp .env.example .env && chmod 600 .env     # 編輯 .env, 填入 ANTHROPIC_API_KEY (.env 已被 .gitignore 排除)
python rag.py ask "Which version fixed the upside-down front camera?"            # 檢索 + 生成
python eval.py                             # 28 題 hit@3 (只看檢索, 不花 API 費用)
python eval.py --bm25                      # 改用 BM25
python eval.py --hybrid                    # 向量 + BM25, RRF 融合
```

### 資料集 (Ubuntu OEM kernel changelog)

來源是 Ubuntu `linux-oem-7.0` 套件的 `debian.oem/changelog` (公開 mirror), 每個版本 (7.0.0-1001.1 到 1018.18)
存成一個檔案. `data/` 不進版控, 用腳本下載:

```bash
python scripts/fetch_oem_changelog.py                         # -> data/oem-7.0/linux-oem-7.0_7.0.0-10xx.xx.md
python scripts/fetch_oem_changelog.py --min 1010 --max 1018   # 只取部分版本
CHUNK_TITLE=1 python rag.py index                             # 每個 chunk 前加上版本標題, 索引另存 *_title
CHUNK_TITLE=1 python eval.py                                  # 與沒加標題的 hit@3 比較
```

### 設定金鑰 (.env)

- API key 到 Claude Console (console.anthropic.com) 建立, 僅顯示一次, 請立刻存好.
- `.env` 只放在本機, **不要提交到 git**; `.gitignore` 已排除它. 可提交的是 `.env.example` (金鑰欄位留空).
- `rag.py` 啟動時會自動讀取同資料夾的 `.env` (由 python-dotenv 處理); 沒有 `.env` 時 `--no-llm` 模式仍可使用.
- 金鑰外洩時, 到 Console 刪除並重建.

第一次執行會下載 embedding 模型, 需要網路. 可用環境變數 `EMBED_MODEL`, `LLM_MODEL` 替換模型;
`EMBED_BACKEND=openvino` 與 `EMBED_DEVICE=CPU|GPU|NPU` 切換到 OpenVINO 後端.

## 檔案說明

- `rag.py`: 索引, 檢索 (vector / BM25 / hybrid), 生成的主程式
- `eval.py`: hit@k 檢索評估
- `eval_oem.py`: 題庫 28 題 (原本 20 題 + 8 題換句話說)
- `embed_ov.py`: OpenVINO embedding 後端 (CPU / GPU / NPU), `EMBED_BACKEND=openvino` 時使用
- `demo.py`: 現場 demo, 問 changelog 問題, 同一個本機小模型依序跑在 GPU 與 NPU, 即時顯示速度與功耗 (見下方「現場 demo」)
- `bench/`: CPU / GPU / NPU 效能與功耗 benchmark (見下方「在 CPU / GPU / NPU 上跑」); `bench/bench_ov_cpu.py` 找 OpenVINO CPU 比 PyTorch 慢的原因
- `scripts/fetch_oem_changelog.py`: 下載 kernel changelog 到 `data/oem-7.0/`
- `data/`: 資料集資料夾 (**不進版控**; 預設讀 `data/oem-7.0/`, 也可用環境變數 `DATA_DIR` 指定其他資料夾)
- `docs/`: GitHub Pages 學習日誌 (英文與繁體中文), `docs/notes/` 是較詳細的學習筆記
- `archive/`: 早期用過的資料集與題庫 (中文藝人維基條目, Kubernetes 文件), 說明見 [`archive/README.md`](archive/README.md)
- `requirements.txt`: 相依套件
- `.env.example`: 環境變數範本 (複製為 `.env` 後填值)
- `.gitignore`: 排除 `.env`, 虛擬環境, 可重建的索引檔

## 程式在做什麼 (對照 rag.py 的 Step 1 到 5)

1. **載入與切塊**: `chunk_text` 用固定長度滑動視窗 (400 字元, 重疊 80). 簡單但會切斷語意.
2. **嵌入與建索引**: `embed` 把每個 chunk 轉成正規化向量, 存成 `index_oem-7.0.npz`, 原文存 `chunks_oem-7.0.json`.
3. **檢索**: 問題也轉成向量, 因為向量已正規化, 矩陣相乘就是 cosine similarity, 取分數最高的 k 個.
   用 `RETRIEVAL` 環境變數 (或 eval.py 的 `--bm25` / `--hybrid`) 切換三種方式:
   - `vector` (預設): 只用向量.
   - `bm25`: 手寫 BM25 關鍵字檢索. 保留 `0xc05`, `ov02c10`, `1018.18` 這類字, 讓它們能精確比對 (中文則用相鄰兩字 bigram).
     直接從 `chunks*.json` 建, 不需另外的索引檔.
   - `hybrid`: 向量與 BM25 各取前 50 名, 用 RRF (`1/(60 + 名次)` 加總) 融合. 只看名次, 所以兩邊分數尺度不同也沒關係.
     `BM25_WEIGHT=2` 把 BM25 那一邊乘 2 (加權 RRF), 適合向量明顯比較弱的資料.
4. **組 prompt 與生成**: 把 chunk 編號後放進提示, 要求只依資料回答, 句尾標引用, 資料不足就拒答.
5. **量測**: 印出檢索耗時, LLM 耗時與 token 數, 這是日後做延遲與成本拆解的起點.

## 評估結果 (2026-11-07, hit@3)

設定: 固定長度切塊 (400 字元, 重疊 80), embedding 模型 paraphrase-multilingual-MiniLM-L12-v2, k=3.
第 19, 20 題應拒答, 沒有正確文件, 不計分. 第 21 到 28 題用使用者描述症狀的方式問, 避開 changelog 原文用字.

| 方法 | 原 20 題中的 18 題 | 換句話說 8 題 | 全部 26 題 |
|---|---|---|---|
| vector (預設) | 13/18 | 3/8 | 16/26 = 0.62 |
| **bm25** | **18/18** | 3/8 | **21/26 = 0.81** |
| hybrid (RRF 等權重) | 17/18 | 3/8 | 20/26 = 0.77 |
| hybrid (`BM25_WEIGHT=2`) | 17/18 | 2/8 | 19/26 = 0.73 |

觀察 (詳細見 [Week 2–3 學習日誌](https://chinyakao.github.io/rag_minimal/week2-3-zh.html)):

- 版本號, `0xC05` 這類精確字串, 向量檢索抓不到, BM25 可以. 每個 chunk 加版本標題 (`CHUNK_TITLE=1`) 沒有幫助.
- 題目照抄原文時 BM25 看起來完美; 換成使用者的說法後, 每種方法都只對 3 題以下.
- 換句話說的題目, 向量和 BM25 對的是不同題 (聯集 5/8), 但 RRF 獎勵兩邊都排前面的 chunk, 所以 hybrid 拿不到聯集. 加權也沒用.
- 瓶頸在 embedding 模型: 小型通用模型對 kernel 領域的換句話說很弱.

下一步: 換英文檢索模型 (例如 `EMBED_MODEL=BAAI/bge-small-en-v1.5`) 重跑; 試 cross-encoder reranker; 用問題裡的版本號過濾.

## 在 CPU / GPU / NPU 上跑 (OpenVINO) 與效能比較

目標: 同一個 embedding 模型 (與一個小型 LLM) 分別跑在 Intel Meteor Lake 筆電的 CPU, 內顯 (GPU) 與 NPU,
比較速度, 功耗與結果是否一致. NPU 驅動安裝與功耗量測方法的筆記放在
[client-linux-notes](https://github.com/chinyakao/client-linux-notes).

### 前置條件 (Linux)

- kernel 6.8 以上 (內建 `intel_vpu` 驅動), 安裝 [intel/linux-npu-driver](https://github.com/intel/linux-npu-driver) Release 的 user-space 套件.
- `/dev/accel/accel0` 存在, 帳號在 `render` 群組 (`sudo usermod -aG render $USER` 後重新登入).
- GPU 需要 Intel compute runtime (`intel-opencl-icd` 或 Release 的 .deb).
- 確認 OpenVINO 看得到三個裝置: `python embed_ov.py` 應列出 `CPU`, `GPU`, `NPU`.

```bash
pip install -r requirements.txt -r bench/requirements-bench.txt
python embed_ov.py                                        # 列出可用裝置

# rag.py 改用 OpenVINO 後端 (第一次會把模型轉成 IR, 存到 ov_models/)
EMBED_BACKEND=openvino EMBED_DEVICE=NPU python rag.py index
EMBED_BACKEND=openvino EMBED_DEVICE=NPU python eval.py    # hit@3 應與 PyTorch 版本相同或幾乎相同
```

注意: 建索引與查詢要用同一個後端 (向量來自同一個模型, 但不同裝置的數值有微小差異).

### Embedding benchmark (`bench/bench_embed.py`)

```bash
python bench/bench_embed.py                               # st-CPU (PyTorch 基準), CPU, GPU, NPU
python bench/bench_embed.py --devices NPU --no-cache      # 量 NPU 冷啟動編譯時間
```

量測: 編譯時間, 建索引吞吐量 (chunks/s, batch 8), 單筆查詢延遲 p50/p95 (batch 1), 與 PyTorch 向量的 cosine,
以及 RAPL 平均功率, 每個 chunk 的能量 (J/chunk) 和 NPU 使用率. 結果存在 `bench/results/*.jsonl` (不進版控).

NPU 只支援靜態形狀, 所以 OpenVINO 後端一律用固定的 `[batch, 128]` 輸入 (128 是此模型的 max_seq_length),
三個裝置用同樣形狀才公平.

### 小型 LLM benchmark (`bench/bench_llm.py`)

```bash
optimum-cli export openvino -m Qwen/Qwen2.5-1.5B-Instruct \
    --weight-format int4 --sym --group-size -1 --ratio 1.0 ov_models/qwen2.5-1.5b-int4
python bench/bench_llm.py --model ov_models/qwen2.5-1.5b-int4
```

用 `rag.py` 真正檢索出的 RAG prompt, 量載入時間, TTFT (prefill), 每個 token 時間 (decode), tokens/s 與 J/token.

### 現場 demo (`demo.py`)

給分享會用: 輸入一個 changelog 問題, 先顯示 BM25 找到的段落, 再把同一個 prompt **依序**丟給 GPU 與 NPU 上的
Qwen2.5-1.5B (int4), 回答一個字一個字流出來, 每個裝置結束後顯示首字時間, tokens/s, SoC 功率與比閒置多用的能量,
最後並排比較. 依序跑是因為 GPU 與 NPU 共用記憶體頻寬與 SoC 功耗預算, 同時跑數字會失真.

```bash
python demo.py --model ov_models/qwen2.5-1.5b-int4                      # 互動模式 (q 離開)
python demo.py --model ov_models/qwen2.5-1.5b-int4 --claude             # 最後加上雲端 Claude 當品質對照
sudo -E .venv/bin/python demo.py --model ov_models/qwen2.5-1.5b-int4    # 要顯示功耗時
```

- 啟動時就載入並暖機所有裝置 (NPU 第一次編譯最久, 有 `ov_cache/` 後會快很多), 請在開場前先開好.
- 預設檢索是 BM25, 不需要先建 embedding 索引; `--retrieval vector|hybrid` 則要先 `python rag.py index`.
- 生成中按 Ctrl-C 只中斷目前的裝置; 某個裝置載入失敗會略過, 不影響其他裝置.
- 回答用 greedy decoding, 同一題每次結果相同, 方便事先排練.
- 本機小模型用專用的英文 prompt (每段資料前標版本號, 自訂 system message) 並加 `--repetition-penalty 1.1`,
  避免 1.5B 模型在 greedy 下重複鬼打牆; `--claude` 仍用 `rag.py` 原本的 prompt.

### 功耗量測的權限

`bench/power.py` 讀 `/sys/class/powercap/intel-rapl:*/energy_uj` (package 包含 CPU, iGPU 與 NPU)
和 `/sys/class/accel/accel0/device/npu_busy_time_us`. 多數發行版 `energy_uj` 只有 root 可讀, 兩種做法:

```bash
sudo -E .venv/bin/python bench/bench_embed.py              # 直接用 root 跑
sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj # 暫時開放讀取 (重開機後恢復)
```

讀不到時 benchmark 照常跑, 只是沒有功耗欄位. 量測時請接上電源, 關掉其他程式, 每個裝置之間有冷卻時間.

## 已知限制 (就是之後要改進的地方)

- 固定長度切塊會切斷句子, 可改成依標題或段落切.
- BM25 每次啟動都從 chunks 重建 (幾千個 chunk 不到 1 秒); 資料量大時要改用倒排索引或搜尋引擎.
- 索引存在 numpy 檔, 無法增量更新, 無多租戶 (第 5 到 8 週).
- `eval.py` 只量檢索命中, 不量答案品質 (第 9 週).
- 沒有錯誤處理, 重試與逾時.
