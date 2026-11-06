# rag_minimal: 不用框架的最小 RAG

目的: 先懂原理, 再用框架. 整個專案只有約 170 行, 被追問「這一步在做什麼」時都答得出來.

## 安裝與執行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rag.py index                        # 讀 data/ 建索引
python rag.py ask "cgroups 可以限制什麼?" --no-llm   # 只看檢索, 不需 API key
cp .env.example .env && chmod 600 .env     # 編輯 .env, 填入 ANTHROPIC_API_KEY (.env 已被 .gitignore 排除)
python rag.py ask "cgroups 可以限制什麼?"            # 檢索 + 生成
python eval.py                              # 計算 hit@3
```

### 取得資料集 (維基百科條目)

`data/` 內的維基條目文字採 CC BY-SA 4.0 授權, 本 repo **不轉載**, 請自行下載:

```bash
pip install requests
WIKI_CONTACT=你的信箱 python scripts/fetch_wiki_artists.py   # Wikimedia 要求 User-Agent 帶聯絡方式
python rag.py index
python eval.py
```

- 下載後每份檔案開頭會附原網址, 版本連結與授權說明; 使用時請保留.
- 維基條目會被持續編輯, 你下載到的版本與我評估時不同, **hit@3 等數字可能略有差異**.
  我評估時用的版本見 `wiki_manifest.tsv` (版本永久連結).
- 腳本抓取的是各條目當下的最新版本, 無法精確還原舊版本.

### 英文資料集 (Kubernetes 官方文件)

`data/k8s/` 是 28 頁 Kubernetes concepts 文件 (CC BY 4.0), 同樣不進版控, 用腳本下載. 腳本固定在某個 commit, 結果可重現; 來源見 `k8s_manifest.tsv`.

```bash
python scripts/fetch_k8s_docs.py           # 下載到 data/k8s/
DATA_DIR=data/k8s python rag.py index      # 建英文索引 (存成 index_k8s.npz, 不會蓋掉中文索引)
DATA_DIR=data/k8s python rag.py ask "What is the default restartPolicy of a Pod?" --no-llm
python eval.py --k8s                       # 英文 20 題 (題庫在 eval_k8s.py)
```

### 取得資料集 (Ubuntu OEM kernel changelog)

CPU / GPU / NPU benchmark 用的英文資料集. 來源是 Ubuntu `linux-oem-7.0` 套件的 `debian.oem/changelog`
(公開 mirror), 每個版本一個檔案:

```bash
python scripts/fetch_oem_changelog.py                     # -> data/oem-7.0/linux-oem-7.0_7.0.0-10xx.xx.md
python scripts/fetch_oem_changelog.py --min 1010 --max 1018   # 只取部分版本
DATA_DIR=data/oem-7.0 python rag.py index
DATA_DIR=data/oem-7.0 python rag.py ask "Which version fixed the upside-down front camera?" --no-llm
python eval.py --oem                                      # 20 題 kernel 題庫 (eval_oem.py)
CHUNK_TITLE=1 DATA_DIR=data/oem-7.0 python rag.py index  # 每個 chunk 前加上標題 (版本號), 索引另存 *_title
CHUNK_TITLE=1 python eval.py --oem                        # 與沒加標題的 hit@3 比較
```

### 設定金鑰 (.env)

- API key 到 Claude Console (console.anthropic.com) 建立, 僅顯示一次, 請立刻存好.
- `.env` 只放在本機, **不要提交到 git**; `.gitignore` 已排除它. 可提交的是 `.env.example` (金鑰欄位留空).
- `rag.py` 啟動時會自動讀取同資料夾的 `.env` (由 python-dotenv 處理); 沒有 `.env` 時 `--no-llm` 模式仍可使用.
- 金鑰外洩時, 到 Console 刪除並重建.

第一次執行會下載 embedding 模型, 需要網路. 可用環境變數 `EMBED_MODEL`, `LLM_MODEL` 替換模型;
`EMBED_BACKEND=openvino` 與 `EMBED_DEVICE=CPU|GPU|NPU` 切換到 OpenVINO 後端.

## 檔案說明

- `rag.py`: 索引, 檢索, 生成的主程式
- `eval.py`: hit@k 檢索評估 (`--k8s`, `--oem` 切換英文題庫)
- `eval_oem.py`: Ubuntu linux-oem-7.0 kernel changelog 題庫 (20 題)
- `embed_ov.py`: OpenVINO embedding 後端 (CPU / GPU / NPU), `EMBED_BACKEND=openvino` 時使用
- `bench/`: CPU / GPU / NPU 效能與功耗 benchmark (見下方「在 CPU / GPU / NPU 上跑」); `bench/bench_ov_cpu.py` 找 OpenVINO CPU 比 PyTorch 慢的原因
- `data/`: 資料集資料夾 (藝人維基條目, **不進版控**, 用下方腳本下載; 也可用環境變數 `DATA_DIR` 指定其他資料夾)
- `docs/`: GitHub Pages 要發布的網頁檔案 (`index.html` 英文版, `zh.html` 繁體中文版, `og-image.png` 分享預覽圖)
- `scripts/fetch_wiki_artists.py`: 下載 20 位藝人的維基條目並轉成 Markdown
- `scripts/fetch_oem_changelog.py`: 下載 Ubuntu linux-oem-7.0 kernel changelog, 每個版本 (7.0.0-1001.1 到 1018.18) 存成一個檔案到 `data/oem-7.0/`
- `wiki_manifest.tsv`: 本專案評估所用條目的原始網址與版本連結
- `requirements.txt`: 相依套件
- `.env.example`: 環境變數範本 (複製為 `.env` 後填值)
- `.gitignore`: 排除 `.env`, 虛擬環境, 可重建的索引檔

## 程式在做什麼 (對照 rag.py 的 Step 1 到 5)

1. **載入與切塊**: `chunk_text` 用固定長度滑動視窗 (400 字元, 重疊 80). 簡單但會切斷語意.
2. **嵌入與建索引**: `embed` 把每個 chunk 轉成正規化向量, 存成 `index.npz`, 原文存 `chunks.json`.
3. **檢索**: 問題也轉成向量, 因為向量已正規化, 矩陣相乘就是 cosine similarity, 取分數最高的 k 個.
4. **組 prompt 與生成**: 把 chunk 編號後放進提示, 要求只依資料回答, 句尾標引用, 資料不足就拒答.
5. **量測**: 印出檢索耗時, LLM 耗時與 token 數, 這是日後做延遲與成本拆解的起點.

## 第一週練習

- [v] 換上自己的文件, 跑 `index` 與 `ask`, 觀察分數與命中的 chunk.
- [v] 在 `eval.py` 補到 20 題, 記下 baseline 的 hit@3.
- [v] 改 `CHUNK_SIZE` 與 `CHUNK_OVERLAP`, 重建索引再評估, 記錄差異.
- [v] 問一個文件裡沒有答案的問題, 看模型是否拒答.
- [v] 寫 5 行筆記: 問題, 方案, 權衡, 結果.

## Baseline 評估結果 (2026-10-22)

設定: 藝人維基文件 20 份 (`data/`), 固定長度切塊 (400 字元, 重疊 80), 向量檢索 only
(paraphrase-multilingual-MiniLM-L12-v2, cosine), 評估 k=3, 生成用 `TOP_K=4`. 題目共 20 題, 見 `eval.py`.
執行指令: `python eval.py` (只看檢索) 與 `python eval.py --llm` (另測拒答與人工檢查題).

**總結: hit@3 = 8/16 = 0.50** (有固定正確文件的 16 題; 12, 14, 18, 20 題沒有固定正確文件, 不計入)

| 題型 | hit@3 |
|---|---|
| single (單一條目) | 2/5 |
| multi_para (多段落整合) | 4/5 |
| cross (跨條目) | 1/3 |
| reasoning (多步推理) | 1/2 |
| boundary (檢索邊界) | 0/1 |

逐題結果 (OK = 正確文件出現在前 3 名):

| # | 結果 | 備註 |
|---|---|---|
| 1 周杰倫出道專輯 | OK | top1=0.657 |
| 2 鄧麗君過世 | OK | |
| 3 陳綺貞出道專輯 | MISS | 前 3 名是李宗盛, 周杰倫, top1=0.678 |
| 4 林志玲入行 | MISS | 前 3 名是刘若英, 蔡依林, 田馥甄 |
| 5 盧廣仲金曲獎 | MISS | 前 3 名是周杰倫, 李宗盛 |
| 6 蔡依林風格轉變 | OK | 前 3 名全是蔡依林 |
| 7 李宗盛幕後角色 | OK | |
| 8 張惠妹出道 | OK | |
| 9 劉若英跨足歌手 | MISS | 前 3 名是蕭敬騰, 李宗盛, 張惠妹, top1=0.732 |
| 10 羅大佑影響 | OK | top1=0.796 |
| 11 周杰倫 vs 蔡依林獎項 | OK | recall 1/2 |
| 12 歌手兼演員 (全庫) | 人工 | 見下方 |
| 13 田馥甄, 楊丞琳, 王心凌 | MISS | recall 0/3 |
| 14 金曲獎最佳演唱 (全庫) | 人工 | 見下方 |
| 15 伍佰 vs 羅大佑 | MISS | recall 0/2 |
| 16 金城武出生地 | OK | |
| 17 林依晨, 周渝民 | MISS | recall 0/2 |
| 18 出道最早 (全庫) | 人工 | 見下方 |
| 19 蕭敬騰經紀公司 | MISS | 正確文件沒被檢索到 |
| 20 張韶涵 x 羅志祥 | 拒答 | `REFUSED`, 拒答率 1/1 |

### 觀察

- **分數不能當相關度的依據**: 檢索錯誤的題目 top1 也有 0.62 到 0.73 (第 3, 4, 5, 9 題),
  甚至高於不少答對的題目. 所以光靠分數門檻過濾沒有用.
- **有明確人名的題目反而常失敗**: 第 3, 4, 5, 9, 13, 15, 17 題的問題都點名了藝人, 但檢索回來的是別人.
  可能原因 (尚未驗證): 切塊後的段落內容本身不一定含藝人姓名; 小型多語言模型對專有名詞的區分力有限;
  也可能受簡繁體差異影響 (例如檔名 `刘若英.md`, 題目寫 `劉若英`).
  這正是第 2 週要實驗 BM25 + RRF 混合檢索, 以及切塊時加上標題 (藝人名) 前綴的動機.
- **全庫統計題 (12, 14, 18) 本質上不適合 top-k 檢索**: LLM 的回答只涵蓋檢索到的 3 位藝人,
  並如實說明資料不足. 第 14 題只列出周杰倫與張惠妹, 沒有其他人. 這類題需要掃描全部文件 (例如 Agent 多次查詢,
  或事先建立結構化欄位), 不是調參數能解決的.
- **第 19 題的拒答是巧合, 不是正確推理**: 回答 "資料中找不到答案" 是因為蕭敬騰的文件根本沒被檢索到.
  文件裡其實有 2008 年簽華納唱片, 2014 年經紀公司喜鵲娛樂的資訊. 所以這題的問題出在檢索, 不在生成.
- **第 20 題拒答成功**, 但前 3 名都不是相關文件, 本來就沒有可引用的資料, 難度偏低.
- 兩次執行的檢索結果完全相同 (向量檢索是確定性的), 差別只在是否呼叫 LLM.

### 下一步實驗 (每次只改一個變數, 用同一組 20 題比較 hit@3)

1. 切塊時在每個 chunk 前加上檔名或標題 (藝人名).
2. 加入 BM25 並用 RRF 融合 (第 2 週).
3. 試不同 `CHUNK_SIZE` 與 `CHUNK_OVERLAP`.
4. 把簡體檔名 `刘若英.md` 與內容統一成繁體, 確認簡繁差異的影響.

## 在 CPU / GPU / NPU 上跑 (OpenVINO) 與效能比較

目標: 同一個 embedding 模型 (與一個小型 LLM) 分別跑在 Intel Meteor Lake 筆電的 CPU, 內顯 (GPU) 與 NPU,
比較速度, 功耗與結果是否一致. NPU 驅動安裝與功耗量測方法的筆記放在
[linux-platform-enablement](https://github.com/chinyakao/linux-platform-enablement).

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
- 只有向量檢索, 沒有 BM25 與 RRF (第 2 週).
- 索引存在 numpy 檔, 無法增量更新, 無多租戶 (第 5 到 8 週).
- `eval.py` 只量檢索命中, 不量答案品質 (第 9 週).
- 沒有錯誤處理, 重試與逾時.
