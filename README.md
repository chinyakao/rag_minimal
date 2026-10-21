# rag_minimal: 不用框架的最小 RAG

目的: 先懂原理, 再用框架. 整個專案只有約 170 行, 面試被追問「這一步在做什麼」時你都答得出來.

## 安裝與執行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rag.py index                        # 讀 docs/ 建索引
python rag.py ask "cgroups 可以限制什麼?" --no-llm   # 只看檢索, 不需 API key
cp .env.example .env && chmod 600 .env     # 編輯 .env, 填入 ANTHROPIC_API_KEY (.env 已被 .gitignore 排除)
python rag.py ask "cgroups 可以限制什麼?"            # 檢索 + 生成
python eval.py                              # 計算 hit@3
```

### 設定金鑰 (.env)

- API key 到 Claude Console (console.anthropic.com) 建立, 僅顯示一次, 請立刻存好.
- `.env` 只放在本機, **不要提交到 git**; `.gitignore` 已排除它. 可提交的是 `.env.example` (金鑰欄位留空).
- `rag.py` 啟動時會自動讀取同資料夾的 `.env` (由 python-dotenv 處理); 沒有 `.env` 時 `--no-llm` 模式仍可使用.
- 金鑰外洩時, 到 Console 刪除並重建.

第一次執行會下載 embedding 模型, 需要網路. 可用環境變數 `EMBED_MODEL`, `LLM_MODEL` 替換模型.
`docs/` 內附兩份範例文件, 請換成你自己的 (非公司內部) 資料.

## 檔案說明

- `rag.py`: 索引, 檢索, 生成的主程式
- `eval.py`: hit@k 檢索評估
- `docs/`: 範例文件 (請換成自己的非公司資料)
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

- [ ] 換上自己的文件, 跑 `index` 與 `ask`, 觀察分數與命中的 chunk.
- [ ] 在 `eval.py` 補到 20 題, 記下 baseline 的 hit@3.
- [ ] 改 `CHUNK_SIZE` 與 `CHUNK_OVERLAP`, 重建索引再評估, 記錄差異.
- [ ] 問一個文件裡沒有答案的問題, 看模型是否拒答.
- [ ] 寫 5 行筆記: 問題, 方案, 權衡, 結果.

## 已知限制 (就是之後要改進的地方)

- 固定長度切塊會切斷句子, 可改成依標題或段落切.
- 只有向量檢索, 沒有 BM25 與 RRF (第 2 週).
- 索引存在 numpy 檔, 無法增量更新, 無多租戶 (第 5 到 8 週).
- `eval.py` 只量檢索命中, 不量答案品質 (第 9 週).
- 沒有錯誤處理, 重試與逾時.
