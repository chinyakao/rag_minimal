"""Embedding 效能比較: 同一個模型在 CPU / GPU / NPU (OpenVINO) 與 PyTorch CPU 基準.

量測項目 (每個裝置):
    compile_s     模型編譯時間 (有 CACHE_DIR 時第二次會快很多; 加 --no-cache 量冷啟動)
    index_cps     建索引吞吐量 chunks/s (固定 batch, 持續至少 --min-seconds 秒)
    query_p50/p95 單筆查詢延遲 ms (batch=1, 模擬使用者問問題)
    cos_vs_ref    與 PyTorch 基準向量的 cosine 平均值 (確認換裝置後結果沒跑掉)
    *_W / *_J     RAPL 平均功率與能量 (可讀時), 以及 J/chunk
    npu_util      NPU 使用率 (intel_vpu 的 npu_busy_time_us, 可讀時)

用法 (在 repo 根目錄):
    python rag.py index                                 # 先產生 chunks.json
    python bench/bench_embed.py                         # 預設比較 st-CPU, CPU, GPU, NPU
    python bench/bench_embed.py --devices NPU --no-cache
    sudo -E .venv/bin/python bench/bench_embed.py       # 需要 root 才能讀 RAPL 時
結果另存 bench/results/embed_<時間>.jsonl.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import rag  # noqa: E402
from ownership import give_back  # noqa: E402
from power import PowerMeter, idle_baseline  # noqa: E402


def load_corpus(n):
    if rag.CHUNKS_FILE.exists():
        texts = [r["text"] for r in json.loads(rag.CHUNKS_FILE.read_text(encoding="utf-8"))]
    else:
        texts = [c for _, t in rag.load_docs() for c in rag.chunk_text(t)]
    if not texts:
        sys.exit("沒有資料, 請先下載資料集並執行 python rag.py index")
    # 不夠 n 筆就重複, 讓每個裝置處理的量一樣
    return (texts * (n // len(texts) + 1))[:n]


def load_queries():
    """用 kernel changelog 的評估題目當查詢."""
    from eval_oem import EVAL_SET_OEM as items
    return [item["q"] for item in items]


def make_backend(name, batch, seq_len, use_cache):
    """回傳 (encode 函式, compile 秒數)."""
    if name == "st-CPU":
        from sentence_transformers import SentenceTransformer
        t0 = time.perf_counter()
        m = SentenceTransformer(rag.EMBED_MODEL, device="cpu")
        load_s = time.perf_counter() - t0

        def enc(texts):
            return np.asarray(m.encode(texts, batch_size=batch, normalize_embeddings=True),
                              dtype=np.float32)
        return enc, load_s

    from embed_ov import OVEmbedder
    m = OVEmbedder(rag.EMBED_MODEL, device=name, batch=batch, seq_len=seq_len, use_cache=use_cache)
    return m.encode, m.compile_s


def timed_loop(fn, min_seconds, meter):
    """重複執行 fn 直到超過 min_seconds, 回傳 (執行次數, 秒數, 功耗統計)."""
    meter.start()
    t0, n = time.perf_counter(), 0
    while True:
        fn()
        n += 1
        if time.perf_counter() - t0 >= min_seconds:
            break
    stats = meter.stop()
    return n, stats["seconds"], stats


def bench_device(name, args, corpus, queries, ref_vecs, meter):
    row = {"device": name, "model": rag.EMBED_MODEL, "batch": args.batch, "seq_len": args.seq_len,
           "query_seq_len": args.query_seq_len}

    # 1) 建索引吞吐量
    enc, row["compile_s"] = make_backend(name, args.batch, args.seq_len, not args.no_cache)
    enc(corpus[:args.batch * 2])  # 暖機
    n, secs, p = timed_loop(lambda: enc(corpus), args.min_seconds, meter)
    row["index_cps"] = n * len(corpus) / secs
    for k, v in p.items():
        if k != "seconds" and v is not None:
            row[f"index_{k}"] = v
    pkg = next((k for k in p if k.endswith("_J") and "package" in k), None)
    if pkg:
        row["index_J_per_chunk"] = p[pkg] / (n * len(corpus))

    # 2) 與基準的一致性
    vecs = enc(corpus[:len(ref_vecs)])
    row["cos_vs_ref"] = float(np.mean(np.sum(vecs * ref_vecs, axis=1)))
    row["cos_vs_ref_min"] = float(np.min(np.sum(vecs * ref_vecs, axis=1)))

    # 3) 單筆查詢延遲 (batch=1)
    q_enc, _ = make_backend(name, 1, args.query_seq_len, not args.no_cache)
    for q in queries[:args.warmup]:
        q_enc([q])
    lat = []
    meter.start()
    for i in range(args.query_iters):
        t0 = time.perf_counter()
        q_enc([queries[i % len(queries)]])
        lat.append((time.perf_counter() - t0) * 1000)
    p = meter.stop()
    row["query_p50_ms"] = float(np.percentile(lat, 50))
    row["query_p95_ms"] = float(np.percentile(lat, 95))
    for k, v in p.items():
        if k.endswith("_W") or k == "npu_util":
            row[f"query_{k}"] = v
    return row


def print_table(rows):
    cols = ["device", "compile_s", "index_cps", "query_p50_ms", "query_p95_ms", "cos_vs_ref"]
    extra = [k for k in rows[0] if k.startswith("index_") and (k.endswith("_W") or k == "index_npu_util")]
    extra += ["index_J_per_chunk"] if any("index_J_per_chunk" in r for r in rows) else []
    cols += extra
    print("\n" + " | ".join(f"{c:>14s}" for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:>14.4g}" if isinstance(v, float) else f"{str(v):>14s}")
        print(" | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=["st-CPU", "CPU", "GPU", "NPU"],
                    help="st-CPU = sentence-transformers (PyTorch) 基準; 其餘為 OpenVINO 裝置名")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--query-seq-len", type=int, default=128,
                    help="查詢用的固定長度; 查詢平均約 21 token, 設 32 可少算很多 (NPU 也適用)")
    ap.add_argument("--n-chunks", type=int, default=256, help="建索引測試用的 chunk 數")
    ap.add_argument("--min-seconds", type=float, default=10.0, help="每項量測至少跑幾秒 (功耗需要夠長)")
    ap.add_argument("--query-iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--idle-seconds", type=float, default=5.0, help="先量閒置功耗; 0 = 不量")
    ap.add_argument("--cooldown", type=float, default=5.0, help="裝置之間休息秒數, 讓溫度與功耗回落")
    ap.add_argument("--no-cache", action="store_true", help="不用 OpenVINO 編譯快取 (量冷啟動)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meter = PowerMeter()
    print("功耗計數器:", meter.available)
    corpus, queries = load_corpus(args.n_chunks), load_queries()

    # 基準向量: PyTorch CPU 的結果
    from sentence_transformers import SentenceTransformer
    ref_vecs = np.asarray(SentenceTransformer(rag.EMBED_MODEL, device="cpu").encode(
        corpus[:64], normalize_embeddings=True), dtype=np.float32)

    meta = {"time": datetime.now().isoformat(timespec="seconds")}
    if args.idle_seconds > 0 and meter.domains:
        meta["idle"] = idle_baseline(args.idle_seconds)
        print("閒置功耗:", {k: round(v, 2) for k, v in meta["idle"].items() if v is not None})

    rows = []
    for dev in args.devices:
        print(f"\n== {dev} ==")
        try:
            row = bench_device(dev, args, corpus, queries, ref_vecs, meter)
        except Exception as e:  # 某個裝置不支援時繼續跑下一個
            print(f"  失敗: {type(e).__name__}: {e}")
            continue
        row.update(meta)
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "idle"})
        time.sleep(args.cooldown)

    if not rows:
        sys.exit("沒有任何裝置跑成功")
    print_table(rows)

    out = args.out or ROOT / "bench" / "results" / f"embed_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    give_back(out.parent, ROOT / "ov_models", ROOT / "ov_cache")
    print(f"\n結果已存到 {out}")


if __name__ == "__main__":
    main()
