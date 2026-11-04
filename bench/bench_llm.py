"""小型 LLM 生成效能比較: 同一個 int4 模型用 OpenVINO GenAI 跑在 CPU / GPU / NPU.

prompt 用 rag.py 真的檢索出來的 RAG prompt (問題 + 4 個 chunk), 量的是本專案實際的生成情境.

量測項目 (每個裝置):
    load_s        載入 + 編譯時間 (NPU 第一次很久, 有 CACHE_DIR 後會變快)
    ttft_ms       time to first token, 主要是 prefill (讀完整個 prompt)
    tpot_ms       time per output token, decode 階段每個 token 的時間
    tok_per_s     decode 吞吐量
    *_W / *_J     RAPL 平均功率 / 能量, 以及 J/token (可讀時)
    npu_util      NPU 使用率 (可讀時)

準備模型 (NPU 建議 int4 對稱, channel-wise 量化):
    pip install -r bench/requirements-bench.txt
    optimum-cli export openvino -m Qwen/Qwen2.5-1.5B-Instruct \
        --weight-format int4 --sym --group-size -1 --ratio 1.0 ov_models/qwen2.5-1.5b-int4

用法 (在 repo 根目錄, 先跑過 python rag.py index):
    python bench/bench_llm.py --model ov_models/qwen2.5-1.5b-int4
    python bench/bench_llm.py --model ov_models/qwen2.5-1.5b-int4 --devices NPU --max-new-tokens 256
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import rag  # noqa: E402
from power import PowerMeter, idle_baseline  # noqa: E402

DEFAULT_QUESTIONS = [
    "周杰倫的出道專輯是哪一張? 是在哪一年發行的?",
    "鄧麗君在哪一年過世? 過世地點是哪裡?",
    "陳綺貞的出道專輯名稱是什麼?",
]


def build_prompts(questions):
    prompts = []
    for q in questions:
        hits = rag.retrieve(q)
        prompts.append(rag.build_prompt(q, hits))
    return prompts


def make_pipeline(model_dir, device, max_prompt_len, min_response_len, use_cache):
    import openvino_genai as ov_genai
    config = {}
    if use_cache:
        cache = ROOT / "ov_cache"
        cache.mkdir(exist_ok=True)
        config["CACHE_DIR"] = str(cache)
    if device == "NPU":
        # NPU 是靜態形狀, 要先告訴它 prompt 與回答的最大長度
        config["MAX_PROMPT_LEN"] = max_prompt_len
        config["MIN_RESPONSE_LEN"] = min_response_len
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(model_dir), device, **config)
    return pipe, time.perf_counter() - t0


def bench_device(device, args, prompts, meter):
    import openvino_genai as ov_genai

    row = {"device": device, "model": Path(args.model).name, "max_new_tokens": args.max_new_tokens}
    pipe, row["load_s"] = make_pipeline(args.model, device, args.max_prompt_len,
                                        args.max_new_tokens, not args.no_cache)
    gen = ov_genai.GenerationConfig()
    gen.max_new_tokens = args.max_new_tokens
    gen.do_sample = False  # greedy, 讓每個裝置輸出一樣長度

    pipe.generate([prompts[0]], gen)  # 暖機

    ttft, tpot, tps, in_tok, out_tok = [], [], [], 0, 0
    meter.start()
    for r in range(args.repeat):
        for p in prompts:
            res = pipe.generate([p], gen)
            m = res.perf_metrics
            ttft.append(m.get_ttft().mean)
            tpot.append(m.get_tpot().mean)
            tps.append(m.get_throughput().mean)
            in_tok += m.get_num_input_tokens()
            out_tok += m.get_num_generated_tokens()
    p = meter.stop()

    row.update({
        "ttft_ms": sum(ttft) / len(ttft),
        "tpot_ms": sum(tpot) / len(tpot),
        "tok_per_s": sum(tps) / len(tps),
        "avg_input_tokens": in_tok / len(ttft),
        "avg_output_tokens": out_tok / len(ttft),
    })
    for k, v in p.items():
        if k != "seconds" and v is not None:
            row[k] = v
    pkg = next((k for k in p if k.endswith("_J") and "package" in k), None)
    if pkg and out_tok:
        row["J_per_output_token"] = p[pkg] / out_tok
    row["sample_answer"] = res.texts[0][:200]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="OpenVINO 格式的 LLM 資料夾 (optimum-cli 匯出)")
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-prompt-len", type=int, default=2048, help="NPU 用: prompt 最大 token 數")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--idle-seconds", type=float, default=5.0)
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meter = PowerMeter()
    print("功耗計數器:", meter.available)
    prompts = build_prompts(DEFAULT_QUESTIONS)

    meta = {"time": datetime.now().isoformat(timespec="seconds")}
    if args.idle_seconds > 0 and meter.domains:
        meta["idle"] = idle_baseline(args.idle_seconds)

    rows = []
    for dev in args.devices:
        print(f"\n== {dev} ==")
        try:
            row = bench_device(dev, args, prompts, meter)
        except Exception as e:
            print(f"  失敗: {type(e).__name__}: {e}")
            continue
        row.update(meta)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "idle"})
        time.sleep(args.cooldown)

    if not rows:
        sys.exit("沒有任何裝置跑成功")
    print(f"\n{'device':>6} | {'load_s':>7} | {'ttft_ms':>8} | {'tpot_ms':>8} | {'tok/s':>6} | {'J/token':>8}")
    for r in rows:
        jt = r.get("J_per_output_token")
        print(f"{r['device']:>6} | {r['load_s']:7.1f} | {r['ttft_ms']:8.0f} | {r['tpot_ms']:8.1f} | "
              f"{r['tok_per_s']:6.1f} | {jt if jt is None else round(jt, 3)!s:>8}")

    out = args.out or ROOT / "bench" / "results" / f"llm_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n結果已存到 {out}")


if __name__ == "__main__":
    main()
