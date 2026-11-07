"""現場 demo: 問 kernel changelog 問題, 同一個本機小模型依序跑在 GPU 與 NPU, 比較速度與功耗.

畫面流程 (每個問題):
    1. 檢索: 顯示 BM25 找到的 changelog 段落 (「先 grep」)
    2. 生成: 同一個 prompt 依序在每個裝置上跑, 回答一個字一個字流出來
       每個裝置跑完顯示: 第一個字的時間 (TTFT), tokens/s, SoC 功率, 比閒置多用的能量, NPU 使用率
    3. 總結表: 各裝置並排比較
    (可選) --claude: 再用雲端 Claude 回答一次, 當作品質對照 (需要 .env 的 ANTHROPIC_API_KEY)

為什麼依序跑而不是同時跑: GPU 與 NPU 共用記憶體頻寬與 SoC 功耗預算 (RAPL package 量的是整顆 SoC),
同時跑會互相干擾, 功耗也分不出是誰的.

準備 (與 bench/bench_llm.py 相同的模型):
    pip install -r bench/requirements-bench.txt
    optimum-cli export openvino -m Qwen/Qwen2.5-1.5B-Instruct \
        --weight-format int4 --sym --group-size -1 --ratio 1.0 ov_models/qwen2.5-1.5b-int4

用法:
    python demo.py --model ov_models/qwen2.5-1.5b-int4                 # 互動模式, GPU 與 NPU
    python demo.py --model ov_models/qwen2.5-1.5b-int4 --claude        # 加上雲端 Claude 對照
    python demo.py --model ov_models/qwen2.5-1.5b-int4 -q "Which CVEs were fixed in 7.0.0-1017.17?"
    sudo -E .venv/bin/python demo.py --model ...                       # 要看功耗 (RAPL 需 root, 見 README)

互動模式: 輸入問題後按 Enter; 空白行跳過, 輸入 q 離開. 生成中按 Ctrl-C 只會中斷目前這個裝置.
"""
import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "bench"))

import rag  # noqa: E402
from ownership import give_back  # noqa: E402
from power import PowerMeter, idle_baseline  # noqa: E402

# 現場備用題 (2026-11-07 確認 BM25 檢索得到正確段落): 觀眾沒提問時用
SUGGESTED = [
    "Which kernel version fixed the upside-down front camera image?",       # 答案 7.0.0-1018.18
    "Which version added a bounds check for ivpu firmware runtime memory?",  # 答案 7.0.0-1012.12
    "What accel/ivpu (Intel NPU) driver changes were made?",                 # 1012.12 與 1017.17 都有
]
# 刻意展示檢索失敗用 (三個模型都會回答找不到):
# - "What changes were made to the intel_vpu / NPU driver?": changelog 寫的是 accel/ivpu, 字對不上,
#   BM25 反而找到 mt76 Wi-Fi 的 "NPU" 項目
# - "Which CVEs were fixed in 7.0.0-1017.17?": 版本號只出現在檔案開頭那段, CVE 項目的段落沒有版本號,
#   而且 564 個 CVE 也不是 4 段資料列得完的 (這種「列出全部」的問題用 grep 比較適合)

# 終端機顏色 (分享畫面時比較好讀); NO_COLOR=1 或輸出不是終端機時關閉
_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t):
    return c(t, "1")


def dim(t):
    return c(t, "2")


def cyan(t):
    return c(t, "36")


def yellow(t):
    return c(t, "33")


def green(t):
    return c(t, "32")


def header(title):
    print("\n" + bold(cyan(f"━━ {title} " + "━" * max(0, 60 - len(title)))))


# ---------- 檢索 ----------
def load_records_or_chunk():
    """BM25 只需要 chunks; 沒建過索引時直接切塊 (不用載 embedding 模型), demo 啟動比較快."""
    if rag.CHUNKS_FILE.exists():
        return rag.load_records()
    records = []
    for name, text in rag.load_docs():
        title = rag.doc_title(name, text)
        for i, ch in enumerate(rag.chunk_text(text)):
            if rag.CHUNK_TITLE:
                ch = f"[{title}]\n{ch}"
            records.append({"source": name, "chunk_id": i, "text": ch})
    if not records:
        sys.exit(f"{rag.DATA_DIR} 內沒有資料, 請先執行: python scripts/fetch_oem_changelog.py")
    return records


def retrieve(question, mode, records):
    if mode == "bm25":
        scores = rag.get_bm25(records).scores(question)
        top = scores.argsort()[::-1][:rag.TOP_K]
        return [(float(scores[i]), records[i]) for i in top]
    return rag.retrieve(question, mode=mode)  # vector / hybrid 需要先 python rag.py index


def show_hits(hits, mode, ms):
    header(f"Step 1  檢索 ({mode}, {ms:.0f} ms)")
    for n, (score, r) in enumerate(hits, 1):
        body = " ".join(r["text"].split())
        preview = body[:110] + ("…" if len(body) > 110 else "")
        meta = f"#{r['chunk_id']} score={score:.2f}"
        print(f"{bold(f'[{n}]')} {yellow(r['source'])} {dim(meta)}")
        print(f"    {preview}")


# ---------- 本機 LLM (OpenVINO GenAI) ----------
# 1.5B 的小模型對 prompt 很敏感, 所以本機模型用一份專門的 prompt (rag.build_prompt 留給 Claude):
# - 英文指示: changelog 是英文, 中英混雜的指示小模型比較容易跟不上
# - 自己的 system message: 不然 Qwen 的 chat template 會塞入預設的 "You are Qwen, created by Alibaba Cloud...",
#   小模型有時會把它當成內容講出來 (例如回答 "Qwen, the Qwen, ...")
# - 每段資料前直接寫版本號: 原本版本只藏在檔名裡, 小模型常抓錯是哪一版
LOCAL_SYSTEM = (
    "You answer questions about Ubuntu linux-oem-7.0 kernel changelogs. "
    "Use ONLY the numbered excerpts. Each excerpt starts with the kernel version it belongs to. "
    "Answer in 1 to 3 short sentences: give the kernel version and cite the excerpt like [1]. "
    "If the excerpts do not answer the question, reply exactly: Not found in the changelog."
)
BEST_LAST = True  # main() 會依 --best-first 改掉
_VERSION = re.compile(r"\d+\.\d+\.\d+-\d+\.\d+")


def build_local_prompt(question, hits):
    parts = []
    order = list(enumerate(hits, 1))
    if BEST_LAST:
        # 小模型常偏向最靠近問題的那段 (實測兩次都答成排在最後的 [4]), 把分數最高的放最後, 編號不變
        order.reverse()
    for n, (_, r) in order:
        m = _VERSION.search(r["source"])
        version = m.group(0) if m else r["source"]
        parts.append(f"[{n}] kernel version {version}:\n{' '.join(r['text'].split())}")
    user = "Excerpts:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"
    return [{"role": "system", "content": LOCAL_SYSTEM}, {"role": "user", "content": user}]


def to_prompt_text(pipe, messages):
    """自己套 chat template (帶上我們的 system message), 之後 generate 時就不要再套一次."""
    return pipe.get_tokenizer().apply_chat_template(messages, add_generation_prompt=True)


class Interrupt:
    """生成中按 Ctrl-C: 只停掉目前這個裝置, 不結束整個 demo."""

    def __init__(self):
        self.flag = False

    def __enter__(self):
        self.flag = False
        self._old = signal.signal(signal.SIGINT, self._handler)
        return self

    def _handler(self, *_):
        self.flag = True

    def __exit__(self, *_):
        signal.signal(signal.SIGINT, self._old)


def load_pipelines(args):
    import openvino_genai as ov_genai

    pipes = {}
    cache = ROOT / "ov_cache"
    cache.mkdir(exist_ok=True)
    for dev in args.devices:
        config = {"CACHE_DIR": str(cache)}
        if dev == "NPU":
            # NPU 是靜態形狀, 要先告訴它 prompt 與回答的最大長度
            config["MAX_PROMPT_LEN"] = args.max_prompt_len
            config["MIN_RESPONSE_LEN"] = args.max_new_tokens
        print(f"載入 {dev} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            pipe = ov_genai.LLMPipeline(str(args.model), dev, **config)
            # 暖機: 第一次 generate 會比較慢 (配置記憶體, 編譯 kernel), 先在開場前跑掉
            pipe.generate("Hello", max_new_tokens=8)
        except Exception as e:  # noqa: BLE001  現場不要因為一個裝置失敗就整個掛掉
            print(c(f"失敗, 略過: {type(e).__name__}: {e}", "31"))
            continue
        pipes[dev] = pipe
        print(green(f"OK ({time.perf_counter() - t0:.1f}s)"))
    if not pipes:
        sys.exit("沒有任何裝置載入成功")
    return pipes


def _streaming_status(stop):
    """openvino_genai 新版用 StreamingStatus, 舊版回傳 bool (True = 停止)."""
    import openvino_genai as ov_genai
    status = getattr(ov_genai, "StreamingStatus", None)
    if status is None:
        return stop
    return status.STOP if stop else status.RUNNING


def run_device(dev, pipe, prompt, args, idle_w):
    import openvino_genai as ov_genai

    gen = ov_genai.GenerationConfig()
    gen.max_new_tokens = args.max_new_tokens
    gen.do_sample = False  # greedy: 同一題每次回答一樣, 現場比較可預期
    # greedy 加上小模型很容易陷入重複迴圈 ("[7.0] is the kernel version ..." 一直重複), 加一點重複懲罰
    gen.repetition_penalty = args.repetition_penalty
    if hasattr(gen, "apply_chat_template"):
        gen.apply_chat_template = False  # prompt 已經套過 chat template
    # 小模型答完常會接著亂寫: 自己編下一題 ("Question: ..."), 列一串 "[1] [2] ... [28]",
    # 或把資料原文 "[1] kernel version ..." 抄一遍; 這些都是換行後接 "[" 或 "Question:", 看到就停
    gen.stop_strings = {"Question:", "\n["}

    header(f"Step 2  {dev} 回答")
    meter = PowerMeter()
    first = [None]
    with Interrupt() as intr:
        def streamer(sub):
            if first[0] is None:
                first[0] = time.perf_counter()
            print(sub, end="", flush=True)
            return _streaming_status(intr.flag)

        meter.start()
        t0 = time.perf_counter()
        try:
            res = pipe.generate([prompt], gen, streamer)  # 傳 list 才會回傳含 perf_metrics 的結果, 傳 str 只回傳字串
        except RuntimeError as e:
            # 萬一某個裝置 (例如較舊版的 NPU pipeline) 不支援 repetition_penalty / stop_strings, 就不加再跑一次
            if gen.repetition_penalty == 1.0 and not gen.stop_strings:
                raise
            print(dim(f"({dev} 不支援 repetition_penalty 或 stop_strings, 改用預設值: {e})"))
            gen.repetition_penalty = 1.0
            gen.stop_strings = set()
            first[0] = None
            t0 = time.perf_counter()
            res = pipe.generate([prompt], gen, streamer)  # 傳 list 才會回傳含 perf_metrics 的結果, 傳 str 只回傳字串
        t_end = time.perf_counter()
        p = meter.stop()
        stopped = intr.flag
    print(dim("  (Ctrl-C 中斷)") if stopped else "")

    m = res.perf_metrics
    out_tok = m.get_num_generated_tokens()
    row = {
        "device": dev,
        "ttft_s": (first[0] or t_end) - t0,
        "total_s": t_end - t0,
        "tok_per_s": m.get_throughput().mean,
        "out_tokens": out_tok,
    }
    pkg = next((k for k in p if k.endswith("_W") and "package" in k), None)
    if pkg and p.get(pkg) is not None:
        row["W"] = p[pkg]
        if idle_w is not None:
            row["J_above_idle"] = max(0.0, p[pkg] - idle_w) * p["seconds"]
    if p.get("npu_util") is not None:
        row["npu_util"] = p["npu_util"]

    parts = [f"首字 {row['ttft_s']:.2f}s", f"{row['tok_per_s']:.1f} tok/s",
             f"共 {out_tok} tokens / {row['total_s']:.1f}s"]
    if "W" in row:
        parts.append(f"SoC {row['W']:.1f} W")
    if "J_above_idle" in row:
        parts.append(f"比閒置多 {row['J_above_idle']:.0f} J")
    if "npu_util" in row:
        parts.append(f"NPU 使用率 {row['npu_util'] * 100:.0f}%")
    print(bold(green("→ " + " | ".join(parts))))
    return row


def run_claude(prompt):
    header("對照  雲端 Claude 回答")
    t0 = time.perf_counter()
    try:
        answer, _, tout = rag.generate(prompt)
    except BaseException as e:  # noqa: BLE001  沒金鑰或沒網路: 現場略過即可 (rag.generate 可能 sys.exit)
        print(c(f"略過: {type(e).__name__}: {e}", "31"))
        return
    print(answer)
    print(bold(green(f"→ 總時間 {time.perf_counter() - t0:.1f}s | {tout} tokens | 功耗在雲端, 本機量不到")))


def summary(rows):
    if len(rows) < 2:
        return
    header("Step 3  比較")
    has_w = any("W" in r for r in rows)
    has_j = any("J_above_idle" in r for r in rows)
    # 表頭用英文: 中文字在終端機佔兩格, 會讓欄位對不齊
    cols = f"{'device':<8}{'TTFT(s)':>9}{'tok/s':>8}{'total(s)':>11}"
    cols += f"{'SoC(W)':>8}" if has_w else ""
    cols += f"{'J>idle':>9}" if has_j else ""
    print(bold(cols))
    for r in rows:
        line = f"{r['device']:<8}{r['ttft_s']:>9.2f}{r['tok_per_s']:>8.1f}{r['total_s']:>11.1f}"
        if has_w:
            line += f"{r.get('W', float('nan')):>8.1f}"
        if has_j:
            line += f"{r.get('J_above_idle', float('nan')):>9.0f}"
        print(line)


def answer(question, args, pipes, records, idle_w):
    t0 = time.perf_counter()
    hits = retrieve(question, args.retrieval, records)
    show_hits(hits, args.retrieval, (time.perf_counter() - t0) * 1000)
    messages = build_local_prompt(question, hits)

    rows = []
    for i, (dev, pipe) in enumerate(pipes.items()):
        if i and args.cooldown > 0:
            time.sleep(args.cooldown)  # 讓功耗回落, 下一個裝置的數字比較乾淨
        rows.append(run_device(dev, pipe, to_prompt_text(pipe, messages), args, idle_w))
    summary(rows)
    if args.claude:
        run_claude(rag.build_prompt(question, hits))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="OpenVINO 格式的 LLM 資料夾 (optimum-cli 匯出)")
    ap.add_argument("--devices", nargs="+", default=["GPU", "NPU"], help="依序跑的裝置, 預設 GPU NPU")
    ap.add_argument("--retrieval", default="bm25", choices=["bm25", "vector", "hybrid"],
                    help="檢索方式, 預設 bm25 (oem-7.0 上 hit@3 最高, 也不用載 embedding 模型)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--repetition-penalty", type=float, default=1.1,
                    help="大於 1 會壓低重複的字, 避免小模型鬼打牆; 1.0 = 不懲罰")
    ap.add_argument("--max-prompt-len", type=int, default=2048, help="NPU 用: prompt 最大 token 數")
    ap.add_argument("--cooldown", type=float, default=2.0, help="裝置之間的間隔秒數")
    ap.add_argument("--idle-seconds", type=float, default=3.0, help="開場量閒置功耗的秒數, 0 = 不量")
    ap.add_argument("--best-first", action="store_true",
                    help="本機模型的資料依分數由高到低排 (預設反過來, 最相關的放在最靠近問題的位置)")
    ap.add_argument("--claude", action="store_true", help="最後用雲端 Claude 回答一次當對照")
    ap.add_argument("-q", "--question", action="append", help="直接問這題 (可重複), 不進互動模式")
    args = ap.parse_args()
    global BEST_LAST
    BEST_LAST = not args.best_first

    records = load_records_or_chunk() if args.retrieval == "bm25" else None
    if records is not None:
        rag.get_bm25(records)  # 先建好 BM25, 第一題不用等
    pipes = load_pipelines(args)

    idle_w = None
    meter = PowerMeter()
    if args.idle_seconds > 0 and meter.domains:
        idle = idle_baseline(args.idle_seconds)
        idle_w = next((v for k, v in idle.items() if k.endswith("_W") and "package" in k), None)
    if idle_w is not None:
        print(dim(f"閒置功耗 {idle_w:.1f} W"))
    else:
        print(dim("讀不到 RAPL 功耗 (要看功耗請用 sudo -E 執行, 見 README), 只顯示速度"))

    try:
        if args.question:
            for q in args.question:
                answer(q, args, pipes, records, idle_w)
            return
        print(bold("\n準備好了. 輸入 changelog 問題 (q 離開). 備用題:"))
        for s in SUGGESTED:
            print(dim(f"  - {s}"))
        while True:
            try:
                q = input(bold(cyan("\n問題> "))).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in {"q", "quit", "exit"}:
                break
            if q:
                answer(q, args, pipes, records, idle_w)
    finally:
        give_back(ROOT / "ov_cache")


if __name__ == "__main__":
    main()
