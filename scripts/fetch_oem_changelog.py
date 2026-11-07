"""下載 Ubuntu OEM kernel 的 debian changelog, 每個版本切成一個檔案, 給 rag.py 當資料集.

預設來源: anthonywong/ubuntu-oem-kernel-mirror 的 oem-7.0-next 分支 debian.oem/changelog
預設範圍: linux-oem-7.0 的 7.0.0-1001.1 到 7.0.0-1018.18 (用 ABI 號 1001..1018 判斷)

用法:
    python scripts/fetch_oem_changelog.py                       # 輸出到 data/oem-7.0/
    python scripts/fetch_oem_changelog.py --min 1010 --max 1018
    python scripts/fetch_oem_changelog.py --file changelog      # 用本機已下載的檔案
    python rag.py index                                         # data/oem-7.0 是預設資料集

輸出檔名: linux-oem-7.0_7.0.0-1018.18.md, 內容 = 標頭 (版本, 發行日期, 來源) + 該版本的 changelog 本文.
維護者 email 那一行 (" -- 名字 <email>  日期") 不保留, 只留日期.
"""
import argparse
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DEFAULT_URL = ("https://raw.githubusercontent.com/anthonywong/ubuntu-oem-kernel-mirror/"
               "oem-7.0-next/debian.oem/changelog")

# Debian changelog 每個版本的開頭: "linux-oem-7.0 (7.0.0-1018.18) resolute; urgency=medium"
HEADER_RE = re.compile(r"^(?P<pkg>[a-z0-9][a-z0-9.+-]*) \((?P<ver>[^)]+)\) (?P<dist>[^;]+);", re.M)
# 結尾: " -- Maintainer Name <email>  Fri, 31 Oct 2026 10:00:00 +0800"
TRAILER_RE = re.compile(r"^ -- .*?>\s+(?P<date>.+)$", re.M)


def read_changelog(args):
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    print(f"下載 {args.url}")
    req = urllib.request.Request(args.url, headers={"User-Agent": "rag_minimal-changelog-fetcher"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def split_entries(text):
    """回傳 [(pkg, version, dist, body, date)], body 不含標頭與結尾行."""
    heads = list(HEADER_RE.finditer(text))
    entries = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        block = text[m.end():end]
        t = TRAILER_RE.search(block)
        date = t.group("date").strip() if t else ""
        body = block[:t.start()] if t else block
        # 去掉標頭那行剩下的 "urgency=medium"
        body = body.split("\n", 1)[1] if "\n" in body else ""
        entries.append((m["pkg"], m["ver"], m["dist"].strip(), body.strip("\n"), date))
    return entries


def abi_of(version):
    """7.0.0-1018.18 -> 1018"""
    m = re.match(r"[\d.]+-(\d+)\.", version)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--file", help="改讀本機 changelog 檔, 不下載")
    ap.add_argument("--package", default="linux-oem-7.0")
    ap.add_argument("--min", type=int, default=1001, help="最小 ABI 號 (含)")
    ap.add_argument("--max", type=int, default=1018, help="最大 ABI 號 (含)")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "oem-7.0")
    args = ap.parse_args()

    entries = split_entries(read_changelog(args))
    picked = [e for e in entries
              if e[0] == args.package and (a := abi_of(e[1])) is not None and args.min <= a <= args.max]
    if not picked:
        sys.exit(f"找不到 {args.package} ABI {args.min}..{args.max} 的版本")

    args.out.mkdir(parents=True, exist_ok=True)
    source = args.file or args.url
    for pkg, ver, dist, body, date in picked:
        path = args.out / f"{pkg}_{ver}.md"
        path.write_text(
            f"# {pkg} {ver}\n\n"
            f"- 套件: {pkg}\n- 版本: {ver}\n- 發行版: {dist}\n- 日期: {date}\n- 來源: {source}\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        print(f"{path.name:40s} {len(body.splitlines()):6d} 行  {date}")
    print(f"\n共 {len(picked)} 個版本 -> {args.out}")
    print(f"建索引: DATA_DIR={args.out.relative_to(HERE) if args.out.is_relative_to(HERE) else args.out} "
          f"python rag.py index")


if __name__ == "__main__":
    main()
