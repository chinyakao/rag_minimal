"""簡易檢索評估: 藝人文件集 20 題

用法:
    python eval.py            # 只評估檢索 (不呼叫 LLM, 不花 API 費用), k=3
    python eval.py 5          # 改用 k=5
    python eval.py --llm      # 另外呼叫 LLM, 檢查 refuse 類題目是否拒答

欄位說明:
    type             題型 (single / multi_para / cross / reasoning / boundary)
    expected_sources 應出現在檢索結果中的文件 (可多個). 空 list 表示不做 hit 判斷
    expect           answer = 資料中有答案; refuse = 資料中找不到, 應拒答;
                     manual = 需人工看回答 (例如全庫統計題, 或資料只有部分資訊)
    note             備註, 只在輸出中顯示
"""
import sys

from rag import retrieve

REFUSAL_TEXT = "資料中找不到答案"

EVAL_SET = [
    # ---- 單一條目事實查詢 ----
    {"id": 1, "type": "single", "expect": "answer",
     "q": "周杰倫的出道專輯是哪一張? 是在哪一年發行的?",
     "expected_sources": ["周杰倫.md"]},
    {"id": 2, "type": "single", "expect": "answer",
     "q": "鄧麗君在哪一年過世? 過世地點是哪裡?",
     "expected_sources": ["鄧麗君.md"]},
    {"id": 3, "type": "single", "expect": "answer",
     "q": "陳綺貞的出道專輯名稱是什麼?",
     "expected_sources": ["陳綺貞.md"]},
    {"id": 4, "type": "single", "expect": "answer",
     "q": "林志玲是在哪一年, 以什麼身分進入演藝圈的?",
     "expected_sources": ["林志玲.md"]},
    {"id": 5, "type": "single", "expect": "answer",
     "q": "盧廣仲有哪些作品拿過金曲獎?",
     "expected_sources": ["盧廣仲.md"]},

    # ---- 需要整合條目多個段落 ----
    {"id": 6, "type": "multi_para", "expect": "answer",
     "q": "蔡依林的音樂風格在職業生涯中經歷了哪些轉變?",
     "expected_sources": ["蔡依林.md"]},
    {"id": 7, "type": "multi_para", "expect": "answer",
     "q": "李宗盛除了歌手身分, 還擔任過哪些幕後角色?",
     "expected_sources": ["李宗盛.md"]},
    {"id": 8, "type": "multi_para", "expect": "answer",
     "q": "張惠妹的出道經過是什麼? 她早期的代表作有哪些?",
     "expected_sources": ["張惠妹.md"]},
    {"id": 9, "type": "multi_para", "expect": "answer",
     "q": "劉若英從演員跨足歌手的過程是怎樣的?",
     "expected_sources": ["刘若英.md"],
     "note": "檔名是簡體 刘若英.md"},
    {"id": 10, "type": "multi_para", "expect": "answer",
     "q": "羅大佑的作品對華語流行音樂產生了什麼影響?",
     "expected_sources": ["羅大佑.md"]},

    # ---- 跨條目比較與統整 ----
    {"id": 11, "type": "cross", "expect": "answer",
     "q": "周杰倫與蔡依林各自得過哪些重要獎項? 兩人的獲獎情況有什麼不同?",
     "expected_sources": ["周杰倫.md", "蔡依林.md"]},
    {"id": 12, "type": "cross", "expect": "manual",
     "q": "這 20 位藝人中, 誰同時有歌手與演員兩種身分? 請列出並簡述各自代表作品.",
     "expected_sources": [],
     "note": "全庫統計題, 沒有固定的正確文件; 須人工檢查回答是否完整"},
    {"id": 13, "type": "cross", "expect": "answer",
     "q": "田馥甄, 楊丞琳, 王心凌三人的出身背景 (出道方式或團體) 有何異同?",
     "expected_sources": ["田馥甄.md", "楊丞琳.md", "王心凌.md"]},
    {"id": 14, "type": "cross", "expect": "manual",
     "q": "哪些藝人曾經在金曲獎獲得最佳演唱類獎項?",
     "expected_sources": [],
     "note": "全庫統計題, 須人工檢查回答是否完整"},
    {"id": 15, "type": "cross", "expect": "answer",
     "q": "伍佰與羅大佑的音樂風格有什麼差別?",
     "expected_sources": ["伍佰.md", "羅大佑.md"]},

    # ---- 多步推理 ----
    {"id": 16, "type": "reasoning", "expect": "answer",
     "q": "金城武出生於哪個地方? 他主要在哪些地區發展演藝事業?",
     "expected_sources": ["金城武.md"]},
    {"id": 17, "type": "reasoning", "expect": "answer",
     "q": "林依晨與周渝民分別以哪部戲劇走紅? 這兩位是否合作過?",
     "expected_sources": ["林依晨.md", "周渝民.md"]},
    {"id": 18, "type": "reasoning", "expect": "manual",
     "q": "哪位藝人是 20 人中出道最早的? 依據是什麼?",
     "expected_sources": [],
     "note": "全庫比較題, 須人工比對各藝人出道年份"},

    # ---- 檢索邊界測試 ----
    {"id": 19, "type": "boundary", "expect": "manual",
     "q": "蕭敬騰目前的經紀公司是哪一家?",
     "expected_sources": ["蕭敬騰.md"],
     "note": "文件只提到 2008 簽華納唱片, 2014 經紀公司喜鵲娛樂, 沒有說目前是哪家; "
             "好的回答應說明資料只有過去資訊, 不應直接斷言目前"},
    {"id": 20, "type": "boundary", "expect": "refuse",
     "q": "張韶涵與羅志祥有沒有共同合作過的作品?",
     "expected_sources": [],
     "note": "兩人文件互相沒有提到對方, 應拒答"},
]


def run(k=3, use_llm=False):
    hit_total = hit_ok = 0
    by_type = {}
    refuse_total = refuse_ok = 0

    for item in EVAL_SET:
        results = retrieve(item["q"], k=k)
        sources = [r["source"] for _, r in results]
        top1 = results[0][0]
        exp = item["expected_sources"]
        tag = f"#{item['id']:<2} [{item['type']}]"

        if exp:
            found = [s for s in exp if s in sources]
            ok = len(found) > 0
            recall = len(found) / len(exp)
            hit_total += 1
            hit_ok += ok
            t = by_type.setdefault(item["type"], [0, 0])
            t[0] += 1
            t[1] += ok
            status = "OK  " if ok else "MISS"
            extra = f" recall={len(found)}/{len(exp)}" if len(exp) > 1 else ""
            print(f"{status} {tag}{extra} {item['q']}")
            print(f"       -> {sources} (top1={top1:.3f})")
        else:
            print(f"---- {tag} {item['q']}")
            print(f"       -> {sources} (top1={top1:.3f})")

        if item.get("note"):
            print(f"       note: {item['note']}")

        if use_llm and item["expect"] in ("refuse", "manual"):
            from rag import build_prompt, generate
            answer, _, _ = generate(build_prompt(item["q"], results))
            if item["expect"] == "refuse":
                refused = REFUSAL_TEXT in answer
                refuse_total += 1
                refuse_ok += refused
                print(f"       LLM: {'REFUSED' if refused else 'ANSWERED (可能幻覺)'}")
            else:
                print(f"       LLM 回答 (請人工檢查): {answer[:200]}...")

    print(f"\nhit@{k} = {hit_ok}/{hit_total} = {hit_ok / hit_total:.2f}")
    for name, (n, ok) in by_type.items():
        print(f"  {name:<10} {ok}/{n}")
    if use_llm and refuse_total:
        print(f"拒答率 = {refuse_ok}/{refuse_total}")


if __name__ == "__main__":
    args = sys.argv[1:]
    k = next((int(a) for a in args if a.isdigit()), 3)
    run(k=k, use_llm="--llm" in args)
