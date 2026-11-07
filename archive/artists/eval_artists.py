"""中文題庫: 20 位華語藝人維基條目, 20 題 (第 1 週, 2026-10-22).

原本放在 eval.py 的 EVAL_SET, 2026-11-07 移到這裡封存.
欄位意義與 eval.py 相同. 要重跑請見 archive/README.md.
"""

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
