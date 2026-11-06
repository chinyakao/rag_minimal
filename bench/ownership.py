"""用 sudo 跑 benchmark 時, 把產生的檔案還給原本的使用者.

sudo 執行時新建的 bench/results/, ov_models/, ov_cache/ 會變成 root 所有,
之後不加 sudo 再跑就會 PermissionError. sudo 會留下 SUDO_UID / SUDO_GID,
這裡用它們把擁有者改回來; 不是用 sudo 跑時什麼都不做.
"""
import os
from pathlib import Path


def give_back(*paths):
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() != 0 or not uid:
        return
    uid, gid = int(uid), int(gid or uid)
    for p in map(Path, paths):
        if not p.exists():
            continue
        targets = [p] + (list(p.rglob("*")) if p.is_dir() else [])
        for t in targets:
            try:
                os.chown(t, uid, gid, follow_symlinks=False)
            except OSError:
                pass
