# Linux 隔離機制簡介

Linux 容器的隔離主要由兩個核心機制組成：namespace 與 cgroups。

namespace 負責「看見什麼」。每種 namespace 隔離一類系統資源，例如 PID namespace 讓行程只看得到自己群組內的行程，mount namespace 讓檔案系統掛載點彼此獨立，network namespace 提供獨立的網路堆疊，user namespace 則能把容器內的 root 對應到主機上的一般使用者。

cgroups（control groups）負責「能用多少」。它可以限制並統計一組行程使用的 CPU、記憶體、I/O 與行程數量。當行程超出記憶體上限時，系統可能觸發 OOM killer 終止它。

seccomp 則進一步限制行程可以呼叫哪些系統呼叫（syscall），常用於縮小被攻破後的影響範圍。

在設計 AI Agent 的程式碼執行沙箱時，這三者通常搭配使用：用 namespace 隔離視野、用 cgroups 限制資源、用 seccomp 限制系統呼叫，並且以最小權限執行。
