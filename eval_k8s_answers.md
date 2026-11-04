# eval_k8s 答案對照表與 k8s 入門

這份給你核對 `eval_k8s.py` 用。每題都附原文出處和逐字引文，用「找字」就能核對，不需要先懂 k8s。

## 一、eval 怎麼判定對錯

- **`python eval.py --k8s`（hit@k）完全自動，不用你判斷答案。** 它只檢查檢索到的前 k 個 chunk 裡，有沒有來自 `expected_sources` 那個檔案的 chunk。題目的正確性我已經對照原文確認過，就是下面這張表。
- **只有加 `--llm` 時才需要人工看。** 而且只看 `expect` 是 `manual` 或 `refuse` 的題目（第 19、20 題）：
  - 第 19 題：回答說「文件沒寫 MIG／time-slicing 怎麼設定」就算對。如果講出一套設定步驟，就是模型自己編的。
  - 第 20 題：回答拒答就算對。如果回答出一個節點數字，那是模型憑記憶答的，算錯。
- 想看模型回答得好不好，拿回答跟下表的「原文」比對：關鍵字有對上就算對。

## 二、k8s 入門：只講這 20 題會用到的

用 Linux 來類比。Kubernetes（k8s）就是「把很多台 Linux 機器當成一台來排程程式」的系統。

| 名詞 | 白話 | Linux 類比 |
| --- | --- | --- |
| Node | 叢集裡的一台機器 | 一台 Linux 主機 |
| Pod | 最小的部署單位，裡面是一到多個容器 | 一組共用 namespace 與 cgroups 的行程 |
| kubelet | 每台 Node 上的代理程式，負責啟動 Pod、回報狀態 | 類似每台機器上的 systemd |
| kube-scheduler | 決定 Pod 要放到哪台 Node | 排程器 |
| Deployment／ReplicaSet | 維持「固定數量的 Pod 一直在跑」；Deployment 管理 ReplicaSet，並負責滾動更新 | 會自動重啟、自動補數量的服務 |
| DaemonSet | 每台 Node 都跑一份 | 每台機器都裝的系統常駐程式（例如 log 收集） |
| StatefulSet | 有固定名字和固定儲存的 Pod，例如資料庫 | 固定 hostname，並綁定自己磁碟的服務 |
| Job | 跑完就結束的批次工作，失敗會重試 | 批次腳本 |
| Service | 給一組 Pod 一個固定的存取入口與負載平衡 | 固定的虛擬 IP 加負載平衡器 |
| requests／limits | requests 是排程時的保留量，limits 是上限 | cgroups 的資源限制；超過記憶體上限就被 OOM kill |
| Eviction | Node 資源不足時，kubelet 主動趕走某些 Pod | 記憶體不足時挑行程砍掉 |
| Taint／Toleration | Node 設 taint 排斥 Pod，Pod 有對應的 toleration 才能放上去 | 「這台機器只給特定程式用」 |
| Node affinity | Pod 偏好或要求放在某類 Node | 「這個程式要跑在有 SSD 的機器」 |
| ResourceQuota／LimitRange | 前者限制整個 namespace 的總量，後者限制單一 Pod 或容器的範圍 | 整個部門的配額，以及每個人的上下限 |
| ConfigMap／Secret | 設定檔與機密資料 | /etc 下的設定檔，以及權限 600 的密碼檔 |
| Device plugin | 把 GPU 等硬體回報給 k8s，讓 Pod 可以申請 | GPU 驅動，加上向排程器回報「我有幾張卡」 |

## 三、20 題答案對照

| # | 題型 | 出處檔案 | 原文（可直接搜尋） | 白話答案 |
| --- | --- | --- | --- | --- |
| 1 | single | configmap.md | "ConfigMap cannot exceed 1 MiB." | 上限 1 MiB |
| 2 | single | pod-lifecycle.md | "The default value is Always." | 預設 Always（容器結束一律重啟） |
| 3 | single | daemonset.md | "A DaemonSet ensures that all (or some) Nodes run a copy of a Pod." | 每台（或指定的）Node 都跑一份 |
| 4 | single | device-plugins.md | "an NVIDIA GPU is advertised as `nvidia.com/gpu`." | 以 `nvidia.com/gpu` 這個資源名稱回報 |
| 5 | single | job.md | "The `.spec.backoffLimit` is set by default to 6" | 預設重試 6 次 |
| 6 | multi_para | pod-lifecycle.md | 表格列出 `Pending`、`Running`、`Succeeded`、`Failed`、`Unknown` | 五種：等待中、執行中、全部成功結束、有容器失敗、狀態取不到 |
| 7 | multi_para | pod-lifecycle.md | "Startup probes verify whether the application within a container is started." / "Liveness probes determine when to restart a container." / "Readiness probes determine when a container is ready to accept traffic." | startup：啟動好了沒；liveness：卡死就重啟；readiness：可以接流量了沒 |
| 8 | multi_para | deployment.md | "(25% max unavailable)" / "(25% max surge)" | 逐批換新 Pod；maxUnavailable 是最多幾個可以停，maxSurge 是最多可以多開幾個，預設都是 25% |
| 9 | multi_para | service.md | `ClusterIP`: "Exposes the Service on a cluster-internal IP." 另有 `NodePort`、`LoadBalancer`、`ExternalName` | 四種：只在叢集內、每台 Node 開固定埠、外部負載平衡器、對應到外部網域名稱 |
| 10 | multi_para | taint-and-toleration.md | `NoExecute`: "Pods that do not tolerate the taint are evicted immediately"；`NoSchedule`: "No new Pods will be scheduled on the tainted node"；`PreferNoSchedule`: "a "preference" or "soft" version of `NoSchedule`" | NoExecute：連已在跑的都趕走；NoSchedule：不放新的；PreferNoSchedule：盡量不放 |
| 11 | cross | manage-resources-containers.md + node-pressure-eviction.md | "the kube-scheduler uses this information to decide which node to place the Pod on." / "Whether the pod's resource usage exceeds requests" | requests 決定排到哪台；資源不足時，用量超過 requests 的 Pod 先被趕 |
| 12 | cross | taint-and-toleration.md + assign-pod-node.md | "Node affinity is a property of Pods that *attracts* them to a set of nodes ... Taints are the opposite -- they allow a node to repel a set of pods." | affinity 是 Pod 去吸引 Node，taint 是 Node 排斥 Pod |
| 13 | cross | resource-quotas.md + limit-range.md | "provides constraints that limit aggregate resource consumption per namespace" / "A LimitRange is a policy to constrain the resource allocations ... for each applicable object kind" | Quota 管 namespace 總量，LimitRange 管單一 Pod 或容器 |
| 14 | cross | deployment.md + replicaset.md | "a Deployment is a higher-level concept that manages ReplicaSets" | Deployment 管 ReplicaSet，ReplicaSet 維持 Pod 數量 |
| 15 | cross | device-plugins.md + manage-resources-containers.md | "users can request devices as part of a Pod specification" / "Requesting extended resources is similar to how you manage requests and limits" | device plugin 向 kubelet 註冊並回報 GPU；Pod 像申請 CPU 一樣，在 resources 裡申請 `nvidia.com/gpu` |
| 16 | reasoning | node-pressure-eviction.md | "`BestEffort` or `Burstable` pods where the usage exceeds requests" are evicted first; "The kubelet does not use the pod's QoS class to determine the eviction order." | 會比較早被趕。沒設 requests 時用量一定超過 requests，所以排在前面 |
| 17 | reasoning | pod-priority-preemption.md | "the scheduler tries to preempt (evict) lower priority Pods" / "After victim Pods are preempted, they get their graceful termination period." | 搶占：趕走低優先權的 Pod，被趕的 Pod 有優雅結束的時間 |
| 18 | reasoning | statefulset.md | "The identity sticks to the Pod, regardless of which node it's (re)scheduled on." | 會保留。名字、網路身分、儲存都跟著 Pod 走 |
| 19 | boundary | device-plugins.md | "Devices cannot be shared between containers." | 文件只說裝置不能在容器之間共用，沒教 MIG 或 time-slicing，應說明資料不足 |
| 20 | boundary | （無） | 這 28 頁沒有提到叢集最大節點數 | 應拒答 |
