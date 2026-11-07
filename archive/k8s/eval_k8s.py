"""English eval set: Kubernetes concepts docs, 20 questions.

Used by eval.py with the --k8s flag:
    python eval.py --k8s            # retrieval only, k=3
    python eval.py --k8s 5 --llm    # k=5, also call the LLM for refuse/manual questions

Same fields and type mix as the Chinese set in eval.py
(single 5 / multi_para 5 / cross 5 / reasoning 3 / boundary 2).
Every "answer" question was checked against the docs at the commit pinned in
scripts/fetch_k8s_docs.py.
"""

EVAL_SET_K8S = [
    # ---- single: one fact in one page ----
    {"id": 1, "type": "single", "expect": "answer",
     "q": "What is the maximum size of the data stored in a ConfigMap?",
     "expected_sources": ["configmap.md"],
     "note": "1 MiB"},
    {"id": 2, "type": "single", "expect": "answer",
     "q": "What is the default restartPolicy of a Pod?",
     "expected_sources": ["pod-lifecycle.md"],
     "note": "Always"},
    {"id": 3, "type": "single", "expect": "answer",
     "q": "What does a DaemonSet guarantee about the nodes in a cluster?",
     "expected_sources": ["daemonset.md"]},
    {"id": 4, "type": "single", "expect": "answer",
     "q": "How is an NVIDIA GPU advertised to Kubernetes by a device plugin?",
     "expected_sources": ["device-plugins.md"],
     "note": "as the extended resource nvidia.com/gpu"},
    {"id": 5, "type": "single", "expect": "answer",
     "q": "By default, how many times does a Job retry failed Pods before it is marked as failed?",
     "expected_sources": ["job.md"],
     "note": ".spec.backoffLimit defaults to 6"},

    # ---- multi_para: the answer spans several paragraphs of one page ----
    {"id": 6, "type": "multi_para", "expect": "answer",
     "q": "What are the possible values of a Pod's phase, and what does each one mean?",
     "expected_sources": ["pod-lifecycle.md"]},
    {"id": 7, "type": "multi_para", "expect": "answer",
     "q": "What is the difference between liveness, readiness, and startup probes?",
     "expected_sources": ["pod-lifecycle.md"]},
    {"id": 8, "type": "multi_para", "expect": "answer",
     "q": "How does a Deployment perform a rolling update, and what do maxUnavailable and maxSurge control?",
     "expected_sources": ["deployment.md"],
     "note": "defaults are 25% max unavailable and 25% max surge"},
    {"id": 9, "type": "multi_para", "expect": "answer",
     "q": "What Service types does Kubernetes support, and when would you use each one?",
     "expected_sources": ["service.md"]},
    {"id": 10, "type": "multi_para", "expect": "answer",
     "q": "What are the three taint effects, and how does each affect Pods that do not tolerate the taint?",
     "expected_sources": ["taint-and-toleration.md"],
     "note": "NoSchedule, PreferNoSchedule, NoExecute"},

    # ---- cross: the answer needs two or more pages ----
    {"id": 11, "type": "cross", "expect": "answer",
     "q": "How do container resource requests and limits affect both scheduling and node-pressure eviction?",
     "expected_sources": ["manage-resources-containers.md", "node-pressure-eviction.md"]},
    {"id": 12, "type": "cross", "expect": "answer",
     "q": "Compare taints and tolerations with node affinity. What problem does each one solve?",
     "expected_sources": ["taint-and-toleration.md", "assign-pod-node.md"]},
    {"id": 13, "type": "cross", "expect": "answer",
     "q": "How do ResourceQuota and LimitRange work together to control resource usage in a namespace?",
     "expected_sources": ["resource-quotas.md", "limit-range.md"]},
    {"id": 14, "type": "cross", "expect": "answer",
     "q": "What is the relationship between a Deployment, a ReplicaSet, and its Pods?",
     "expected_sources": ["deployment.md", "replicaset.md"]},
    {"id": 15, "type": "cross", "expect": "answer",
     "q": "How does a container request a GPU, and how does the device plugin make the GPU available to it?",
     "expected_sources": ["device-plugins.md", "manage-resources-containers.md"]},

    # ---- reasoning: combine facts to reach a conclusion ----
    {"id": 16, "type": "reasoning", "expect": "answer",
     "q": "A Pod with no resource requests uses a lot of memory on a node under memory pressure. "
          "Is it likely to be evicted before a Pod whose usage stays within its requests? Why?",
     "expected_sources": ["node-pressure-eviction.md"],
     "note": "kubelet ranks by usage exceeding requests, then priority; it does not use the QoS class directly"},
    {"id": 17, "type": "reasoning", "expect": "answer",
     "q": "A high-priority Pod cannot be scheduled because the cluster is full. "
          "What can the scheduler do, and what happens to the lower-priority Pods?",
     "expected_sources": ["pod-priority-preemption.md"],
     "note": "preemption; victims get their graceful termination period"},
    {"id": 18, "type": "reasoning", "expect": "answer",
     "q": "A StatefulSet Pod is rescheduled onto another node. Does it keep the same network identity and storage? Why?",
     "expected_sources": ["statefulset.md"]},

    # ---- boundary: the answer is not (fully) in the corpus ----
    {"id": 19, "type": "boundary", "expect": "manual",
     "q": "How do I share one NVIDIA GPU between several Pods using MIG or time-slicing?",
     "expected_sources": ["device-plugins.md"],
     "note": "the docs only say extended resources cannot be overcommitted and devices cannot be shared between containers; "
             "MIG and time-slicing are vendor features not described here, so a good answer says so"},
    {"id": 20, "type": "boundary", "expect": "refuse",
     "q": "What is the maximum number of nodes supported in a single Kubernetes cluster?",
     "expected_sources": [],
     "note": "covered on a page outside this corpus (large-cluster best practices); should refuse, not answer from memory"},
]
