#!/usr/bin/env python3
"""Download Kubernetes official docs (concepts) and convert them to plain Markdown.

Install:  pip install requests
Run:      python scripts/fetch_k8s_docs.py
Output:   <repo>/data/k8s/<page>.md  (data/ is not version-controlled, see .gitignore)
          <repo>/k8s_manifest.tsv     (source URL and pinned GitHub permalink per file)

The Kubernetes docs are licensed CC BY 4.0 (https://github.com/kubernetes/website/blob/main/LICENSE).
Each file starts with its source URL, a permalink to the pinned commit, and the license.
Pages are fetched from a pinned commit, so the eval results are reproducible.
"""
import os
import re

import requests

# kubernetes/website main branch, 2026-10-24
COMMIT = "bf999f0184431d6fd0908b8f8340ddf42fc4e6a6"
RAW = f"https://raw.githubusercontent.com/kubernetes/website/{COMMIT}/content/en/docs/concepts/"
BLOB = f"https://github.com/kubernetes/website/blob/{COMMIT}/content/en/docs/concepts/"
SITE = "https://kubernetes.io/docs/concepts/"

# 28 pages: workloads, networking, configuration, scheduling, storage, architecture, policy, GPU
PAGES = [
    "overview/components",
    "architecture/nodes",
    "architecture/garbage-collection",
    "workloads/pods/_index",
    "workloads/pods/pod-lifecycle",
    "workloads/pods/init-containers",
    "workloads/controllers/deployment",
    "workloads/controllers/replicaset",
    "workloads/controllers/statefulset",
    "workloads/controllers/daemonset",
    "workloads/controllers/job",
    "workloads/autoscaling",
    "services-networking/service",
    "services-networking/ingress",
    "services-networking/network-policies",
    "services-networking/dns-pod-service",
    "configuration/configmap",
    "configuration/secret",
    "configuration/manage-resources-containers",
    "scheduling-eviction/kube-scheduler",
    "scheduling-eviction/assign-pod-node",
    "scheduling-eviction/taint-and-toleration",
    "scheduling-eviction/node-pressure-eviction",
    "scheduling-eviction/pod-priority-preemption",
    "storage/persistent-volumes",
    "policy/resource-quotas",
    "policy/limit-range",
    "extend-kubernetes/compute-storage-net/device-plugins",
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "k8s")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "k8s_manifest.tsv")

ADMONITIONS = {"note": "Note", "caution": "Caution", "warning": "Warning"}
HEADINGS = {"whatsnext": "What's next", "prerequisites": "Before you begin"}


def clean(md: str) -> tuple[str, str]:
    """Strip Hugo front matter and shortcodes, keep the prose. Returns (title, body)."""
    m = re.match(r"^---\n(.*?)\n---\n", md, re.S)
    front, body = (m.group(1), md[m.end():]) if m else ("", md)
    t = re.search(r'^title:\s*"?(.+?)"?\s*$', front, re.M)
    title = t.group(1) if t else "Untitled"

    def tooltip(mt):
        attrs = mt.group(1)
        a = re.search(r'text="([^"]*)"', attrs) or re.search(r'term_id="([^"]*)"', attrs)
        return a.group(1) if a else ""

    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"\{\{<\s*glossary_tooltip\b(.*?)>\}\}", tooltip, body, flags=re.S)
    body = re.sub(r'\{\{%\s*heading\s+"(\w+)"\s*%\}\}',
                  lambda mt: HEADINGS.get(mt.group(1), mt.group(1)), body)
    for tag, label in ADMONITIONS.items():
        body = re.sub(r"\{\{[<%]\s*" + tag + r"\b[^>%]*[>%]\}\}", f"**{label}:** ", body)
    # Any other shortcode (feature-state, code_sample, tabs, skew, ...) is removed
    body = re.sub(r"\{\{[<%].*?[>%]\}\}", "", body, flags=re.S)
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", body)              # images
    body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)          # links -> link text
    body = re.sub(r"<[^>\n]+>", "", body)                         # leftover HTML tags
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return title, body.strip()


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = ["file\tsource_url\tpermalink"]
    for page in PAGES:
        r = requests.get(RAW + page + ".md", timeout=30)
        r.raise_for_status()
        title, body = clean(r.text)
        path = page[: -len("/_index")] if page.endswith("/_index") else page
        name = path.rsplit("/", 1)[-1] + ".md"
        url, link = SITE + path + "/", BLOB + page + ".md"
        header = (
            f"# {title}\n\n"
            f"> Source: <{url}>\n"
            f"> Permalink: <{link}>\n"
            f"> License: CC BY 4.0 (The Kubernetes Authors)\n\n---\n\n"
        )
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(header + body + "\n")
        rows.append(f"{name}\t{url}\t{link}")
        print(f"{name:<32} {len(body):>6} chars")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
