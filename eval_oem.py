"""English eval set: Ubuntu linux-oem-7.0 kernel changelog, 20 questions.

Corpus: data/oem-7.0/, one file per version (scripts/fetch_oem_changelog.py).
Used by eval.py with the --oem flag:
    python eval.py --oem            # retrieval only, k=3
    python eval.py --oem 5 --llm    # k=5, also call the LLM for refuse/manual questions

Same fields and type mix as eval.py / eval_k8s.py
(single 5 / multi_para 5 / cross 5 / reasoning 3 / boundary 2).
Every "answer" question was checked against the changelog on 2026-11-06
(latest version 7.0.0-1018.18).

Known difficulty: the version number is only in each file's header, so chunks from the
middle of a long entry (1017.17 has 5000+ lines) do not say which version they belong to.
Questions that ask "which version" test exactly this weakness.
"""


def v(ver):
    return f"linux-oem-7.0_7.0.0-{ver}.md"


EVAL_SET_OEM = [
    # ---- single: one fact in one version ----
    {"id": 1, "type": "single", "expect": "answer",
     "q": "Which change fixed the roughly 42x slowdown in SDXL inference?",
     "expected_sources": [v("1010.10")],
     "note": "drm/amdgpu: drop retry loop in amdgpu_hmm_range_get_pages"},
    {"id": 2, "type": "single", "expect": "answer",
     "q": "How was the black screen on an 8K monitor fixed in the AMD display driver?",
     "expected_sources": [v("1010.10")],
     "note": "drm/amd/display: Skip PHY SSC reduction on some 8K panels"},
    {"id": 3, "type": "single", "expect": "answer",
     "q": "Which ACPI ID was added to ipu-bridge for the Sony IMX471 camera sensor?",
     "expected_sources": [v("1004.4")],
     "note": "TBE20A0"},
    {"id": 4, "type": "single", "expect": "answer",
     "q": "Which real-time patchset version was applied to the Resolute kernel?",
     "expected_sources": [v("1012.12")],
     "note": "7.0.1-rt2"},
    {"id": 5, "type": "single", "expect": "answer",
     "q": "Why did the RTC device disappear after updating to 7.0.0-39-generic, and what was changed?",
     "expected_sources": [v("1017.17")],
     "note": "rtc-cmos now binds via a platform device; CMOS RTC PNP device support dropped"},

    # ---- multi_para: needs more than one line of the same entry ----
    {"id": 6, "type": "multi_para", "expect": "answer",
     "q": "Which IPU7 modules were shipped unsigned, what did that break, and how was it fixed?",
     "expected_sources": [v("1004.4")],
     "note": "intel-ipu7 / intel-ipu7-isys; broke Secure Boot; added to signature inclusion list"},
    {"id": 7, "type": "multi_para", "expect": "answer",
     "q": "What was the backlight regression fix?",
     "expected_sources": [v("1012.12")],
     "note": "Revert \"drm/i915/backlight: Remove try_vesa_interface\""},
    {"id": 8, "type": "multi_para", "expect": "answer",
     "q": "How was the IPU firmware made to be packed into the initramfs automatically?",
     "expected_sources": [v("1015.15")],
     "note": "ipu6 and ipu7 drivers declare their module firmwares"},
    {"id": 9, "type": "multi_para", "expect": "answer",
     "q": "What caused the internal display black screen on Intel Lunar Lake with an eDP panel, and what was the fix?",
     "expected_sources": [v("1009.9")],
     "note": "drm/i915/alpm: allow LOBF only for platforms with Always on VRR TG"},
    {"id": 10, "type": "multi_para", "expect": "answer",
     "q": "What was done so the system can enter s0ix suspend with RTL8116AF ethernet?",
     "expected_sources": [v("1016.16")],
     "note": "Revert the r8169 SAUCE quirk for RTL8116af SerDes"},

    # ---- cross: answer spans several versions ----
    {"id": 11, "type": "cross", "expect": "answer",
     "q": "Which versions include fixes to the Intel NPU driver (accel/ivpu)?",
     "expected_sources": [v("1009.9"), v("1012.12"), v("1015.15"), v("1017.17")]},
    {"id": 12, "type": "cross", "expect": "answer",
     "q": "Which versions mention adding support for the amd-pmf AMDI0112 ID?",
     "expected_sources": [v("1014.14"), v("1016.16")]},
    {"id": 13, "type": "cross", "expect": "answer",
     "q": "Which versions contain fixes for the TI TAS2783 amplifier?",
     "expected_sources": [v("1014.14"), v("1015.15"), v("1018.18")]},
    {"id": 14, "type": "cross", "expect": "answer",
     "q": "Which versions fix problems with the Intel E810 network card (ice driver)?",
     "expected_sources": [v("1009.9"), v("1012.12"), v("1015.15")]},
    {"id": 15, "type": "cross", "expect": "answer",
     "q": "Which versions mention the nova-core GPU driver conflicting with nouveau?",
     "expected_sources": [v("1006.6"), v("1007.7"), v("1009.9"), v("1017.17")]},

    # ---- reasoning: needs a judgement on top of retrieval ----
    {"id": 16, "type": "reasoning", "expect": "manual",
     "q": "Is the OV02C10 front camera upside-down image fixed if I am on 7.0.0-1017.17?",
     "expected_sources": [v("1018.18")],
     "note": "No, the ipu-bridge rotation quirk fix is in 1018.18"},
    {"id": 17, "type": "reasoning", "expect": "manual",
     "q": "I am on 7.0.0-1007.7 and iwlwifi reports it failed to handle oversized command 0xC05. Which version should I upgrade to at least?",
     "expected_sources": [v("1008.8")],
     "note": "1008.8 (avoid oversized UATS command copy); 1012.12 has the same bug with more fixes"},
    {"id": 18, "type": "reasoning", "expect": "manual",
     "q": "Was the MediaTek MT7920 Bluetooth firmware declared before 7.0.0-1016.16?",
     "expected_sources": [v("1016.16")],
     "note": "No, btmtk declares MT7920 (MT7961 1a) firmware in 1016.16"},

    # ---- boundary: not in the changelog, should refuse ----
    {"id": 19, "type": "boundary", "expect": "refuse",
     "q": "Which Mesa version ships with linux-oem-7.0?",
     "expected_sources": [],
     "note": "Mesa is user space, not in the kernel changelog"},
    {"id": 20, "type": "boundary", "expect": "refuse",
     "q": "When will linux-oem-7.0 7.0.0-1019 be released?",
     "expected_sources": [],
     "note": "Future version, not in the data"},
]
