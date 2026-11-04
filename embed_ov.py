"""OpenVINO embedding 後端: 讓同一個 embedding 模型跑在 CPU / GPU / NPU.

流程:
    1. 第一次使用時, 用 optimum-intel 把 Hugging Face 模型轉成 OpenVINO IR (.xml/.bin),
       存到 ov_models/<模型名>/, 之後直接讀 IR.
    2. 用 openvino.Core 讀 IR, 把輸入 reshape 成固定形狀 [batch, seq_len].
       NPU 只吃靜態形狀, 為了公平比較, CPU / GPU 也用同樣的固定形狀.
    3. compile_model 到指定裝置. 設 CACHE_DIR 後, 第二次編譯會直接讀快取 (NPU 差很多).
    4. 推論後做 mean pooling + L2 正規化, 與 sentence-transformers 的
       paraphrase-multilingual-MiniLM-L12-v2 行為一致 (該模型 max_seq_length = 128).

只依賴 openvino + transformers (tokenizer). 轉檔那一步才需要 optimum-intel.
"""
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OV_MODEL_ROOT = Path(os.environ.get("OV_MODEL_DIR", HERE / "ov_models"))
OV_CACHE_DIR = Path(os.environ.get("OV_CACHE_DIR", HERE / "ov_cache"))


def ir_dir_for(model_id):
    return OV_MODEL_ROOT / model_id.replace("/", "__")


def export_ir(model_id):
    """把 HF 模型轉成 OpenVINO IR (只做一次). 回傳 IR 所在資料夾."""
    out = ir_dir_for(model_id)
    if (out / "openvino_model.xml").exists():
        return out
    from optimum.intel import OVModelForFeatureExtraction
    from transformers import AutoTokenizer

    print(f"轉換 {model_id} 為 OpenVINO IR -> {out}")
    model = OVModelForFeatureExtraction.from_pretrained(model_id, export=True, compile=False)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out)
    return out


class OVEmbedder:
    """固定形狀的 OpenVINO embedding 模型.

    device: "CPU" / "GPU" / "NPU" (也可用 "AUTO")
    batch:  一次推論幾筆; 不足時以空白補齊, 結果會丟掉補的那幾筆
    seq_len: 固定的 token 長度, 超過截斷, 不足補 pad
    """

    def __init__(self, model_id, device="CPU", batch=8, seq_len=128, use_cache=True):
        import openvino as ov
        from transformers import AutoTokenizer

        self.device, self.batch, self.seq_len = device, batch, seq_len
        ir = export_ir(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(ir)

        core = ov.Core()
        plain = device.split(":")[0] not in {"AUTO", "MULTI", "HETERO"}
        if plain and not any(d.startswith(device.split(".")[0]) for d in core.available_devices):
            raise RuntimeError(f"OpenVINO 找不到裝置 {device}, 可用: {core.available_devices}")

        model = core.read_model(ir / "openvino_model.xml")
        self.input_names = [i.get_any_name() for i in model.inputs]
        model.reshape({name: [batch, seq_len] for name in self.input_names})

        config = {}
        if use_cache:
            OV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            config["CACHE_DIR"] = str(OV_CACHE_DIR)
        t0 = time.perf_counter()
        self.compiled = core.compile_model(model, device, config)
        self.compile_s = time.perf_counter() - t0
        self.request = self.compiled.create_infer_request()

    def _infer(self, texts):
        n = len(texts)
        padded = list(texts) + [""] * (self.batch - n)
        enc = self.tokenizer(
            padded, padding="max_length", truncation=True,
            max_length=self.seq_len, return_tensors="np",
        )
        feed = {name: enc[name].astype(np.int64) for name in self.input_names}
        out = self.request.infer(feed)
        hidden = out[self.compiled.output(0)]          # [batch, seq_len, dim]
        mask = enc["attention_mask"][..., None].astype(np.float32)
        vecs = (hidden * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)  # mean pooling
        vecs /= np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        return vecs[:n]

    def encode(self, texts):
        out = [self._infer(texts[i:i + self.batch]) for i in range(0, len(texts), self.batch)]
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 0), np.float32)


def available_devices():
    import openvino as ov
    core = ov.Core()
    return {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices}


if __name__ == "__main__":
    for dev, name in available_devices().items():
        print(f"{dev:6s} {name}")
