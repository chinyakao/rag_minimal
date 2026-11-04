"""讀取 Linux 上的功耗與 NPU 使用率計數器 (讀不到就略過, 不影響速度量測).

- RAPL (intel-rapl powercap): /sys/class/powercap/intel-rapl:*/energy_uj
  累積能量 (微焦耳), 前後相減 / 時間 = 平均功率. package 包含 CPU 核心, iGPU 與 NPU
  (Meteor Lake 的 NPU 在 SoC 內), 所以看的是「整顆 SoC 多耗了多少電」.
  多數發行版 energy_uj 只有 root 可讀, 見 README 的權限說明.
- NPU busy time (intel_vpu 驅動): /sys/class/accel/accel*/device/npu_busy_time_us
  NPU 實際在工作的累積時間, 前後相減 / 時間 = NPU 使用率.
"""
import glob
import os
import time


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def rapl_domains():
    """回傳可讀的 RAPL domain: {名稱: (energy_uj 路徑, 溢位上限)}."""
    out = {}
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        name_path, energy_path = os.path.join(d, "name"), os.path.join(d, "energy_uj")
        if _read_int(energy_path) is None:
            continue
        try:
            name = open(name_path).read().strip()
        except OSError:
            name = os.path.basename(d)
        # 子 domain (intel-rapl:0:0) 名稱加上前綴, 避免和 package 重名
        if d.count(":") > 1:
            name = f"{os.path.basename(d)}-{name}"
        out[name] = (energy_path, _read_int(os.path.join(d, "max_energy_range_uj")) or 2**32)
    return out


def npu_busy_path():
    paths = glob.glob("/sys/class/accel/accel*/device/npu_busy_time_us")
    return paths[0] if paths else None


class PowerMeter:
    """用法:
        pm = PowerMeter()
        pm.start(); ...工作...; stats = pm.stop()
    stats 例: {"seconds": 10.2, "package-0_W": 8.1, "npu_util": 0.93}
    """

    def __init__(self):
        self.domains = rapl_domains()
        self.npu_path = npu_busy_path()

    @property
    def available(self):
        return {"rapl": list(self.domains), "npu_busy": bool(self.npu_path)}

    def _snapshot(self):
        snap = {"t": time.perf_counter()}
        for name, (path, _) in self.domains.items():
            snap[name] = _read_int(path)
        if self.npu_path:
            snap["npu_us"] = _read_int(self.npu_path)
        return snap

    def start(self):
        self._t0 = self._snapshot()

    def stop(self):
        t1, t0 = self._snapshot(), self._t0
        dt = t1["t"] - t0["t"]
        stats = {"seconds": dt}
        for name, (_, max_range) in self.domains.items():
            if t0.get(name) is None or t1.get(name) is None:
                continue
            delta = t1[name] - t0[name]
            if delta < 0:  # 計數器溢位繞回
                delta += max_range
            stats[f"{name}_J"] = delta / 1e6
            stats[f"{name}_W"] = delta / 1e6 / dt if dt > 0 else None
        if self.npu_path and t0.get("npu_us") is not None and t1.get("npu_us") is not None:
            stats["npu_util"] = (t1["npu_us"] - t0["npu_us"]) / 1e6 / dt if dt > 0 else None
        return stats


def idle_baseline(seconds=5.0):
    """量測閒置功耗, 讓結果可以換算成「比閒置多耗多少」."""
    pm = PowerMeter()
    pm.start()
    time.sleep(seconds)
    return pm.stop()


if __name__ == "__main__":
    pm = PowerMeter()
    print("可用計數器:", pm.available)
    print("閒置 5 秒:", idle_baseline(5))
