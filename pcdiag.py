#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import curses
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import psutil

SPARK_CHARS = ".:-=+*#%@"
CPU_TEMP_WATCH_C = 60.0
CPU_TEMP_NEAR_C = 64.0
CPU_TEMP_LIMIT_C = 68.0
UTIL_WARN_PCT = 70.0
UTIL_DANGER_PCT = 90.0
MEM_WARN_PCT = 80.0
MEM_DANGER_PCT = 90.0
SORT_FIELDS = ["impact", "cpu_pct", "ram_pct", "gpu_pct", "vram_mib", "rss_bytes"]
SORT_LABELS = {
    "impact": "IMPACT",
    "cpu_pct": "CPU%",
    "ram_pct": "RAM%",
    "gpu_pct": "GPU%",
    "vram_mib": "VRAM",
    "rss_bytes": "RSS",
}
PAIR_ACCENT = 1
PAIR_GOOD = 2
PAIR_WARN = 3
PAIR_DANGER = 4
METRIC_LABEL_WIDTH = 10
METRIC_VALUE_WIDTH = 8


@dataclass
class CpuTempInfo:
    main_label: str | None
    main_temp_c: float | None
    detail: str


@dataclass
class GpuDevice:
    index: int
    uuid: str
    name: str
    util_pct: float
    temp_c: float
    mem_used_mib: float
    mem_total_mib: float
    fan_pct: float | None
    power_draw_w: float | None
    power_limit_w: float | None
    temp_limit_c: float | None
    sw_thermal_slowdown: bool
    hw_thermal_slowdown: bool


@dataclass
class GpuInfo:
    label: str
    util_pct: float | None
    temp_c: float | None
    mem_pct: float | None
    mem_used_mib: float | None
    mem_total_mib: float | None
    fan_pct: float | None
    power_draw_w: float | None
    power_limit_w: float | None
    temp_limit_c: float | None
    thermal_slowdown: bool
    total_mem_mib_by_uuid: dict[str, float]


@dataclass
class ProcessUsage:
    pid: int
    name: str
    cpu_pct: float
    ram_pct: float
    rss_bytes: int
    gpu_pct: float | None
    vram_mib: float | None
    score: float


@dataclass
class Snapshot:
    timestamp: datetime
    cpu_util_pct: float
    cpu_temps: CpuTempInfo
    gpu: GpuInfo
    ram_pct: float
    ram_used_bytes: int
    ram_total_bytes: int
    processes: list[ProcessUsage]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def flag_is_active(value: str) -> bool:
    return value.strip().lower() == "active"


def safe_float(value: str) -> float | None:
    text = value.strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return text[:1]
    return text[: width - 1] + "."


def gib_from_bytes(value: int) -> float:
    return value / (1024 ** 3)


def format_bytes_short(value: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(number)}{unit}"
            return f"{number:.1f}{unit}"
        number /= 1024.0
    return f"{number:.1f}T"


def format_mib_short(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1024:
        return f"{value / 1024:.1f}G"
    return f"{value:.0f}M"


def format_pct_short(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}%"


def format_watts_short(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}W"
    return f"{value:.1f}W"


def gpu_power_pct(gpu: GpuInfo) -> float | None:
    if gpu.power_draw_w is None or gpu.power_limit_w is None or gpu.power_limit_w <= 0:
        return None
    return gpu.power_draw_w / gpu.power_limit_w * 100.0


def gauge(value: float | None, max_value: float, width: int) -> str:
    if width <= 0:
        return ""
    if value is None:
        return "." * width
    ratio = clamp(value / max_value, 0.0, 1.0)
    filled = int(round(ratio * width))
    return "#" * filled + "." * (width - filled)


def sparkline(history: Iterable[float], max_value: float, width: int) -> str:
    if width <= 0:
        return ""
    values = list(history)[-width:]
    if len(values) < width:
        values = [0.0] * (width - len(values)) + values
    chars: list[str] = []
    top_index = len(SPARK_CHARS) - 1
    for value in values:
        ratio = clamp(value / max_value, 0.0, 1.0)
        chars.append(SPARK_CHARS[int(round(ratio * top_index))])
    return "".join(chars)


def run_command(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def query_nvidia_devices() -> list[GpuDevice]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,temperature.gpu,memory.used,memory.total,fan.speed,power.draw,power.limit,temperature.gpu.tlimit,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_thermal_slowdown",
            "--format=csv,noheader,nounits",
        ]
    )
    expected_columns = 13
    fallback = False
    if not output:
        fallback = True
        output = run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    if not output:
        return []

    devices: list[GpuDevice] = []
    reader = csv.reader(output.splitlines())
    for row in reader:
        if not fallback and len(row) != expected_columns:
            continue
        if fallback and len(row) != 7:
            continue
        util_pct = safe_float(row[3]) or 0.0
        temp_c = safe_float(row[4]) or 0.0
        mem_used_mib = safe_float(row[5]) or 0.0
        mem_total_mib = safe_float(row[6]) or 0.0
        fan_pct: float | None = None
        power_draw_w: float | None = None
        power_limit_w: float | None = None
        temp_limit_c: float | None = None
        sw_thermal_slowdown = False
        hw_thermal_slowdown = False
        if not fallback:
            fan_pct = safe_float(row[7])
            power_draw_w = safe_float(row[8])
            power_limit_w = safe_float(row[9])
            temp_margin_c = safe_float(row[10])
            if temp_margin_c is not None:
                temp_limit_c = temp_c + temp_margin_c
            sw_thermal_slowdown = flag_is_active(row[11])
            hw_thermal_slowdown = flag_is_active(row[12])
        try:
            devices.append(
                GpuDevice(
                    index=int(row[0].strip()),
                    uuid=row[1].strip(),
                    name=row[2].strip(),
                    util_pct=util_pct,
                    temp_c=temp_c,
                    mem_used_mib=mem_used_mib,
                    mem_total_mib=mem_total_mib,
                    fan_pct=fan_pct,
                    power_draw_w=power_draw_w,
                    power_limit_w=power_limit_w,
                    temp_limit_c=temp_limit_c,
                    sw_thermal_slowdown=sw_thermal_slowdown,
                    hw_thermal_slowdown=hw_thermal_slowdown,
                )
            )
        except ValueError:
            continue
    return devices


def summarize_gpus(devices: list[GpuDevice]) -> GpuInfo:
    if not devices:
        return GpuInfo(
            label="No NVIDIA GPU detected",
            util_pct=None,
            temp_c=None,
            mem_pct=None,
            mem_used_mib=None,
            mem_total_mib=None,
            fan_pct=None,
            power_draw_w=None,
            power_limit_w=None,
            temp_limit_c=None,
            thermal_slowdown=False,
            total_mem_mib_by_uuid={},
        )

    total_mem_used = sum(device.mem_used_mib for device in devices)
    total_mem_total = sum(device.mem_total_mib for device in devices)
    mem_pct = (total_mem_used / total_mem_total * 100.0) if total_mem_total else None
    total_mem_mib_by_uuid = {device.uuid: device.mem_total_mib for device in devices}

    if len(devices) == 1:
        device = devices[0]
        return GpuInfo(
            label=device.name,
            util_pct=device.util_pct,
            temp_c=device.temp_c,
            mem_pct=mem_pct,
            mem_used_mib=device.mem_used_mib,
            mem_total_mib=device.mem_total_mib,
            fan_pct=device.fan_pct,
            power_draw_w=device.power_draw_w,
            power_limit_w=device.power_limit_w,
            temp_limit_c=device.temp_limit_c,
            thermal_slowdown=device.sw_thermal_slowdown or device.hw_thermal_slowdown,
            total_mem_mib_by_uuid=total_mem_mib_by_uuid,
        )

    temp_limits = [device.temp_limit_c for device in devices if device.temp_limit_c is not None]
    fan_values = [device.fan_pct for device in devices if device.fan_pct is not None]
    power_draw_values = [device.power_draw_w for device in devices if device.power_draw_w is not None]
    power_limit_values = [device.power_limit_w for device in devices if device.power_limit_w is not None]
    return GpuInfo(
        label=f"{len(devices)} NVIDIA GPUs",
        util_pct=sum(device.util_pct for device in devices) / len(devices),
        temp_c=max(device.temp_c for device in devices),
        mem_pct=mem_pct,
        mem_used_mib=total_mem_used,
        mem_total_mib=total_mem_total,
        fan_pct=(sum(fan_values) / len(fan_values)) if fan_values else None,
        power_draw_w=sum(power_draw_values) if power_draw_values else None,
        power_limit_w=sum(power_limit_values) if power_limit_values else None,
        temp_limit_c=min(temp_limits) if temp_limits else None,
        thermal_slowdown=any(
            device.sw_thermal_slowdown or device.hw_thermal_slowdown for device in devices
        ),
        total_mem_mib_by_uuid=total_mem_mib_by_uuid,
    )


def query_gpu_process_usage() -> dict[int, float]:
    output = run_command(["nvidia-smi", "pmon", "-c", "1"])
    if not output:
        return {}

    usage_by_pid: dict[int, float] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 9)
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        util_pct = safe_float(parts[3]) or 0.0
        usage_by_pid[pid] = usage_by_pid.get(pid, 0.0) + util_pct
    return usage_by_pid


def query_gpu_process_memory() -> dict[int, float]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {}

    memory_by_pid: dict[int, float] = {}
    reader = csv.reader(output.splitlines())
    for row in reader:
        if len(row) != 3:
            continue
        try:
            pid = int(row[0].strip())
        except ValueError:
            continue
        memory_mib = safe_float(row[2]) or 0.0
        if memory_mib <= 0:
            continue
        memory_by_pid[pid] = memory_by_pid.get(pid, 0.0) + memory_mib
    return memory_by_pid


def is_plausible_temp(value: float | None) -> bool:
    return value is not None and 0.0 < value < 125.0


def extract_cpu_temps() -> CpuTempInfo:
    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        sensors = {}

    best_by_label: dict[str, tuple[str, float]] = {}
    for entries in sensors.values():
        for entry in entries:
            label = (entry.label or "").strip()
            current = float(entry.current)
            if not is_plausible_temp(current):
                continue
            key = label.lower() or "unlabeled"
            existing = best_by_label.get(key)
            if existing is None or current > existing[1]:
                best_by_label[key] = (label or "temp", current)

    details: list[str] = []

    hottest_tccd: tuple[str, float] | None = None
    for key, entry in best_by_label.items():
        if not key.startswith("tccd"):
            continue
        if hottest_tccd is None or entry[1] > hottest_tccd[1]:
            hottest_tccd = entry

    main_entry: tuple[str, float] | None = None
    for primary_key in ("tdie", "cpu temperature"):
        entry = best_by_label.get(primary_key)
        if entry:
            main_entry = entry
            break

    if main_entry is None and hottest_tccd:
        main_entry = hottest_tccd

    if main_entry is None and "tctl" in best_by_label:
        main_entry = best_by_label["tctl"]

    if main_entry is None:
        return CpuTempInfo(main_label=None, main_temp_c=None, detail="No CPU temp sensor found")

    detail_labels: list[str] = []
    for key in ("tdie",):
        entry = best_by_label.get(key)
        if entry:
            label, current = entry
            details.append(f"{label} {current:.1f}C")
            detail_labels.append(label)

    if hottest_tccd and hottest_tccd[0] not in detail_labels:
        label, current = hottest_tccd
        details.append(f"{label} {current:.1f}C")
        detail_labels.append(label)

    if main_label := (main_entry[0] if main_entry else None):
        if main_label == "Tctl" and main_label not in detail_labels:
            details.append(f"{main_label} {main_entry[1]:.1f}C")
            detail_labels.append(main_label)

    board_cpu_temp = best_by_label.get("cpu temperature")
    if board_cpu_temp and board_cpu_temp[0] not in detail_labels and "tdie" not in best_by_label:
        label, current = board_cpu_temp
        details.append(f"{label} {current:.1f}C")
        detail_labels.append(label)

    main_label, main_temp = main_entry
    return CpuTempInfo(
        main_label=main_label,
        main_temp_c=main_temp,
        detail=" | ".join(details),
    )


def classify_utilization(
    value: float | None, warn_pct: float = UTIL_WARN_PCT, danger_pct: float = UTIL_DANGER_PCT
) -> tuple[str, str]:
    if value is None:
        return ("n/a", "neutral")
    if value >= danger_pct:
        return ("high", "danger")
    if value >= warn_pct:
        return ("busy", "warn")
    return ("ok", "good")


def classify_cpu_temp(value: float | None) -> tuple[str, str]:
    if value is None:
        return ("n/a", "neutral")
    if value >= CPU_TEMP_LIMIT_C:
        return ("max", "danger")
    if value >= CPU_TEMP_NEAR_C:
        return ("near-limit", "warn")
    if value >= CPU_TEMP_WATCH_C:
        return ("watch", "warn")
    return ("ok", "good")


def gpu_temp_thresholds(limit_c: float | None) -> tuple[float, float, float]:
    if limit_c is None:
        return (75.0, 80.0, 84.0)
    watch_c = max(limit_c - 10.0, 0.0)
    near_c = max(limit_c - 5.0, 0.0)
    return (watch_c, near_c, limit_c)


def classify_gpu_temp(
    value: float | None,
    limit_c: float | None,
    thermal_slowdown: bool,
) -> tuple[str, str]:
    if value is None:
        return ("n/a", "neutral")
    if thermal_slowdown:
        return ("throttling", "danger")
    watch_c, near_c, danger_c = gpu_temp_thresholds(limit_c)
    if value >= danger_c:
        return ("limit", "danger")
    if value >= near_c:
        return ("near-limit", "warn")
    if value >= watch_c:
        return ("watch", "warn")
    return ("ok", "good")


def sort_value(process: ProcessUsage, sort_field: str) -> float:
    if sort_field == "impact":
        return process.score
    if sort_field == "cpu_pct":
        return process.cpu_pct
    if sort_field == "ram_pct":
        return process.ram_pct
    if sort_field == "gpu_pct":
        return process.gpu_pct or 0.0
    if sort_field == "vram_mib":
        return process.vram_mib or 0.0
    if sort_field == "rss_bytes":
        return float(process.rss_bytes)
    return process.score


def sort_processes(processes: list[ProcessUsage], sort_field: str) -> list[ProcessUsage]:
    return sorted(
        processes,
        key=lambda process: (
            sort_value(process, sort_field),
            process.score,
            process.gpu_pct or 0.0,
            process.vram_mib or 0.0,
            process.cpu_pct,
            process.ram_pct,
            process.rss_bytes,
        ),
        reverse=True,
    )


class Monitor:
    def __init__(self, history_size: int, top_n: int) -> None:
        self.history: dict[str, deque[float]] = {
            "cpu_util": deque(maxlen=history_size),
            "cpu_temp": deque(maxlen=history_size),
            "gpu_util": deque(maxlen=history_size),
            "gpu_mem": deque(maxlen=history_size),
            "gpu_fan": deque(maxlen=history_size),
            "gpu_power": deque(maxlen=history_size),
            "gpu_temp": deque(maxlen=history_size),
            "ram": deque(maxlen=history_size),
        }
        self.top_n = top_n
        self.cpu_count = max(psutil.cpu_count(logical=True) or 1, 1)
        self.primed_pids: set[int] = set()
        self.prime_process_counters()

    def prime_process_counters(self) -> None:
        psutil.cpu_percent(interval=None)
        for proc in psutil.process_iter():
            try:
                proc.cpu_percent(None)
                self.primed_pids.add(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def reset_history(self) -> None:
        for series in self.history.values():
            series.clear()

    def collect(self) -> Snapshot:
        cpu_util_pct = psutil.cpu_percent(interval=None)
        cpu_temps = extract_cpu_temps()
        gpu_devices = query_nvidia_devices()
        gpu = summarize_gpus(gpu_devices)
        gpu_usage_by_pid = query_gpu_process_usage()
        gpu_memory_by_pid = query_gpu_process_memory()
        ram = psutil.virtual_memory()

        if cpu_temps.main_temp_c is not None:
            self.history["cpu_temp"].append(cpu_temps.main_temp_c)
        if gpu.util_pct is not None:
            self.history["gpu_util"].append(gpu.util_pct)
        if gpu.mem_pct is not None:
            self.history["gpu_mem"].append(gpu.mem_pct)
        if gpu.fan_pct is not None:
            self.history["gpu_fan"].append(gpu.fan_pct)
        power_pct = gpu_power_pct(gpu)
        if power_pct is not None:
            self.history["gpu_power"].append(power_pct)
        if gpu.temp_c is not None:
            self.history["gpu_temp"].append(gpu.temp_c)
        self.history["cpu_util"].append(cpu_util_pct)
        self.history["ram"].append(ram.percent)

        processes = self.collect_processes(gpu_usage_by_pid, gpu_memory_by_pid, gpu.mem_total_mib)
        return Snapshot(
            timestamp=datetime.now(),
            cpu_util_pct=cpu_util_pct,
            cpu_temps=cpu_temps,
            gpu=gpu,
            ram_pct=ram.percent,
            ram_used_bytes=ram.used,
            ram_total_bytes=ram.total,
            processes=processes,
        )

    def collect_processes(
        self,
        gpu_usage_by_pid: dict[int, float],
        gpu_memory_by_pid: dict[int, float],
        gpu_total_mib: float | None,
    ) -> list[ProcessUsage]:
        rows: list[ProcessUsage] = []
        live_pids: set[int] = set()

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                live_pids.add(proc.pid)
                if proc.pid not in self.primed_pids:
                    proc.cpu_percent(None)
                    self.primed_pids.add(proc.pid)
                    cpu_pct = 0.0
                else:
                    cpu_pct = proc.cpu_percent(None)

                with proc.oneshot():
                    ram_pct = proc.memory_percent()
                    rss_bytes = proc.memory_info().rss
                    cmdline = proc.info.get("cmdline") or []
                    name = " ".join(cmdline).strip() or proc.info.get("name") or str(proc.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

            gpu_pct = gpu_usage_by_pid.get(proc.pid)
            vram_mib = gpu_memory_by_pid.get(proc.pid)
            vram_pct = 0.0
            if vram_mib is not None and gpu_total_mib:
                vram_pct = vram_mib / gpu_total_mib * 100.0

            score = (cpu_pct / self.cpu_count) + ram_pct + (gpu_pct or 0.0) + vram_pct
            if score <= 0.05:
                continue

            rows.append(
                ProcessUsage(
                    pid=proc.pid,
                    name=name,
                    cpu_pct=cpu_pct,
                    ram_pct=ram_pct,
                    rss_bytes=rss_bytes,
                    gpu_pct=gpu_pct,
                    vram_mib=vram_mib,
                    score=score,
                )
            )

        self.primed_pids.intersection_update(live_pids)
        rows.sort(
            key=lambda row: (
                row.score,
                row.gpu_pct or 0.0,
                row.vram_mib or 0.0,
                row.cpu_pct,
                row.ram_pct,
            ),
            reverse=True,
        )
        return rows


def render_metric_line(
    width: int,
    label: str,
    value: str,
    bar: str,
    trace: str,
    status: str = "",
    detail: str = "",
    *,
    label_width: int = METRIC_LABEL_WIDTH,
    value_width: int = METRIC_VALUE_WIDTH,
) -> str:
    parts = [f"{label:<{label_width}}", f"{value:>{value_width}}", f"[{bar}]"]
    line = " ".join(parts)
    if trace:
        line += " " + trace
    if status:
        line += " " + status
    if detail:
        line += " " + detail
    return truncate(line, width)


def trace_levels_for_values(
    values: Iterable[float],
    metric_name: str,
    gpu: GpuInfo | None = None,
) -> list[str]:
    levels: list[str] = []
    for value in values:
        if metric_name == "cpu_temp":
            _, level = classify_cpu_temp(value)
        elif metric_name == "gpu_temp":
            _, level = classify_gpu_temp(value, gpu.temp_limit_c if gpu else None, False)
        elif metric_name in {"gpu_mem", "ram"}:
            _, level = classify_utilization(value, MEM_WARN_PCT, MEM_DANGER_PCT)
        else:
            _, level = classify_utilization(value)
        levels.append(level)
    return levels


def section_divider_text(width: int, label: str) -> str:
    if not label:
        return "-" * max(0, width)
    if width <= len(label) + 2:
        return truncate(label, width)
    inner = f" {label} "
    dash_count = max(0, width - len(inner))
    left = dash_count // 2
    right = dash_count - left
    return "-" * left + inner + "-" * right


def sparkline_values(history: Iterable[float], width: int) -> list[float]:
    values = list(history)[-width:]
    if len(values) < width:
        values = [0.0] * (width - len(values)) + values
    return values


def compact_bar_width(width: int) -> int:
    return max(6, min(12, (width - 22) // 2))


def compact_status_text(status: str) -> str:
    mapping = {
        "ok": "ok",
        "busy": "busy",
        "watch": "watch",
        "near-limit": "warn",
        "high": "high",
        "limit": "hot",
        "max": "hot",
        "throttling": "hot",
        "n/a": "na",
    }
    return mapping.get(status, status[:4].lower())


def compact_process_name(command: str) -> str:
    if not command:
        return "-"
    head = command.split()[0]
    tail = head.rsplit("/", 1)[-1]
    return tail or head


def compact_process_lines(
    snapshot: Snapshot,
    width: int,
    rows: int,
    sort_field: str,
) -> list[str]:
    lines = [truncate(f"top {SORT_LABELS[sort_field]}", width)]
    for process in sort_processes(snapshot.processes, sort_field)[:rows]:
        gpu_pct = "-" if process.gpu_pct is None else f"{process.gpu_pct:.0f}"
        row = (
            f"{process.pid} "
            f"c{process.cpu_pct:.0f} "
            f"r{process.ram_pct:.0f} "
            f"g{gpu_pct} "
            f"{compact_process_name(process.name)}"
        )
        lines.append(truncate(row, width))
    return lines


def render_process_table(snapshot: Snapshot, width: int, rows: int, sort_field: str) -> list[str]:
    header = (
        f"{'PID':>7} {'IMPACT':>6} {'CPU%':>7} {'RAM%':>7} {'GPU%':>7} "
        f"{'VRAM':>7} {'RSS':>8} Command"
    )
    lines = [truncate(header, width), truncate("-" * min(width, len(header)), width)]

    for process in sort_processes(snapshot.processes, sort_field)[:rows]:
        gpu_pct = f"{process.gpu_pct:.1f}" if process.gpu_pct is not None else "-"
        row = (
            f"{process.pid:>7} "
            f"{process.score:>6.1f} "
            f"{process.cpu_pct:>7.1f} "
            f"{process.ram_pct:>7.1f} "
            f"{gpu_pct:>7} "
            f"{format_mib_short(process.vram_mib):>7} "
            f"{format_bytes_short(process.rss_bytes):>8} "
            f"{process.name}"
        )
        lines.append(truncate(row, width))

    if len(lines) == 2:
        lines.append("No active processes crossed the display threshold.")

    return lines


def render_snapshot(
    snapshot: Snapshot,
    width: int,
    history: dict[str, deque[float]],
    top_n: int,
    sort_field: str,
) -> list[str]:
    gauge_width = 22 if width >= 110 else 18 if width >= 90 else 14
    trace_width = gauge_width

    cpu_util_status, _ = classify_utilization(snapshot.cpu_util_pct)
    cpu_temp_status, _ = classify_cpu_temp(snapshot.cpu_temps.main_temp_c)
    gpu_util_status, _ = classify_utilization(snapshot.gpu.util_pct)
    gpu_mem_status, _ = classify_utilization(snapshot.gpu.mem_pct, MEM_WARN_PCT, MEM_DANGER_PCT)
    gpu_fan_status, _ = classify_utilization(snapshot.gpu.fan_pct)
    gpu_power_status, _ = classify_utilization(gpu_power_pct(snapshot.gpu))
    gpu_temp_status, _ = classify_gpu_temp(
        snapshot.gpu.temp_c, snapshot.gpu.temp_limit_c, snapshot.gpu.thermal_slowdown
    )
    ram_status, _ = classify_utilization(snapshot.ram_pct, MEM_WARN_PCT, MEM_DANGER_PCT)

    cpu_temp_value = "n/a"
    if snapshot.cpu_temps.main_temp_c is not None:
        cpu_temp_value = f"{snapshot.cpu_temps.main_temp_c:5.1f}C"

    gpu_util_value = "n/a" if snapshot.gpu.util_pct is None else f"{snapshot.gpu.util_pct:5.1f}%"
    gpu_mem_value = "n/a" if snapshot.gpu.mem_pct is None else f"{snapshot.gpu.mem_pct:5.1f}%"
    gpu_fan_value = "n/a" if snapshot.gpu.fan_pct is None else f"{snapshot.gpu.fan_pct:5.1f}%"
    power_pct = gpu_power_pct(snapshot.gpu)
    gpu_power_value = "n/a" if power_pct is None else f"{power_pct:5.1f}%"
    gpu_temp_value = "n/a" if snapshot.gpu.temp_c is None else f"{snapshot.gpu.temp_c:5.1f}C"

    gpu_mem_detail = ""
    if snapshot.gpu.mem_used_mib is not None and snapshot.gpu.mem_total_mib:
        gpu_mem_detail = (
            f"{snapshot.gpu.mem_used_mib:.0f}/{snapshot.gpu.mem_total_mib:.0f} MiB"
        )
    gpu_fan_detail = ""
    if snapshot.gpu.fan_pct is not None:
        gpu_fan_detail = format_pct_short(snapshot.gpu.fan_pct)
    gpu_power_detail = ""
    if snapshot.gpu.power_draw_w is not None:
        gpu_power_detail = format_watts_short(snapshot.gpu.power_draw_w)
        if snapshot.gpu.power_limit_w is not None:
            gpu_power_detail += f"/{format_watts_short(snapshot.gpu.power_limit_w)}"

    lines = [
        truncate(
            f"pcdiag  {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"q quit  r reset history  left/right sort  top {top_n}",
            width,
        ),
        truncate("=" * width, width),
        render_metric_line(
            width,
            "CPU Util",
            f"{snapshot.cpu_util_pct:5.1f}%",
            gauge(snapshot.cpu_util_pct, 100.0, gauge_width),
            sparkline(history["cpu_util"], 100.0, trace_width),
            cpu_util_status,
        ),
        render_metric_line(
            width,
            "RAM Used",
            f"{snapshot.ram_pct:5.1f}%",
            gauge(snapshot.ram_pct, 100.0, gauge_width),
            sparkline(history["ram"], 100.0, trace_width),
            ram_status,
            f"{gib_from_bytes(snapshot.ram_used_bytes):.1f}/{gib_from_bytes(snapshot.ram_total_bytes):.1f} GiB",
        ),
        "",
        render_metric_line(
            width,
            "CPU Temp",
            cpu_temp_value,
            gauge(snapshot.cpu_temps.main_temp_c, 100.0, gauge_width),
            sparkline(history["cpu_temp"], 100.0, trace_width),
            cpu_temp_status,
            snapshot.cpu_temps.detail,
        ),
        "",
        section_divider_text(width, ""),
        "",
        render_metric_line(
            width,
            "GPU Util",
            gpu_util_value,
            gauge(snapshot.gpu.util_pct, 100.0, gauge_width),
            sparkline(history["gpu_util"], 100.0, trace_width),
            gpu_util_status,
            snapshot.gpu.label,
        ),
        render_metric_line(
            width,
            "GPU RAM",
            gpu_mem_value,
            gauge(snapshot.gpu.mem_pct, 100.0, gauge_width),
            sparkline(history["gpu_mem"], 100.0, trace_width),
            gpu_mem_status,
            gpu_mem_detail,
        ),
        "",
        render_metric_line(
            width,
            "GPU Temp",
            gpu_temp_value,
            gauge(snapshot.gpu.temp_c, 100.0, gauge_width),
            sparkline(history["gpu_temp"], 100.0, trace_width),
            gpu_temp_status,
        ),
        render_metric_line(
            width,
            "GPU Fan",
            gpu_fan_value,
            gauge(snapshot.gpu.fan_pct, 100.0, gauge_width),
            sparkline(history["gpu_fan"], 100.0, trace_width),
            gpu_fan_status,
            gpu_fan_detail,
        ),
        render_metric_line(
            width,
            "GPU Power",
            gpu_power_value,
            gauge(power_pct, 100.0, gauge_width),
            sparkline(history["gpu_power"], 100.0, trace_width),
            gpu_power_status,
            gpu_power_detail,
        ),
        "",
        f"Top Processes  sort: {SORT_LABELS[sort_field]}  use left/right arrows",
    ]
    lines.extend(render_process_table(snapshot, width, top_n, sort_field))
    return lines


def render_snapshot_compact(
    snapshot: Snapshot,
    width: int,
    history: dict[str, deque[float]],
    top_n: int,
    sort_field: str,
) -> list[str]:
    gauge_width = compact_bar_width(width)
    trace_width = gauge_width

    cpu_util_status, _ = classify_utilization(snapshot.cpu_util_pct)
    cpu_temp_status, _ = classify_cpu_temp(snapshot.cpu_temps.main_temp_c)
    gpu_util_status, _ = classify_utilization(snapshot.gpu.util_pct)
    gpu_mem_status, _ = classify_utilization(snapshot.gpu.mem_pct, MEM_WARN_PCT, MEM_DANGER_PCT)
    gpu_fan_status, _ = classify_utilization(snapshot.gpu.fan_pct)
    power_pct = gpu_power_pct(snapshot.gpu)
    gpu_power_status, _ = classify_utilization(power_pct)
    gpu_temp_status, _ = classify_gpu_temp(
        snapshot.gpu.temp_c, snapshot.gpu.temp_limit_c, snapshot.gpu.thermal_slowdown
    )
    ram_status, _ = classify_utilization(snapshot.ram_pct, MEM_WARN_PCT, MEM_DANGER_PCT)

    gpu_ram_detail = ""
    if snapshot.gpu.mem_used_mib is not None and snapshot.gpu.mem_total_mib:
        gpu_ram_detail = f"{snapshot.gpu.mem_used_mib / 1024:.1f}/{snapshot.gpu.mem_total_mib / 1024:.0f}G"
    gpu_power_detail = ""
    if snapshot.gpu.power_draw_w is not None:
        gpu_power_detail = format_watts_short(snapshot.gpu.power_draw_w)
        if snapshot.gpu.power_limit_w is not None:
            gpu_power_detail += f"/{format_watts_short(snapshot.gpu.power_limit_w)}"

    lines = [
        truncate(
            f"pcdiag {snapshot.timestamp.strftime('%H:%M:%S')}  q quit  <> sort {SORT_LABELS[sort_field]}",
            width,
        ),
        truncate("=" * width, width),
        render_metric_line(
            width,
            "CPU",
            f"{snapshot.cpu_util_pct:4.0f}%",
            gauge(snapshot.cpu_util_pct, 100.0, gauge_width),
            sparkline(history["cpu_util"], 100.0, trace_width),
            compact_status_text(cpu_util_status),
            label_width=4,
            value_width=6,
        ),
        render_metric_line(
            width,
            "RAM",
            f"{snapshot.ram_pct:4.0f}%",
            gauge(snapshot.ram_pct, 100.0, gauge_width),
            sparkline(history["ram"], 100.0, trace_width),
            compact_status_text(ram_status),
            f"{gib_from_bytes(snapshot.ram_used_bytes):.1f}G",
            label_width=4,
            value_width=6,
        ),
        render_metric_line(
            width,
            "CTMP",
            "n/a" if snapshot.cpu_temps.main_temp_c is None else f"{snapshot.cpu_temps.main_temp_c:4.0f}C",
            gauge(snapshot.cpu_temps.main_temp_c, 100.0, gauge_width),
            sparkline(history["cpu_temp"], 100.0, trace_width),
            compact_status_text(cpu_temp_status),
            label_width=4,
            value_width=6,
        ),
        section_divider_text(width, ""),
        render_metric_line(
            width,
            "GPU",
            "n/a" if snapshot.gpu.util_pct is None else f"{snapshot.gpu.util_pct:4.0f}%",
            gauge(snapshot.gpu.util_pct, 100.0, gauge_width),
            sparkline(history["gpu_util"], 100.0, trace_width),
            compact_status_text(gpu_util_status),
            label_width=4,
            value_width=6,
        ),
        render_metric_line(
            width,
            "GRAM",
            "n/a" if snapshot.gpu.mem_pct is None else f"{snapshot.gpu.mem_pct:4.0f}%",
            gauge(snapshot.gpu.mem_pct, 100.0, gauge_width),
            sparkline(history["gpu_mem"], 100.0, trace_width),
            compact_status_text(gpu_mem_status),
            gpu_ram_detail,
            label_width=4,
            value_width=6,
        ),
        render_metric_line(
            width,
            "GTMP",
            "n/a" if snapshot.gpu.temp_c is None else f"{snapshot.gpu.temp_c:4.0f}C",
            gauge(snapshot.gpu.temp_c, 100.0, gauge_width),
            sparkline(history["gpu_temp"], 100.0, trace_width),
            compact_status_text(gpu_temp_status),
            label_width=4,
            value_width=6,
        ),
        render_metric_line(
            width,
            "FAN",
            "n/a" if snapshot.gpu.fan_pct is None else f"{snapshot.gpu.fan_pct:4.0f}%",
            gauge(snapshot.gpu.fan_pct, 100.0, gauge_width),
            sparkline(history["gpu_fan"], 100.0, trace_width),
            compact_status_text(gpu_fan_status),
            label_width=4,
            value_width=6,
        ),
        render_metric_line(
            width,
            "PWR",
            "n/a" if power_pct is None else f"{power_pct:4.0f}%",
            gauge(power_pct, 100.0, gauge_width),
            sparkline(history["gpu_power"], 100.0, trace_width),
            compact_status_text(gpu_power_status),
            gpu_power_detail,
            label_width=4,
            value_width=6,
        ),
    ]

    if width >= 34:
        lines.append("")
        lines.extend(compact_process_lines(snapshot, width, min(top_n, 3), sort_field))
    return lines


def snapshot_to_text(
    snapshot: Snapshot,
    width: int,
    history: dict[str, deque[float]],
    top_n: int,
    sort_field: str,
) -> str:
    if width < 88:
        return "\n".join(render_snapshot_compact(snapshot, width, history, top_n, sort_field))
    return "\n".join(render_snapshot(snapshot, width, history, top_n, sort_field))


def run_snapshot_mode(monitor: Monitor, interval: float, top_n: int) -> int:
    time.sleep(interval)
    snapshot = monitor.collect()
    width = shutil.get_terminal_size((120, 40)).columns
    print(snapshot_to_text(snapshot, width, monitor.history, top_n, "impact"))
    return 0


def color_attr(level: str, *, bold: bool = False) -> int:
    attr = curses.A_BOLD if bold else 0
    if not curses.has_colors():
        return attr
    if level == "good":
        return attr | curses.color_pair(PAIR_GOOD)
    if level == "warn":
        return attr | curses.color_pair(PAIR_WARN)
    if level == "danger":
        return attr | curses.color_pair(PAIR_DANGER)
    if level == "accent":
        return attr | curses.color_pair(PAIR_ACCENT)
    return attr


def section_divider_segments(width: int, label: str) -> list[tuple[str, int]]:
    text = section_divider_text(width, label)
    if not label:
        return [(text, 0)]
    label_text = f" {label} "
    start = text.find(label_text)
    if start < 0:
        return [(text, color_attr("accent", bold=True))]
    end = start + len(label_text)
    segments: list[tuple[str, int]] = []
    if start > 0:
        segments.append((text[:start], 0))
    segments.append((label_text, color_attr("accent", bold=True)))
    if end < len(text):
        segments.append((text[end:], 0))
    return segments


def init_colors() -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(PAIR_ACCENT, curses.COLOR_CYAN, -1)
    curses.init_pair(PAIR_GOOD, curses.COLOR_GREEN, -1)
    curses.init_pair(PAIR_WARN, curses.COLOR_YELLOW, -1)
    curses.init_pair(PAIR_DANGER, curses.COLOR_RED, -1)


def draw_segments(
    window: curses.window,
    y: int,
    width: int,
    segments: list[tuple[str, int]],
) -> None:
    x = 0
    for text, attr in segments:
        if x >= width or not text:
            break
        available = width - x
        window.addnstr(y, x, text, available, attr)
        x += min(len(text), available)


def metric_segments(
    label: str,
    value: str,
    bar: str,
    trace: str,
    trace_levels: list[str] | None,
    level: str,
    status: str = "",
    detail: str = "",
    *,
    label_width: int = METRIC_LABEL_WIDTH,
    value_width: int = METRIC_VALUE_WIDTH,
) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = [
        (f"{label:<{label_width}}", 0),
        (" ", 0),
        (f"{value:>{value_width}}", color_attr(level, bold=True)),
        (" ", 0),
        ("[", 0),
        (bar, color_attr(level, bold=True)),
        ("]", 0),
    ]
    if trace:
        segments.append((" ", 0))
        if trace_levels:
            for index, char in enumerate(trace):
                char_level = trace_levels[index] if index < len(trace_levels) else level
                segments.append((char, color_attr(char_level)))
        else:
            segments.append((trace, color_attr(level)))
    if status:
        segments.extend([(" ", 0), (status.upper(), color_attr(level, bold=True))])
    if detail:
        segments.extend([(" ", 0), (detail, 0)])
    return segments


def draw_process_header(
    window: curses.window,
    y: int,
    width: int,
    sort_field: str,
) -> None:
    columns = [
        ("pid", "PID", 7),
        ("impact", "IMPACT", 6),
        ("cpu_pct", "CPU%", 7),
        ("ram_pct", "RAM%", 7),
        ("gpu_pct", "GPU%", 7),
        ("vram_mib", "VRAM", 7),
        ("rss_bytes", "RSS", 8),
    ]
    x = 0
    for key, label, col_width in columns:
        text = f"{label:>{col_width}}"
        attr = curses.A_BOLD
        if key == sort_field:
            attr |= curses.A_REVERSE | color_attr("accent", bold=True)
        window.addnstr(y, x, text, max(0, width - x), attr)
        x += len(text)
        if x < width:
            window.addnstr(y, x, " ", 1)
            x += 1
    if x < width:
        window.addnstr(y, x, "Command", max(0, width - x), curses.A_BOLD)


def draw_process_rows(
    window: curses.window,
    start_y: int,
    width: int,
    height: int,
    processes: list[ProcessUsage],
    sort_field: str,
) -> None:
    visible_rows = max(0, height - start_y)
    for row_index, process in enumerate(processes[:visible_rows]):
        y = start_y + row_index
        x = 0
        row_columns = [
            ("pid", f"{process.pid:>7}"),
            ("impact", f"{process.score:>6.1f}"),
            ("cpu_pct", f"{process.cpu_pct:>7.1f}"),
            ("ram_pct", f"{process.ram_pct:>7.1f}"),
            ("gpu_pct", f"{process.gpu_pct:.1f}" if process.gpu_pct is not None else f"{'-':>7}"),
            ("vram_mib", f"{format_mib_short(process.vram_mib):>7}"),
            ("rss_bytes", f"{format_bytes_short(process.rss_bytes):>8}"),
        ]
        for key, text in row_columns:
            attr = color_attr("accent", bold=True) if key == sort_field else 0
            window.addnstr(y, x, text, max(0, width - x), attr)
            x += len(text)
            if x < width:
                window.addnstr(y, x, " ", 1)
                x += 1
        if x < width:
            window.addnstr(y, x, process.name, max(0, width - x))


def draw_dashboard(
    stdscr: curses.window,
    snapshot: Snapshot,
    monitor: Monitor,
    top_n: int,
    sort_field: str,
) -> None:
    height, width = stdscr.getmaxyx()
    draw_width = max(0, width - 1)
    gauge_width = 22 if draw_width >= 110 else 18 if draw_width >= 90 else 14
    trace_width = gauge_width

    cpu_util_status, cpu_util_level = classify_utilization(snapshot.cpu_util_pct)
    cpu_temp_status, cpu_temp_level = classify_cpu_temp(snapshot.cpu_temps.main_temp_c)
    gpu_util_status, gpu_util_level = classify_utilization(snapshot.gpu.util_pct)
    gpu_mem_status, gpu_mem_level = classify_utilization(
        snapshot.gpu.mem_pct, MEM_WARN_PCT, MEM_DANGER_PCT
    )
    gpu_fan_status, gpu_fan_level = classify_utilization(snapshot.gpu.fan_pct)
    power_pct = gpu_power_pct(snapshot.gpu)
    gpu_power_status, gpu_power_level = classify_utilization(power_pct)
    gpu_temp_status, gpu_temp_level = classify_gpu_temp(
        snapshot.gpu.temp_c, snapshot.gpu.temp_limit_c, snapshot.gpu.thermal_slowdown
    )
    ram_status, ram_level = classify_utilization(snapshot.ram_pct, MEM_WARN_PCT, MEM_DANGER_PCT)

    cpu_temp_value = "n/a"
    if snapshot.cpu_temps.main_temp_c is not None:
        cpu_temp_value = f"{snapshot.cpu_temps.main_temp_c:5.1f}C"

    gpu_util_value = "n/a" if snapshot.gpu.util_pct is None else f"{snapshot.gpu.util_pct:5.1f}%"
    gpu_mem_value = "n/a" if snapshot.gpu.mem_pct is None else f"{snapshot.gpu.mem_pct:5.1f}%"
    gpu_fan_value = "n/a" if snapshot.gpu.fan_pct is None else f"{snapshot.gpu.fan_pct:5.1f}%"
    gpu_power_value = "n/a" if power_pct is None else f"{power_pct:5.1f}%"
    gpu_temp_value = "n/a" if snapshot.gpu.temp_c is None else f"{snapshot.gpu.temp_c:5.1f}C"

    gpu_mem_detail = ""
    if snapshot.gpu.mem_used_mib is not None and snapshot.gpu.mem_total_mib:
        gpu_mem_detail = (
            f"{snapshot.gpu.mem_used_mib:.0f}/{snapshot.gpu.mem_total_mib:.0f} MiB"
        )
    gpu_fan_detail = ""
    if snapshot.gpu.fan_pct is not None:
        gpu_fan_detail = format_pct_short(snapshot.gpu.fan_pct)
    gpu_power_detail = ""
    if snapshot.gpu.power_draw_w is not None:
        gpu_power_detail = format_watts_short(snapshot.gpu.power_draw_w)
        if snapshot.gpu.power_limit_w is not None:
            gpu_power_detail += f"/{format_watts_short(snapshot.gpu.power_limit_w)}"

    cpu_util_trace_values = sparkline_values(monitor.history["cpu_util"], trace_width)
    cpu_temp_trace_values = sparkline_values(monitor.history["cpu_temp"], trace_width)
    gpu_util_trace_values = sparkline_values(monitor.history["gpu_util"], trace_width)
    gpu_mem_trace_values = sparkline_values(monitor.history["gpu_mem"], trace_width)
    gpu_fan_trace_values = sparkline_values(monitor.history["gpu_fan"], trace_width)
    gpu_power_trace_values = sparkline_values(monitor.history["gpu_power"], trace_width)
    gpu_temp_trace_values = sparkline_values(monitor.history["gpu_temp"], trace_width)
    ram_trace_values = sparkline_values(monitor.history["ram"], trace_width)

    rows: list[list[tuple[str, int]]] = [
        [
            ("pcdiag  ", color_attr("accent", bold=True)),
            (snapshot.timestamp.strftime("%Y-%m-%d %H:%M:%S"), 0),
            ("  q quit  r reset history  left/right sort  top ", 0),
            (str(top_n), color_attr("accent", bold=True)),
            ("  sort: ", 0),
            (SORT_LABELS[sort_field], color_attr("accent", bold=True)),
        ],
        [(("=" * draw_width) if draw_width > 0 else "", 0)],
        metric_segments(
            "CPU Util",
            f"{snapshot.cpu_util_pct:5.1f}%",
            gauge(snapshot.cpu_util_pct, 100.0, gauge_width),
            sparkline(cpu_util_trace_values, 100.0, trace_width),
            trace_levels_for_values(cpu_util_trace_values, "cpu_util"),
            cpu_util_level,
            cpu_util_status,
        ),
        metric_segments(
            "RAM Used",
            f"{snapshot.ram_pct:5.1f}%",
            gauge(snapshot.ram_pct, 100.0, gauge_width),
            sparkline(ram_trace_values, 100.0, trace_width),
            trace_levels_for_values(ram_trace_values, "ram"),
            ram_level,
            ram_status,
            f"{gib_from_bytes(snapshot.ram_used_bytes):.1f}/{gib_from_bytes(snapshot.ram_total_bytes):.1f} GiB",
        ),
        [],
        metric_segments(
            "CPU Temp",
            cpu_temp_value,
            gauge(snapshot.cpu_temps.main_temp_c, 100.0, gauge_width),
            sparkline(cpu_temp_trace_values, 100.0, trace_width),
            trace_levels_for_values(cpu_temp_trace_values, "cpu_temp"),
            cpu_temp_level,
            cpu_temp_status,
            snapshot.cpu_temps.detail,
        ),
        [],
        section_divider_segments(draw_width, ""),
        [],
        metric_segments(
            "GPU Util",
            gpu_util_value,
            gauge(snapshot.gpu.util_pct, 100.0, gauge_width),
            sparkline(gpu_util_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_util_trace_values, "gpu_util"),
            gpu_util_level,
            gpu_util_status,
            snapshot.gpu.label,
        ),
        metric_segments(
            "GPU RAM",
            gpu_mem_value,
            gauge(snapshot.gpu.mem_pct, 100.0, gauge_width),
            sparkline(gpu_mem_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_mem_trace_values, "gpu_mem"),
            gpu_mem_level,
            gpu_mem_status,
            gpu_mem_detail,
        ),
        [],
        metric_segments(
            "GPU Temp",
            gpu_temp_value,
            gauge(snapshot.gpu.temp_c, 100.0, gauge_width),
            sparkline(gpu_temp_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_temp_trace_values, "gpu_temp", snapshot.gpu),
            gpu_temp_level,
            gpu_temp_status,
        ),
        metric_segments(
            "GPU Fan",
            gpu_fan_value,
            gauge(snapshot.gpu.fan_pct, 100.0, gauge_width),
            sparkline(gpu_fan_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_fan_trace_values, "gpu_fan"),
            gpu_fan_level,
            gpu_fan_status,
            gpu_fan_detail,
        ),
        metric_segments(
            "GPU Power",
            gpu_power_value,
            gauge(power_pct, 100.0, gauge_width),
            sparkline(gpu_power_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_power_trace_values, "gpu_power"),
            gpu_power_level,
            gpu_power_status,
            gpu_power_detail,
        ),
        [],
        [
            ("Top Processes  sort: ", 0),
            (SORT_LABELS[sort_field], color_attr("accent", bold=True)),
            ("  use left/right arrows", 0),
        ],
    ]

    for line_index, segments in enumerate(rows):
        if line_index >= height - 1:
            return
        draw_segments(stdscr, line_index, draw_width, segments)

    table_y = len(rows)
    if table_y >= height - 2:
        return

    draw_process_header(stdscr, table_y, draw_width, sort_field)
    if table_y + 1 < height:
        stdscr.addnstr(table_y + 1, 0, "-" * draw_width, draw_width)

    sorted_process_list = sort_processes(snapshot.processes, sort_field)[:top_n]
    if not sorted_process_list and table_y + 2 < height:
        stdscr.addnstr(table_y + 2, 0, "No active processes crossed the display threshold.", draw_width)
        return

    draw_process_rows(stdscr, table_y + 2, draw_width, height - 1, sorted_process_list, sort_field)


def draw_compact_dashboard(
    stdscr: curses.window,
    snapshot: Snapshot,
    monitor: Monitor,
    top_n: int,
    sort_field: str,
) -> None:
    height, width = stdscr.getmaxyx()
    draw_width = max(0, width - 1)
    gauge_width = compact_bar_width(draw_width)
    trace_width = gauge_width

    cpu_util_status, cpu_util_level = classify_utilization(snapshot.cpu_util_pct)
    cpu_temp_status, cpu_temp_level = classify_cpu_temp(snapshot.cpu_temps.main_temp_c)
    gpu_util_status, gpu_util_level = classify_utilization(snapshot.gpu.util_pct)
    gpu_mem_status, gpu_mem_level = classify_utilization(
        snapshot.gpu.mem_pct, MEM_WARN_PCT, MEM_DANGER_PCT
    )
    gpu_fan_status, gpu_fan_level = classify_utilization(snapshot.gpu.fan_pct)
    power_pct = gpu_power_pct(snapshot.gpu)
    gpu_power_status, gpu_power_level = classify_utilization(power_pct)
    gpu_temp_status, gpu_temp_level = classify_gpu_temp(
        snapshot.gpu.temp_c, snapshot.gpu.temp_limit_c, snapshot.gpu.thermal_slowdown
    )
    ram_status, ram_level = classify_utilization(snapshot.ram_pct, MEM_WARN_PCT, MEM_DANGER_PCT)

    gpu_ram_detail = ""
    if snapshot.gpu.mem_used_mib is not None and snapshot.gpu.mem_total_mib:
        gpu_ram_detail = f"{snapshot.gpu.mem_used_mib / 1024:.1f}/{snapshot.gpu.mem_total_mib / 1024:.0f}G"
    gpu_power_detail = ""
    if snapshot.gpu.power_draw_w is not None:
        gpu_power_detail = format_watts_short(snapshot.gpu.power_draw_w)
        if snapshot.gpu.power_limit_w is not None:
            gpu_power_detail += f"/{format_watts_short(snapshot.gpu.power_limit_w)}"

    cpu_util_trace_values = sparkline_values(monitor.history["cpu_util"], trace_width)
    ram_trace_values = sparkline_values(monitor.history["ram"], trace_width)
    cpu_temp_trace_values = sparkline_values(monitor.history["cpu_temp"], trace_width)
    gpu_util_trace_values = sparkline_values(monitor.history["gpu_util"], trace_width)
    gpu_mem_trace_values = sparkline_values(monitor.history["gpu_mem"], trace_width)
    gpu_temp_trace_values = sparkline_values(monitor.history["gpu_temp"], trace_width)
    gpu_fan_trace_values = sparkline_values(monitor.history["gpu_fan"], trace_width)
    gpu_power_trace_values = sparkline_values(monitor.history["gpu_power"], trace_width)

    rows: list[list[tuple[str, int]]] = [
        [
            ("pcdiag ", color_attr("accent", bold=True)),
            (snapshot.timestamp.strftime("%H:%M:%S"), 0),
            ("  <> ", 0),
            (SORT_LABELS[sort_field], color_attr("accent", bold=True)),
            ("  q", 0),
        ],
        [(("=" * draw_width) if draw_width > 0 else "", 0)],
        metric_segments(
            "CPU",
            f"{snapshot.cpu_util_pct:4.0f}%",
            gauge(snapshot.cpu_util_pct, 100.0, gauge_width),
            sparkline(cpu_util_trace_values, 100.0, trace_width),
            trace_levels_for_values(cpu_util_trace_values, "cpu_util"),
            cpu_util_level,
            compact_status_text(cpu_util_status),
            label_width=4,
            value_width=6,
        ),
        metric_segments(
            "RAM",
            f"{snapshot.ram_pct:4.0f}%",
            gauge(snapshot.ram_pct, 100.0, gauge_width),
            sparkline(ram_trace_values, 100.0, trace_width),
            trace_levels_for_values(ram_trace_values, "ram"),
            ram_level,
            compact_status_text(ram_status),
            f"{gib_from_bytes(snapshot.ram_used_bytes):.1f}G",
            label_width=4,
            value_width=6,
        ),
        metric_segments(
            "CTMP",
            "n/a" if snapshot.cpu_temps.main_temp_c is None else f"{snapshot.cpu_temps.main_temp_c:4.0f}C",
            gauge(snapshot.cpu_temps.main_temp_c, 100.0, gauge_width),
            sparkline(cpu_temp_trace_values, 100.0, trace_width),
            trace_levels_for_values(cpu_temp_trace_values, "cpu_temp"),
            cpu_temp_level,
            compact_status_text(cpu_temp_status),
            label_width=4,
            value_width=6,
        ),
        section_divider_segments(draw_width, ""),
        metric_segments(
            "GPU",
            "n/a" if snapshot.gpu.util_pct is None else f"{snapshot.gpu.util_pct:4.0f}%",
            gauge(snapshot.gpu.util_pct, 100.0, gauge_width),
            sparkline(gpu_util_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_util_trace_values, "gpu_util"),
            gpu_util_level,
            compact_status_text(gpu_util_status),
            label_width=4,
            value_width=6,
        ),
        metric_segments(
            "GRAM",
            "n/a" if snapshot.gpu.mem_pct is None else f"{snapshot.gpu.mem_pct:4.0f}%",
            gauge(snapshot.gpu.mem_pct, 100.0, gauge_width),
            sparkline(gpu_mem_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_mem_trace_values, "gpu_mem"),
            gpu_mem_level,
            compact_status_text(gpu_mem_status),
            gpu_ram_detail,
            label_width=4,
            value_width=6,
        ),
        metric_segments(
            "GTMP",
            "n/a" if snapshot.gpu.temp_c is None else f"{snapshot.gpu.temp_c:4.0f}C",
            gauge(snapshot.gpu.temp_c, 100.0, gauge_width),
            sparkline(gpu_temp_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_temp_trace_values, "gpu_temp", snapshot.gpu),
            gpu_temp_level,
            compact_status_text(gpu_temp_status),
            label_width=4,
            value_width=6,
        ),
        metric_segments(
            "FAN",
            "n/a" if snapshot.gpu.fan_pct is None else f"{snapshot.gpu.fan_pct:4.0f}%",
            gauge(snapshot.gpu.fan_pct, 100.0, gauge_width),
            sparkline(gpu_fan_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_fan_trace_values, "gpu_fan"),
            gpu_fan_level,
            compact_status_text(gpu_fan_status),
            label_width=4,
            value_width=6,
        ),
        metric_segments(
            "PWR",
            "n/a" if power_pct is None else f"{power_pct:4.0f}%",
            gauge(power_pct, 100.0, gauge_width),
            sparkline(gpu_power_trace_values, 100.0, trace_width),
            trace_levels_for_values(gpu_power_trace_values, "gpu_power"),
            gpu_power_level,
            compact_status_text(gpu_power_status),
            gpu_power_detail,
            label_width=4,
            value_width=6,
        ),
    ]

    process_header_row = len(rows) + 1
    process_rows_available = max(0, height - process_header_row - 1)
    compact_top_n = min(top_n, 3, process_rows_available)
    if compact_top_n > 0 and draw_width >= 34:
        rows.append([])
        rows.append(
            [
                ("top ", 0),
                (SORT_LABELS[sort_field], color_attr("accent", bold=True)),
            ]
        )
        for line in compact_process_lines(snapshot, draw_width, compact_top_n, sort_field)[1:]:
            rows.append([(line, 0)])

    for line_index, segments in enumerate(rows):
        if line_index >= height - 1:
            return
        draw_segments(stdscr, line_index, draw_width, segments)


def run_tui(stdscr: curses.window, monitor: Monitor, interval: float, top_n: int) -> int:
    curses.curs_set(0)
    stdscr.nodelay(True)
    init_colors()

    snapshot: Snapshot | None = None
    next_refresh = time.monotonic()
    sort_index = 0

    while True:
        now = time.monotonic()
        if snapshot is None or now >= next_refresh:
            snapshot = monitor.collect()
            next_refresh = now + interval

        height, width = stdscr.getmaxyx()
        stdscr.erase()
        if height < 20 or width < 88:
            draw_compact_dashboard(stdscr, snapshot, monitor, top_n, SORT_FIELDS[sort_index])
        else:
            draw_dashboard(stdscr, snapshot, monitor, top_n, SORT_FIELDS[sort_index])
        stdscr.refresh()

        timeout_ms = max(0, int((next_refresh - time.monotonic()) * 1000))
        stdscr.timeout(min(timeout_ms, 250))
        key = stdscr.getch()
        if key == -1:
            continue
        if key in (ord("q"), ord("Q")):
            return 0
        if key in (ord("r"), ord("R")):
            monitor.reset_history()
            next_refresh = time.monotonic()
            continue
        if key in (curses.KEY_RIGHT, curses.KEY_DOWN):
            sort_index = (sort_index + 1) % len(SORT_FIELDS)
            continue
        if key in (curses.KEY_LEFT, curses.KEY_UP):
            sort_index = (sort_index - 1) % len(SORT_FIELDS)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight terminal dashboard for Ubuntu CPU/GPU/RAM monitoring."
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds.")
    parser.add_argument("--history", type=int, default=60, help="History samples to keep per metric.")
    parser.add_argument("--top", type=int, default=10, help="How many processes to show.")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Print one snapshot and exit instead of launching the interactive dashboard.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    interval = max(args.interval, 0.2)
    history = max(args.history, 8)
    top_n = max(args.top, 1)
    monitor = Monitor(history_size=history, top_n=top_n)

    if args.snapshot:
        return run_snapshot_mode(monitor, interval, top_n)

    try:
        return curses.wrapper(run_tui, monitor, interval, top_n)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
