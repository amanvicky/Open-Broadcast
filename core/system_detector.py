"""
OpenBroadcast — System Detection & Auto-Configuration

Detects hardware capabilities and auto-configures optimal settings.
Supports: 8GB-30GB+ RAM, 8GB-40GB+ disk, 2-16+ cores, no GPU to dedicated GPU.
Runs once at startup, takes <2 seconds.
"""

import os
import platform
import re
import shutil
import psutil
import numpy as np

try:
    import wmi as wmi_module
    HAS_WMI = True
except ImportError:
    HAS_WMI = False


def detect_system():
    """Detect all hardware and return comprehensive system info dict."""
    cpu_info = _detect_cpu()
    ram_info = _detect_ram()
    gpu_info = _detect_gpu()
    cameras = _detect_cameras()
    os_info = _detect_os()
    disk_free = _detect_disk()

    tier = _classify_tier(cpu_info, ram_info, gpu_info)
    config = _generate_config(tier, cpu_info, ram_info, cameras)

    return {
        "cpu": cpu_info,
        "ram": ram_info,
        "gpu": gpu_info,
        "cameras": cameras,
        "os": os_info,
        "disk_free_gb": disk_free,
        "tier": tier,
        "config": config,
    }


def _detect_cpu():
    """Detect CPU specifications."""
    brand = "Unknown"
    generation = 0
    has_avx2 = False
    has_avx512 = False

    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw", "Unknown")
        flags = info.get("flags", [])
        has_avx2 = "avx2" in flags
        has_avx512 = "avx512f" in flags
        generation = _parse_intel_generation(brand)
    except Exception:
        pass

    return {
        "brand": brand,
        "generation": generation,
        "physical_cores": psutil.cpu_count(logical=False) or 2,
        "logical_cores": psutil.cpu_count(logical=True) or 2,
        "has_avx2": has_avx2,
        "has_avx512": has_avx512,
    }


def _parse_intel_generation(brand):
    """Extract Intel generation from model string. e.g., 'i5-8250U' → 8"""
    match = re.search(r"i[3579]-(\d)", brand)
    if match:
        return int(match.group(1))
    return 0


def _detect_ram():
    """Detect RAM specifications."""
    mem = psutil.virtual_memory()
    speed = None

    if HAS_WMI:
        try:
            w = wmi_module.WMI()
            modules = w.Win32_PhysicalMemory()
            if modules:
                speed = modules[0].Speed
        except Exception:
            pass

    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "available_gb": round(mem.available / (1024**3), 1),
        "speed_mhz": speed,
    }


def _detect_gpu():
    """Detect GPU(s) — both dedicated and integrated."""
    result = {
        "has_dedicated": False,
        "has_integrated": False,
        "name": "None",
        "vram_mb": 0,
        "vendor": "None",
    }

    if not HAS_WMI:
        return result

    try:
        w = wmi_module.WMI()
        for vc in w.Win32_VideoController():
            name = (vc.Name or "").lower()
            adapter_ram = vc.AdapterRAM or 0
            vram_mb = adapter_ram // (1024 * 1024) if adapter_ram > 0 else 0

            if any(kw in name for kw in ["nvidia", "geforce", "quadro", "rtx", "gtx"]):
                result["has_dedicated"] = True
                result["vendor"] = "NVIDIA"
                result["name"] = vc.Name
                result["vram_mb"] = vram_mb
            elif any(kw in name for kw in ["amd", "radeon"]):
                if vram_mb > 512:
                    result["has_dedicated"] = True
                    result["vendor"] = "AMD"
                    result["name"] = vc.Name
                    result["vram_mb"] = vram_mb
                else:
                    result["has_integrated"] = True
                    if result["name"] == "None":
                        result["name"] = vc.Name
            elif "intel" in name:
                result["has_integrated"] = True
                result["vendor"] = "Intel"
                if result["name"] == "None":
                    result["name"] = vc.Name
    except Exception:
        pass

    return result


def _detect_cameras():
    """Detect available cameras."""
    import cv2

    cameras = []
    for i in range(6):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)

            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                cameras.append({
                    "index": i,
                    "name": f"Camera {i}",
                    "resolution": (w, h),
                    "max_fps": fps if fps > 0 else 30,
                })
                cap.release()
        except Exception:
            continue

    return cameras


def _detect_os():
    """Detect OS version."""
    if platform.system() == "Windows":
        try:
            win_ver = platform.platform()
            build = platform.version()
            match = re.search(r"build=(\d+)", win_ver)
            if match:
                build_num = int(match.group(1))
                os_name = "Windows 11" if build_num >= 22000 else "Windows 10"
            else:
                os_name = "Windows"
            return {"name": os_name, "version": platform.version()}
        except Exception:
            return {"name": "Windows", "version": "Unknown"}

    return {"name": platform.system(), "version": platform.release()}


def _detect_disk():
    """Detect available disk space."""
    try:
        usage = shutil.disk_usage("/")
        return round(usage.free / (1024**3), 1)
    except Exception:
        return 0


def _classify_tier(cpu, ram, gpu):
    """
    Classify system into a performance tier.

    Full RAM range: 8GB → 30GB+
    Full CPU range: 2 cores → 16+ cores

    | RAM      | Cores | GPU        | Tier        |
    |----------|-------|------------|-------------|
    | <4 GB    | ≤2    | None       | ULTRA_LOW   |
    | <8 GB    | ≤4    | Any        | LOW         |
    | 8 GB     | 4+    | Any        | MEDIUM      |
    | 8 GB     | ≤2    | Any        | LOW         |
    | 8-15 GB  | 4-6   | Any        | MEDIUM      |
    | 15-30 GB | 4-6   | Integrated | MEDIUM      |
    | 15-30 GB | 4-6   | Dedicated  | HIGH        |
    | 15-30 GB | 8+    | Any        | HIGH        |
    | 30+ GB   | 6+    | Dedicated  | ULTRA_HIGH  |
    | 30+ GB   | 8+    | Any        | ULTRA_HIGH  |
    | 30+ GB   | 12+   | Any        | ULTRA_HIGH  |
    """
    cores = cpu["physical_cores"]
    logical_cores = cpu["logical_cores"]
    ram_gb = ram["total_gb"]
    has_ded = gpu["has_dedicated"]
    has_int = gpu["has_integrated"]

    # ULTRA_LOW: 2 cores or less, <4GB RAM, no GPU
    if cores <= 2 and ram_gb < 4 and not has_ded:
        return "ULTRA_LOW"

    # LOW: <8GB RAM with weak CPU
    if ram_gb < 8:
        if cores <= 2:
            return "ULTRA_LOW"
        if cores <= 4 and not has_ded:
            return "LOW"
        return "LOW"

    # 8GB RAM — the most common configuration
    if 7.5 <= ram_gb < 9:
        if cores <= 2:
            return "LOW"  # 8GB but weak CPU
        if cores >= 4:
            return "MEDIUM"
        return "MEDIUM"

    # 8-15GB RAM
    if ram_gb >= 8 and ram_gb < 15:
        if cores >= 4:
            return "MEDIUM"
        return "LOW"

    # 15-30GB RAM
    if ram_gb >= 15 and ram_gb < 30:
        if cores >= 8:
            return "HIGH"
        if cores >= 4 and has_ded:
            return "HIGH"
        if cores >= 4:
            return "MEDIUM"
        return "MEDIUM"

    # 30GB+ RAM
    if ram_gb >= 30:
        if cores >= 12:
            return "ULTRA_HIGH"
        if cores >= 8:
            return "ULTRA_HIGH"
        if cores >= 6 and has_ded:
            return "ULTRA_HIGH"
        if cores >= 6:
            return "HIGH"
        if cores >= 4:
            return "HIGH"
        return "MEDIUM"

    # Fallback: MEDIUM
    return "MEDIUM"


def _generate_config(tier, cpu, ram, cameras):
    """Generate optimal configuration for the detected tier and available resources."""
    from config import PERFORMANCE_TIERS

    config = PERFORMANCE_TIERS.get(tier, PERFORMANCE_TIERS["MEDIUM"]).copy()

    ram_gb = ram["total_gb"]
    cores = cpu["physical_cores"]

    # ── RAM-specific overrides ──

    # 8GB: prefer geometric stability, cap FPS to save memory
    if 7.5 <= ram_gb < 9:
        if cores <= 2:
            config["mode"] = "geometric_with_smoothing"
            config["model_file"] = None
            config["neural_frame_skip"] = 0
            config["max_fps"] = 20
        else:
            config["mode"] = "hybrid_balanced"
            config["neural_frame_skip"] = 3
            config["max_fps"] = 25

    # 8-15GB: slightly more aggressive
    elif ram_gb >= 9 and ram_gb < 15:
        config["max_fps"] = min(30, config["max_fps"] + 2)

    # 15-30GB: can handle quality mode
    elif ram_gb >= 15 and ram_gb < 30:
        if config["mode"] == "hybrid_balanced" and cores >= 6:
            config["mode"] = "hybrid_quality"
            config["neural_frame_skip"] = 2
            config["max_fps"] = 30

    # 30GB+: maximum settings
    elif ram_gb >= 30:
        config["mode"] = "hybrid_quality"
        config["neural_frame_skip"] = 1
        config["inference_threads"] = min(6, cores)
        config["max_fps"] = 30
        config["feather_radius"] = 20
        config["correction_strength_default"] = 0.95

    # ── Camera overrides ──
    if cameras:
        best_cam = max(cameras, key=lambda c: c["resolution"][0] * c["resolution"][1])
        config["camera_index"] = best_cam["index"]

        cam_w, cam_h = best_cam["resolution"]
        proc_w, proc_h = config["processing_resolution"]
        if cam_w < proc_w:
            config["processing_resolution"] = [cam_w, cam_h]
            config["display_resolution"] = [cam_w, cam_h]

    # Leave 1 core free for UI
    config["inference_threads"] = min(
        config["inference_threads"],
        max(1, cpu["logical_cores"] - 1)
    )

    # ── Memory-aware adjustments ──
    available_gb = ram["available_gb"]
    min_free = 2.0 if ram_gb < 9 else 3.0 if ram_gb < 30 else 4.0

    if available_gb < min_free * 0.25:
        # Critical: <25% of target free — severe downgrade
        config["max_fps"] = max(10, config["max_fps"] - 10)
        if config["mode"].startswith("hybrid"):
            config["mode"] = "geometric_with_smoothing"
            config["model_file"] = None
    elif available_gb < min_free * 0.5:
        # Low memory — reduce FPS
        config["max_fps"] = max(15, config["max_fps"] - 5)

    config["tier"] = tier
    return config


def format_system_report(sys_info):
    """Generate human-readable system report."""
    cpu = sys_info["cpu"]
    ram = sys_info["ram"]
    gpu = sys_info["gpu"]
    os_info = sys_info["os"]
    config = sys_info["config"]

    cam_lines = ""
    for cam in sys_info["cameras"]:
        w, h = cam["resolution"]
        cam_lines += f"  Camera {cam['index']}: {w}x{h} @ {cam['max_fps']:.0f} FPS\n"

    gpu_status = f"{gpu['name']} ({gpu['vram_mb']} MB)" if gpu["has_dedicated"] else (
        f"{gpu['name']} (Integrated)" if gpu["has_integrated"] else "None"
    )

    # RAM category note
    ram_gb = ram['total_gb']
    if ram_gb < 4:
        ram_notes = "\n  ⚠️ Low RAM — minimal mode recommended"
    elif 7.5 <= ram_gb < 9:
        ram_notes = "\n  ⚡ 8GB RAM — optimized for this configuration"
    elif ram_gb >= 15 and ram_gb < 30:
        ram_notes = "\n  💪 Good RAM — quality mode available"
    elif ram_gb >= 30:
        ram_notes = "\n  🚀 High RAM — maximum quality enabled"
    else:
        ram_notes = ""

    report = f"""
{'='*55}
  OPENBROADCAST — SYSTEM SPECIFICATION
{'='*55}

  CPU
  ├─ Model:      {cpu['brand']}
  ├─ Cores:      {cpu['physical_cores']} physical / {cpu['logical_cores']} logical
  ├─ AVX2:       {'Yes' if cpu['has_avx2'] else 'No'}
  └─ AVX-512:    {'Yes' if cpu['has_avx512'] else 'No'}

  RAM
  ├─ Total:      {ram['total_gb']} GB{ram_notes}
  └─ Available:  {ram['available_gb']} GB

  GPU
  └─ {gpu_status}

  OS
  └─ {os_info['name']} ({os_info['version']})

  Camera(s)
{cam_lines if cam_lines else '  None detected'}

  Disk Free:    {sys_info['disk_free_gb']} GB

{'='*55}
  PERFORMANCE TIER: {config['tier']}
  MODE: {config['mode']}
  RESOLUTION: {config['processing_resolution'][0]}x{config['processing_resolution'][1]}
  TARGET FPS: {config['max_fps']}
  DESCRIPTION: {config['description']}
{'='*55}
"""
    return report
