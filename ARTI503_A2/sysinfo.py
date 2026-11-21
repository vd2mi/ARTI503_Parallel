import os
import platform
import subprocess
import sys

UNKNOWN = "unknown"


def _run(cmd, timeout=15):
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def _int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _windows():
    info = {}
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        info["cpu_model"] = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        info["base_freq_mhz"] = _int(winreg.QueryValueEx(key, "~MHz")[0])
    except Exception:
        pass

    ps = _run([
        "powershell", "-NoProfile", "-NonInteractive", "-Command",
        "$p = Get-CimInstance Win32_Processor; "
        "\"$(($p | Measure-Object NumberOfCores -Sum).Sum),"
        "$(($p | Measure-Object MaxClockSpeed -Maximum).Maximum)\"",
    ], timeout=30)
    if "," in ps:
        cores, mhz = ps.split(",", 1)
        info["physical_cores"] = _int(cores)
        info["max_freq_mhz"] = _int(mhz)

    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            info["ram_gb"] = round(stat.ullTotalPhys / 1024 ** 3, 1)

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        power = SYSTEM_POWER_STATUS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
            info["on_ac_power"] = {0: False, 1: True}.get(power.ACLineStatus, UNKNOWN)
    except Exception:
        pass
    return info


def _mac():
    def sysctl(name):
        return _run(["sysctl", "-n", name])

    info = {
        "cpu_model": sysctl("machdep.cpu.brand_string") or None,
        "physical_cores": _int(sysctl("hw.physicalcpu")),
        "ram_gb": None,
    }
    mem = _int(sysctl("hw.memsize"))
    if mem:
        info["ram_gb"] = round(mem / 1024 ** 3, 1)
    freq = _int(sysctl("hw.cpufrequency_max"))
    if freq:
        info["max_freq_mhz"] = freq // 1000000
    perf = _int(sysctl("hw.perflevel0.physicalcpu"))
    eff = _int(sysctl("hw.perflevel1.physicalcpu"))
    if perf:
        info["performance_cores"] = perf
    if eff:
        info["efficiency_cores"] = eff
    batt = _run(["pmset", "-g", "batt"])
    if batt:
        info["on_ac_power"] = "AC Power" in batt
    return info


def _linux():
    info = {}
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            cpuinfo = f.read()
        cores = set()
        phys = core = None
        for line in cpuinfo.splitlines():
            if ":" not in line:
                if phys is not None or core is not None:
                    cores.add((phys, core))
                phys = core = None
                continue
            k, v = [s.strip() for s in line.split(":", 1)]
            if k == "model name" and "cpu_model" not in info:
                info["cpu_model"] = v
            elif k == "physical id":
                phys = v
            elif k == "core id":
                core = v
        if phys is not None or core is not None:
            cores.add((phys, core))
        if cores and None not in {c[1] for c in cores}:
            info["physical_cores"] = len(cores)
    except Exception:
        pass
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
            info["max_freq_mhz"] = int(f.read().strip()) // 1000
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 ** 2, 1)
                    break
    except Exception:
        pass
    try:
        base = "/sys/class/power_supply"
        for name in os.listdir(base):
            path = os.path.join(base, name, "online")
            if name.upper().startswith(("AC", "ADP")) and os.path.exists(path):
                with open(path) as f:
                    info["on_ac_power"] = f.read().strip() == "1"
                break
    except Exception:
        pass
    return info


def collect():
    system = platform.system()
    try:
        if system == "Windows":
            extra = _windows()
        elif system == "Darwin":
            extra = _mac()
        else:
            extra = _linux()
    except Exception:
        extra = {}

    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = UNKNOWN

    info = {
        "os": "{} {}".format(system, platform.release()),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "cpu_model": extra.get("cpu_model") or platform.processor() or UNKNOWN,
        "physical_cores": extra.get("physical_cores") or UNKNOWN,
        "logical_cores": os.cpu_count() or UNKNOWN,
        "max_freq_mhz": extra.get("max_freq_mhz") or extra.get("base_freq_mhz") or UNKNOWN,
        "ram_gb": extra.get("ram_gb") or UNKNOWN,
        "on_ac_power": extra.get("on_ac_power", UNKNOWN),
        "python": "{} {}".format(platform.python_implementation(), platform.python_version()),
        "numpy": numpy_version,
    }
    for k in ("performance_cores", "efficiency_cores"):
        if k in extra:
            info[k] = extra[k]
    return info


if __name__ == "__main__":
    for k, v in collect().items():
        print("{:<18} {}".format(k, v))
    sys.exit(0)
