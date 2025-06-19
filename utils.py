# utils.py
import time

# ----- @profile Speed Trackter -----
def profile(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        # Bar settings
        unit = 0.05  # seconds per '='
        max_len = 20  # max number of '=' in bar
        num_eq = min(int(duration / unit), max_len)
        num_dash = max_len - num_eq

        # ANSI color for yellow: \033[33m
        bar = f"\033[33m{'=' * num_eq}\033[0m{'-' * num_dash}"

        print(f"[{bar}] \033[1;33m⏲ {func.__name__}\033[0m took {duration:.2f} seconds")
        return result
    return wrapper

def get_cpu_temp():
    if is_running_on_raspberry_pi():
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.readline()
        return int(temp_str) / 1000.0
    else: 
        return int(10)

def wait_for_cooldown(threshold=74.0, check_interval=3):
    temp = get_cpu_temp()
    while temp > threshold:
        print(f"CPU Temperature {temp:.2f}°C is above threshold {threshold}°C. Waiting...")
        time.sleep(check_interval)
        temp = get_cpu_temp()
    print(f"CPU Temperature {temp:.2f}°C is below threshold. Continuing...")
    
def is_running_on_raspberry_pi():
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                return "Raspberry Pi" in cpuinfo
        except FileNotFoundError:
            return False
