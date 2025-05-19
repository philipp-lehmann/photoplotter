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