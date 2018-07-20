from concurrent.futures import ThreadPoolExecutor
import threading
import time

def task(n):
    print('{}: sleeping {}'.format(
        threading.current_thread().name,
        n)
    )
    time.sleep(n / 10)
    print('{}: done with {}'.format(
        threading.current_thread().name,
        n)
    )
    return n / 10


with ThreadPoolExecutor(max_workers=4) as executor:
    future = executor.map(task, range(5, 0, -1))
    print(future)