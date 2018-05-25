def main():
    for i in range(1000000):
        continue

import time
start_time = time.time()
main()

t = round(time.time() - start_time, 6)

print("--- %s seconds ---" % t)


