import time
import random
import csv

header = ['Time'] + [f'Column_{i}' for i in range(1, 17)]
filename = 'data.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    start_time = time.time()
    current_time = 0

    while True:
        row = [current_time] + [random.randint(0, 1024) for _ in range(16)]
        writer.writerow(row)
        csvfile.flush()  # Ensure data is written to the file immediately

        time.sleep(0.5)  # Wait for 0.5 seconds
        current_time += 0.5
