import re
import csv
import sys
import os

args = sys.argv

if len(args) < 1:
    print("Usage: python3 data_preprocessing.py <dir name>")
    sys.exit()

input_filename = args[1] + "/log/result_serverlog.csv"
output_filename = args[1] + "/result_server.csv"


receives = []
failures = []
times = [0]
p_rate = []
timeout = 60 #[s] Time out の時間
round_time = 30 #[s]実測値
# Read the content from the input CSV file
with open(input_filename, "r") as infile:
    content = infile.read()
    receved_matches = re.findall(r"evaluate_round \d+ received (\d+)", content)
    failures_matches = re.findall(r"evaluate_round \d+ received \d+ results and (\d+) failures", content)

    for match in receved_matches:
        receives.append(int(match))

    for match in failures_matches:
        failures.append(int(match))

print(receives)

# Extract losses_distributed and metrics_distributed values
losses_pattern = r"losses_distributed\s+\[((?:\s*\(\s*\d+\s*,\s*[\d.]+\s*\)\s*,?)+)]"
metrics_pattern = r"metrics_distributed\s+{.*?:\s+\[((?:\s*\(\s*\d+\s*,\s*[\d.]+\s*\)\s*,?)+)]"
losses_matches = re.findall(losses_pattern, content)
metrics_matches = re.findall(metrics_pattern, content)

# Get losses and accuracies as lists
losses = [float(x[1]) for match in losses_matches for x in re.findall(r"\(\s*(\d+)\s*,\s+([\d.]+)\s*\)", match)]
accuracies = [float(x[1]) for match in metrics_matches for x in re.findall(r"\(\s*(\d+)\s*,\s+([\d.]+)\s*\)", match)]

print(len(receives))

for i in range(len(receives)):
    p_rate.append((receives[i]-failures[i])/receives[i]*100)
    print(times)
    if failures[i] == 0:
        times.append(round_time+times[i])
    else:
        times.append(timeout+times[i])
print(times)
times.pop(0)


# Save to a new CSV file
with open(output_filename, "w", newline="") as csvfile:
    fieldnames = ["Round", "Sim Time [s]", "Loss", "Accuracy", "receved", "failure", "p_rate [%]"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i, (time, loss,  accuracy, receive, failure, rate) in enumerate(zip(times, losses, accuracies, receives, failures, p_rate), start=1):
        writer.writerow({"Round": i, "Sim Time [s]": time,"Loss": loss, "Accuracy": accuracy, "receved": receive, "failure": failure, "p_rate [%]": rate})

print("新しいCSVファイルにデータが保存されました。")

