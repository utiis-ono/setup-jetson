import argparse
import sys
import subprocess
import os
import time


def open_terminal_with_command(command):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == 'posix':  # POSIX (Linux, macOS) system
        script = f"""
        tell application "iTerm"
            activate
            create window with default profile
            tell current session of current window
                write text "cd {current_dir}"
                write text "conda activate flower"
                write text "{command}"
            end tell
        end tell
        """
        subprocess.run(["osascript", "-e", script])
    elif os.name == 'nt':  # Windows system
        subprocess.Popen(["start", "cmd", "/k", f"cd {current_dir} && {command}"], shell=True)


parser = argparse.ArgumentParser(description='CIFAR-100 selective training')
parser.add_argument('--rounds', default=3, type=int, help='Decide number of rounds :default to 3')
parser.add_argument('--timeout', default=60, type=int, help='Decide timeout[s] :default to 60')
parser.add_argument('--roundtime', default=35, type=int, help='Decide roundtime[s] :default to 35')
parser.add_argument('--size', default=1, type=int, help='Number of clients :default to 1')
parser.add_argument('--pretrained_weights', default="None", type=str, help='Path to .pt file with pretrained weights :default to "None"')
parser.add_argument('--method', default="std", type=str, help='Select method type std or my :default to "std"')
parser.add_argument('--select_train', default=[0,1,2,3,4], nargs='+', type=int, help='Select classes to include from 0 to 4 ex) "--select_train 0 1 2" :default to 0 1 3 4 4')
parser.add_argument('--select_test', default=[0,1,2,3,4], nargs='+', type=int, help='Select classes to include from 0 to 4 ex) "--select_test 2 3 4" : default to 0 1 2 3 4')
args = parser.parse_args()

time_out = int(args.timeout) #サーバのタイムアウト[s]
round_time = int(args.roundtime) #[s]実測値

if time_out < round_time:
    print("time outをround timeよりも短く設定することはできません")
    sys.exit(1)

#if len(sys.argv) != 6:
    #print("Usage: python3 run_sim.py [1]<round> [2]<time out[s]> [3]<round time[s]> [4]<number of client> [5]<my or std>")
    #print("Usage: python3 server.py [1]<dir name> [2]<rounds> [3]<time out[s]> [4]<round time[s]>")
    #print("Usage: python3 client_sim2.py [1]<dir name> [5]<number of clients> [4]<round time[s]> [6]<model>")
    #sys.exit(1)

train_list = '-'.join(map(str, args.select_train))
test_list = '-'.join(map(str, args.select_test))
dirpath = f"{args.method}-{args.rounds}round-{args.timeout}timeout/train{train_list}/test{test_list}"
dirname = os.path.join("result/CIFAR-100",dirpath,"log")
if not os.path.exists(dirname):
    os.makedirs(dirname)

iterations = args.size

processes = []

#start server
command =  f"python3 server_cifar100.py --dir_name {dirpath} --rounds {args.rounds} --timeout {args.timeout} --roundtime {args.roundtime} --pretrained_weights {args.pretrained_weights} 2>&1 | tee {dirname}/result_serverlog.csv"
open_terminal_with_command(command)
time.sleep(3)

#start client
train_arg = ' '.join(map(str, args.select_train))
test_arg = ' '.join(map(str, args.select_test))
for i in range(iterations):
    command = f"python3 client_sim4.py --dir_name {args.method}-{args.rounds}round-{args.timeout}timeout --node_id {str(i)} --timeout {args.timeout} --roundtime {args.roundtime} --method {args.method} --select_train {train_arg} --select_test {test_arg} 2>&1 | tee {dirname}/result_clientlog-{str(i)}.csv"
    open_terminal_with_command(command)

print("Please check the iTerm2 windows for the output.")

