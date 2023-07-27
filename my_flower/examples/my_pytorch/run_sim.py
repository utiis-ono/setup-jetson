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

if len(sys.argv) != 6:
    print("Usage: python3 run_sim.py [1]<round> [2]<time out[s]> [3]<round time[s]> [4]<number of client> [5]<my or std>")
    #print("Usage: python3 server.py [1]<dir name> [2]<rounds> [3]<time out[s]> [4]<round time[s]>")
    #print("Usage: python3 client_sim2.py [1]<dir name> [5]<number of clients> [4]<round time[s]> [6]<model>")
    sys.exit(1)

#dirname = 'result/' + sys.argv[1] + "/log"
dirpath = f"{sys.argv[5]}-{sys.argv[1]}round-{sys.argv[2]}timeout"
dirname = os.path.join("result/CIFAR-10",dirpath,"log")
if not os.path.exists(dirname):
    os.makedirs(dirname)

iterations = int(sys.argv[4])

processes = []

#start server
#command =  f"python3 server.py {sys.argv[1]}  {sys.argv[2]} {sys.argv[3]} {sys.argv[4]} 2>&1 | tee {dirname}/result_serverlog.csv"
command =  f"python3 server.py CIFAR-10/{dirpath}  {sys.argv[1]} {sys.argv[2]} {sys.argv[3]} 2>&1 | tee {dirname}/result_serverlog.csv"
open_terminal_with_command(command)
time.sleep(3)
#start client
for i in range(iterations):
    #command = f"python3 client_sim2.py {sys.argv[1]} {str(i)} {sys.argv[4]} {sys.argv[6]} 2>&1 | tee {dirname}/result_clientlog-{str(i)}.csv"
    command = f"python3 client_sim2.py {dirpath} {str(i)} {sys.argv[3]} {sys.argv[2]} {sys.argv[5]} 2>&1 | tee {dirname}/result_clientlog-{str(i)}.csv"
    open_terminal_with_command(command)

print("Please check the iTerm2 windows for the output.")

