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
                write text "conda activate my_env"
                write text "{command}"
            end tell
        end tell
        """
        subprocess.run(["osascript", "-e", script])
    elif os.name == 'nt':  # Windows system
        subprocess.Popen(["start", "cmd", "/k", f"cd {current_dir} && {command}"], shell=True)

if len(sys.argv) != 7:
    print("Usage: python3 server.py [1]<dir name> [2]<rounds> [3]<time out[s]> [4]<round time[s]>")
    
    print("Usage: python3 sim_client2.py [1]<dir name> [5]<number of clients> [4]<round time[s]> [6]<model>")
    sys.exit(1)

dirname = 'result/' + sys.argv[1] + "/log"
if not os.path.exists(dirname):
    os.makedirs(dirname)

iterations = int(sys.argv[5])

processes = []

#start server
#command = "python3 server.py " + sys.argv[1] + " " + sys.argv[2] + " " + sys.argv[3] + " 2>&1 | tee " + dirname + "/result_serverlog.csv"
command =  f"python3 server.py {sys.argv[1]}  {sys.argv[2]} {sys.argv[3]} {sys.argv[4]} 2>&1 | tee {dirname}/result_serverlog.csv"
open_terminal_with_command(command)
time.sleep(3)
#start client
for i in range(iterations):
    #command = f"python3 flower_test.py {i}"    
    #command = "python3 sim_client2.py " + sys.argv[1] + " " + str(i) + " 2>&1 | tee " + dirname + "/result_clientlog-" + str(i) + ".csv"
    command = f"python3 sim_client2.py {sys.argv[1]} {str(i)} {sys.argv[4]} {sys.argv[6]} 2>&1 | tee {dirname}/result_clientlog-{str(i)}.csv"
    open_terminal_with_command(command)

print("Please check the iTerm2 windows for the output.")

