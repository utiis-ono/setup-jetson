import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


args = sys.argv
if len(args) < 4:
    print("Select argument: 1:Loss or Accuracy 2:Time or Round 3:Total time")
    sys.exit()

Vertical_axis = args[1]
Horizontal_axis = args[2]
if args[2] == "Time":
    Horizontal_axis = "Time [s]"

total_time = int(args[3])

#ディレクトリ作成
dirname = "../figure/" + args[3] + "sec"
os.makedirs(dirname, exist_ok=True)

#data 読み込み
path1 = "../stay-remove/result_client-stay.csv"
path2 = "../stay-stay/result_client-stay.csv" 
data1 = pd.read_csv(path1)
data2 = pd.read_csv(path2)
#savename1 = "load-std.pdf"
#savename2 = "load-load.pdf"

print(data1)
print(data2)

#plotdata作成
round_count1 = 0
while int(args[3]) > int(data1.iat[round_count1,1]):
    round_count1 = round_count1 + 1
    #print(round_count1)
    #print(data1.iat[round_count1, 1])
plotdata1 = data1[:round_count1]
print(plotdata1)

round_count2 = 0
while int(args[3]) > int(data2.iat[round_count2,1]):
    round_count2 = round_count2 + 1
plotdata2 = data2[:round_count2]
print(plotdata2)

round_count = max(round_count1, round_count2)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(plotdata1[Horizontal_axis], plotdata1[Vertical_axis], linestyle = "solid",  marker = "o", color = "red",  label = "Moving")
plt.plot(plotdata2[Horizontal_axis], plotdata2[Vertical_axis], linestyle = "solid",  marker = "D", color = "blue",  label = "Staying")



if Horizontal_axis == "Round":
    ax.set_xlim(1, round_count)
    plt.xticks(np.arange(1, round_count + 1, step=2), rotation = 0, fontsize = 18)

if Vertical_axis == "Accuracy":
    ax.set_ylim(0, 0.64)
    plt.yticks(np.arange(0, 0.61, step=0.1), fontsize = 18)
    if Horizontal_axis == "Time [s]":
        #for i in range(900,total_time + 1,1800):
            #ax.axvspan(i,i+900,color='gray',alpha=0.3)
            #plt.vlines(i, 0, 0.6, colors='black', linestyle='dashed', linewidth=1)
        plt.xticks(np.arange(0, total_time + 1, step=900), rotation = 0, fontsize = 12)

if Vertical_axis == "Loss":
    ax.set_ylim(1, 2.51)
    plt.yticks(np.arange(1, 2.51, step=0.5), fontsize = 18)
    if Horizontal_axis == "Time [s]":
        #ax.set_xlim(0, total_time)
        #for i in range(900,total_time + 1,1800):
            #ax.axvspan(i,i+900,color='gray',alpha=0.3)
            #plt.vlines(i, 1, 2.5, colors='black', linestyle='dashed', linewidth=1)
            #plt.vlines(i + 900, 1, 2.5, colors='black', linestyle='dashed', linewidth=1)
        plt.xticks(np.arange(0, total_time + 1, step=900), rotation = 0, fontsize = 12)


plt.ylabel(Vertical_axis, fontsize = 18)
plt.xlabel(Horizontal_axis, fontsize = 18)
savename = args[1] + "-" + args[2] + ".pdf"

fig.set_figheight(4)
fig.set_figwidth(8)
#plt.legend()
plt.legend(loc=(0.8, 0.8))

plt.tight_layout()
plt.savefig(dirname + "/" + savename)
plt.show()
