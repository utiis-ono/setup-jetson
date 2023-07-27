import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_data(directories, x_axis, y_axis):
    colors = ['black', 'blue', 'red', 'orange', 'purple', 'brown', 'green']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']
    labelnum = 0

    for idx, directory in enumerate(directories):
        data = pd.read_csv(os.path.join(directory, 'result_server.csv'))
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(data[x_axis], data[y_axis], marker=marker, label=labels[labelnum], color=color, markersize=3)
        labelnum = labelnum + 1
    
    if x_axis == "Round":
        plt.xlabel("Communication Round", fontsize=20)  # ここでfontsizeを設定
    else:
        plt.xlabel("Time [s]", fontsize=20)  # ここでfontsizeを設定
    plt.ylabel(y_axis, fontsize=20)  # ここでfontsizeを設定
    plt.xticks(fontsize=18)  # ここで数字のfontsizeを設定
    plt.yticks(fontsize=18)  # ここで数字のfontsizeを設定
    #plt.xlabel(x_axis)
    #plt.ylabel(y_axis)
    plt.legend(fontsize=14)

    if y_axis == 'p_rate[%]':
        plt.ylim(0, 109)
        
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    hogehoge = '_'.join([os.path.basename(directory) for directory in directories])
    fig_directory = os.path.join("result", "fig", hogehoge)
    os.makedirs(fig_directory, exist_ok=True)

    fig = plt.gcf()  # 現在の図を取得
    fig.set_size_inches(7, 4)  # 図のサイズを設定
    plt.tight_layout()  # レイアウトを調整

    plt.savefig(os.path.join(fig_directory, f'{y_axis}_vs_{x_axis}-server.pdf'), format='pdf')

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage:python3  plot_fig-server.py <directory1> <directory2> ... <x_axis> <y_axis>")
        sys.exit(1)

    directories = sys.argv[1:-2]
    x_axis = sys.argv[-2]
    y_axis = sys.argv[-1]

    allowed_x_axis = ['Round', 'Sim Time[s]', 'ios_time', 'ALL']
    allowed_y_axis = ['Accuracy', 'Loss', 'Precision', 'Recall', 'F-score', 'p_rate[%]', 'ALL']
    labels = ["(a) w/o charge from 100% battery lebel","(b) w/ charge from 20% battery lebel","(c) w/ charge from 100% battery lebel",]

    if x_axis not in allowed_x_axis:
        print(f"x_axis must be one of {', '.join(allowed_x_axis)}")
        sys.exit(1)

    if y_axis not in allowed_y_axis:
        print(f"y_axis must be one of {', '.join(allowed_y_axis)}")
        sys.exit(1)

    if x_axis == 'ALL' and y_axis == 'ALL':
        for x in allowed_x_axis[:-1]:
            for y in allowed_y_axis[:-1]:
                plt.figure()
                plot_data(directories, x, y)
                plt.close()
    elif x_axis == 'ALL':
        for x in allowed_x_axis[:-1]:
            plt.figure()
            plot_data(directories, x, y_axis)
            plt.close()
    elif y_axis == 'ALL':
        for y in allowed_y_axis[:-1]:
            plt.figure()
            plot_data(directories, x_axis, y)
            plt.close()
    else:
        plot_data(directories, x_axis, y_axis)
