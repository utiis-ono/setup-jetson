import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector, inset_axes, mark_inset

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_lines_from_csv(files, x_axis, y_axis, line_width, figsize, legend_names):
    #colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'black']
    #colors = ['lightcoral', 'salmon', 'red', 'indianred', 'firebrick', 'darkred', 'lightgray', 'gray', 'slategray', 'dimgray', 'darkslategray', 'black',]
    #colors = ['firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick', 'firebrick', 'black', 'black', 'black', 'black', 'black', 'black',]
    colors = ['firebrick', 'firebrick', 'firebrick', 'black', 'black', 'black',]
    #markers = ['o', 's', 'D', '^', 'v', 'x', 'o', 's', 'D', '^', 'v', 'x']
    markers = ['o', 's', 'x', 'o', 's', 'x']
    labelnum = 0
    fig, ax = plt.subplots(figsize=figsize)
    # 子アクセスを作成
    axins = inset_axes(ax, width="40%", height="40%",loc='lower left', bbox_to_anchor=(0.38, 0.36, 1, 1), bbox_transform=ax.transAxes)

    
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    for file in files:
        data = pd.read_csv(file)
        if data[x_axis].max() > max_x:
            max_x = data[x_axis].max()
        if data[x_axis].min() < min_x:
            min_x = data[x_axis].min()
        if data[y_axis].max() > max_y:
            max_y = data[y_axis].max()

    for idx, file in enumerate(files):
        data = pd.read_csv(file)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        labelnum = labelnum + 1

        if x_axis not in data.columns or y_axis not in data.columns:
            print(f"'{x_axis}' or '{y_axis}' not found in {file}. Skipping...")
            continue
        
        axins.plot(data[x_axis], data[y_axis], linewidth=line_width, marker=marker, markersize=3, color=color)  # 子アクセスにもデータをプロット
        ax.plot(data[x_axis], data[y_axis], linewidth=line_width, label=legend_names[idx] if idx < len(legend_names) else f'File {idx+1}: {file}', marker=marker, markersize=5, color=color)
        #ax.plot(data[x_axis], data[y_axis], linewidth=line_width, label=legend_names[idx] if idx < len(legend_names) else f'File {idx+1}: {file}', marker=marker, markersize=5)

    if x_axis == "Round":
        ax.set_xlabel("Communication Round", fontsize=20)  # ここでfontsizeを設定
    else:
        ax.set_xlabel("Time [s]", fontsize=20)  # ここでfontsizeを設定
    #ax.set_xlabel(x_axis, fontsize=20)
    ax.set_ylabel(y_axis, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=14,ncol=2)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_xlim(0, max_x)  # set x-axis limit to 120% of the max x value
    ax.set_ylim(0, max_y * 1.1)  # set y-axis limit to 120% of the max y value
    axins.set_xlim(2000, 4000)  # 子アクセスのx範囲を設定
    axins.set_ylim(0.6, 0.67)   # 子アクセスのy範囲を設定
    axins.set_xticks(np.linspace(1000, 4000, 4))  # x軸の文字間隔を調整
    axins.set_yticks(np.linspace(0.6, 0.67, 3))    # y軸の文字間隔を調整
    axins.grid(True, linestyle='--', color='gray', alpha=0.5)
    # mark_inset関数を追加
    mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5", )
    ax.text(0.76, 0.26, "20-60 indicates the timeout time of the RSU.", fontsize=11, va="bottom", ha="center", transform=ax.transAxes)

    if not os.path.exists("../fig"):
        os.makedirs("../fig")
    plt.tight_layout()
    plt.savefig(f"../fig/{args.plot_name}.pdf")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot line graph from csv files.')
    parser.add_argument('--files', type=str, nargs='+', required=True, help='Path to the csv files.')
    parser.add_argument('--x_axis', type=str, default='Sim Time[s]', help='X-axis label: default to Sim Time[s].')
    parser.add_argument('--y_axis', type=str, default='Accuracy', help='Y-axis label: default to Accuracy.')
    #parser.add_argument('--x_label', type=str, default='X label', help='X label name: default to X label.')
    #parser.add_argument('--y_label', type=str, default='Y label', help='Y label name: default to Y label.')
    parser.add_argument('--line_width', type=float, default=1.5, help='Width of the lines in the plot.')
    #parser.add_argument('--figsize', type=lambda s: tuple(map(int, s.split(','))), default=(10, 5), help='Tuple indicating width and height of the figure e.g. (10,5)')
    parser.add_argument('--figsize', type=lambda s: tuple(map(float, s.split(','))), default=(10, 5), help='Tuple indicating width and height of the figure e.g. 12,6')
    parser.add_argument('--legend_names', type=str, nargs='*', default=[], help='Legend names for the plots. If not provided, it defaults to the filenames.')
    parser.add_argument('--plot_name', type=str,  default='test_plot', help='Plot name. default to "plot"')
    args = parser.parse_args()

    plot_lines_from_csv(args.files, args.x_axis, args.y_axis, args.line_width, args.figsize, args.legend_names)
