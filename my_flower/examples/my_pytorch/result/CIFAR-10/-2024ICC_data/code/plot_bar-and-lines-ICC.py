import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot_from_excel(files, x_axis, y_axis, bar_width, figsize, legend_names):
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['firebrick','black', 'darkred', 'darkslategray',]
   # hatchs = [ "/" , ".",  "\\" , "|" , "-" , "+" , "x", "o", "O", "*" ]
    markers = ['o', 's', 'D', '^', 'v', 'x', 'o', 's', 'D', '^', 'v', 'x']

    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    for file in files:
        data = pd.read_excel(file)
        if data[x_axis].max() > max_x:
            max_x = data[x_axis].max()
        if data[x_axis].min() < min_x:
            min_x = data[x_axis].min()
        if data[y_axis].max() > max_y:
            max_y = data[y_axis].max()

    offset = -(bar_width * len(files) / 2) + bar_width / 2

    for idx, file in enumerate(files):
        data = pd.read_excel(file)
        color = colors[idx % len(colors)]
        line_color = colors[idx % len(colors)+2]
        marker = markers[idx % len(markers)]
       # hatch = hatchs[idx % len(hatchs)]
        annotation = 1

        if x_axis not in data.columns or y_axis not in data.columns:
            print(f"'{x_axis}' or '{y_axis}' not found in {file}. Skipping...")
            continue

        # Use provided legend names or default to file names
        label_name = legend_names[idx] if idx < len(legend_names) else f'File {idx+1}: {file}'

        # Plot bars with the same color regardless of data availability
        #color = f'C{idx}'
        #ax.bar(data[x_axis] + offset, data[y_axis], width=bar_width, label=label_name, color=color, hatch=hatch)
        ax.plot(data[x_axis] + offset, data["all_prog_ave[%]"],  marker=marker, label=label_name + f"*{annotation}", color=line_color)
        ax.bar(data[x_axis] + offset, data[y_axis], width=bar_width, label=label_name + f"*{annotation+1}", color=color)

        # Check and label NaN data points
        for i, y_value in enumerate(data[y_axis]):
            if pd.isna(y_value):
                ax.text(data[x_axis].iloc[i] + offset, 5, 'Model sending failed for all clients.', color='orange', rotation=90, fontfamily="Impact", fontsize=14)
                #ax.text(data[x_axis].iloc[i] + offset, 0, 'Model sending failed for all clients.', ha='center', va='center', color='orange', rotation=90)

        offset += bar_width

    ax.set_xlabel(x_axis, fontsize=20)
    ax.set_ylabel("Learning model receiving rate", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.legend(fontsize=14,ncol=2)
    ax.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.set_xticks(range(int(min_x), int(max_x + 1), 10))  # set x-axis ticks with step of 5
    ax.set_ylim(0, max_y*1.1)  # set y-axis limit to 120% of the max y value
    #ax.text(0.01, 0.75, "*Percentage of computation completion\n per communication round.", fontsize=12, va="bottom", ha="left", transform=ax.transAxes)
    ax.text(0.01, 0.75, "*1 Average learning progress\n     percentage per communication round.", fontsize=12, va="bottom", ha="left", transform=ax.transAxes)
    ax.text(0.01, 0.65, "*2 Proportion of models returned\n     to RSU after client learning.", fontsize=12, va="bottom", ha="left", transform=ax.transAxes)

    if not os.path.exists("../fig"):
        os.makedirs("../fig")
    plt.tight_layout()
    plt.savefig(f"../fig/{args.plot_name}.pdf")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot bar graph from Excel files.')
    parser.add_argument('--files', type=str, nargs='+', required=True, help='Path to the Excel files.')
    parser.add_argument('--x_axis', type=str, default='Time out[s]', help='X-axis label: default to Time out[s].')
    parser.add_argument('--y_axis', type=str, default='Accuracy', help='Y-axis label: default to Accuracy.')
    parser.add_argument('--bar_width', type=float, default=2.0, help='Width of the bars in the plot.')
    parser.add_argument('--figsize', type=lambda s: tuple(map(float, s.split(','))), default=(10, 5), help='Tuple indicating width and height of the figure e.g. 12,6')
    #parser.add_argument('--figsize', type=tuple, default=(10, 5), help='Tuple indicating width and height of the figure e.g. (10,5)')
    parser.add_argument('--legend_names', type=str, nargs='*', default=[], help='List of legend names corresponding to the files.')
    parser.add_argument('--plot_name', type=str,  default='test_plot', help='Plot name. default to "plot"')
    args = parser.parse_args()

    plot_from_excel(args.files, args.x_axis, args.y_axis, args.bar_width, args.figsize, args.legend_names)

