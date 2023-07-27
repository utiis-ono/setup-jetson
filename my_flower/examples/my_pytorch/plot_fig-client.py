import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import glob

def plot_data(directory, x_axis, y_axis, num_files):
    colors = plt.cm.get_cmap('viridis', num_files)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', 'H', '*', '+']

    for i in range(num_files):
        # データの読み込み
        data = pd.read_csv(os.path.join(directory, f'result_client-{i}.csv'))

        # プロット (データ点にドットを追加)
        plt.plot(data[x_axis], data[y_axis], marker=markers[i % len(markers)], color=colors(i), label=f'Client {i}', linestyle='-')

    # グラフのタイトルと軸ラベル
    #plt.title(f'{y_axis} vs {x_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()

    # figディレクトリの作成
    fig_directory = os.path.join(directory, 'fig')
    os.makedirs(fig_directory, exist_ok=True)

    # グラフの保存
    plt.savefig(os.path.join(fig_directory, f'{y_axis}_vs_{x_axis}-client.pdf'), format='pdf')


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 plot_fig-client.py <directory> <x_axis> <y_axis> <num_files>")
        sys.exit(1)

    directory = sys.argv[1]
    x_axis = sys.argv[2]
    y_axis = sys.argv[3]
    num_files = int(sys.argv[4])

    allowed_x_axis = ['Round', 'Sim Time [s]']
    allowed_y_axis = ['Accuracy', 'Loss', 'prog [%]']

    if x_axis not in allowed_x_axis:
        print(f"x_axis must be one of {', '.join(allowed_x_axis)}")
        sys.exit(1)

    if y_axis not in allowed_y_axis:
        print(f"y_axis must be one of {', '.join(allowed_y_axis)}")
        sys.exit(1)

    plot_data(directory, x_axis, y_axis, num_files)


