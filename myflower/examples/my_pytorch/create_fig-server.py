import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_data(directories, x_axis, y_axis):
    # グラフの色とマーカーのリストを定義
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'black']
    markers = ['o', 's', 'D', '^', 'v', '<', '>']

    for idx, directory in enumerate(directories):
        # データの読み込み
        data = pd.read_csv(os.path.join(directory, 'result_server.csv'))

        # プロット (データ点に異なるマーカーを追加)
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.plot(data[x_axis], data[y_axis], marker=marker, label=directory, color=color, markersize=2)


    # グラフのタイトルと軸ラベル
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()


    # ディレクトリ名に依存したhogehogeを作成
    hogehoge = '_'.join([os.path.basename(directory) for directory in directories])

    # figディレクトリの作成
    fig_directory = os.path.join("result", "fig", hogehoge)
    os.makedirs(fig_directory, exist_ok=True)

    # グラフの保存
    plt.savefig(os.path.join(fig_directory, f'{y_axis}_vs_{x_axis}-server.pdf'), format='pdf')

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: plot_data.py <directory1> <directory2> ... <x_axis> <y_axis>")
        sys.exit(1)

    directories = sys.argv[1:-2]
    x_axis = sys.argv[-2]
    y_axis = sys.argv[-1]
    
    # y_axisがp_rate_percentageの場合、Y軸を0~100に設定
    if y_axis == 'p_rate[%]':
        plt.ylim(0, 109)
        
    # グリッド線を追加
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    allowed_x_axis = ['Round', 'Sim Time[s]']
    allowed_y_axis = ['Accuracy', 'Loss', 'p_rate[%]']

    if x_axis not in allowed_x_axis:
        print(f"x_axis must be one of {', '.join(allowed_x_axis)}")
        sys.exit(1)

    if y_axis not in allowed_y_axis:
        print(f"y_axis must be one of {', '.join(allowed_y_axis)}")
        sys.exit(1)

    plot_data(directories, x_axis, y_axis)

