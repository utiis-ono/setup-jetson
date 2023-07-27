
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# コマンドライン引数からディレクトリ名を取得します
if len(sys.argv) < 2:
    print("Usage: plot_battery_info.py <directory1> <directory2> ...")
    sys.exit(1)

directories = sys.argv[1:]
# 使用する色のリスト
colors = ['black', 'blue', 'red']
labels = ['(a) w/o charge from 100% battery level', '(b) w/ charge from 20% battery level', '(c) w/ charge from 100% battery level']

plt.figure(figsize=(16, 8))

# 凡例用のプロットとハンドルを格納するリスト
legend_plots = []
legend_handles = []

# 各ディレクトリのデータを逆順でプロットします
for index, directory in reversed(list(enumerate(directories))):
    # CSVデータを読み込みます
    df = pd.read_csv(f'{directory}/battery_info.csv')

    # エリアプロットをプロットします
    color = colors[index % len(colors)]  # colorsリストから色を選択
    plot = plt.fill_between(df['Elapsed Time'], df['Battery Level'], color=color, alpha=0.5)

    # 凡例用のプロットとハンドルを格納します
    legend_plots.append(plot)
    legend_handles.append(labels[index])

# グラフの設定
plt.xlabel('Time [s]', fontsize=34)
plt.ylabel('Battery level [%]', fontsize=34)

plt.ylim([0, 104])  # Y軸の範囲を0から100まで設定します
plt.xlim([0,11200])  # X軸の範囲を0から100まで設定します

plt.tick_params(axis='both', which='major', labelsize=30)  # 軸のティックのラベルのサイズを変更します

# 凡例の順序を正しい順序に戻します
plt.legend(legend_plots[::-1], legend_handles[::-1], loc='lower center', fontsize=28)

# サブプロットの位置を調整します [left, bottom, width, height]
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
plt.grid()

# 図をPDFとして保存します
plt.savefig(f'battery_info_v2.pdf')

plt.close()
