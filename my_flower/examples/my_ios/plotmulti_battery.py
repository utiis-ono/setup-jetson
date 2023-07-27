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
labels = ['(a) Battery Level - from 100% w/o charge', '(b) Battery Level - from 20% w/ charge', '(c) Battery Level - from 100% w/ charge']

plt.figure(figsize=(16, 8))

# 各ディレクトリのデータをプロットします
for index, directory in enumerate(directories):
    # CSVデータを読み込みます
    df = pd.read_csv(f'{directory}/battery_info.csv')

    # 折れ線グラフをプロットします
    labelname = directory.split("/",1)
    color = colors[index % len(colors)]  # colorsリストから色を選択
#    plt.plot(df['Elapsed Time'], df['Battery Level'], color=color, marker='o', markersize=4, label=f'Battery Level - {labelname[1]}')   
    plt.plot(df['Elapsed Time'], df['Battery Level'], color=color, marker='o', markersize=4, label=labels[index % len(labels)])   

# グラフの設定
plt.xlabel('Time [s]', fontsize=34)
plt.ylabel('Battery level [%]', fontsize=34)

plt.ylim(0, 104)  # Y軸の範囲を0から100まで設定します

plt.tick_params(axis='both', which='major', labelsize=30)  # 軸のティックのラベルのサイズを変更します

plt.legend(fontsize=28)

# サブプロットの位置を調整します [left, bottom, width, height]
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.9)
plt.grid()

# 図をPDFとして保存します
plt.savefig(f'battery_info.pdf')

plt.close()
