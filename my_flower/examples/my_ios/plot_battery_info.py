import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
import sys

# コマンドライン引数からディレクトリ名を取得します
if len(sys.argv) != 2:
    print("Usage: plot_battery_info.py <directory>")
    sys.exit(1)

directory = sys.argv[1]

# CSVデータを読み込みます
df = pd.read_csv(f'{directory}/battery_info.csv')

# バッテリーレベルに応じて色を指定します
df['Color'] = pd.cut(df['Battery Level'], bins=[0, 21, 51, np.inf], labels=['orange', 'blue', 'green'], right=False)

plt.figure(figsize=(16,8))

# 棒グラフを作成します
for color in df['Color'].unique():
    color_str = str(color)
    plt.bar(df[df['Color']==color]['Elapsed Time'], df[df['Color']==color]['Battery Level'], color=color_str, width=5)

# 折れ線グラフを同じ図に追加します
plt.plot(df['Elapsed Time'], df['Battery Level'], color='black', marker='o', markersize=4, label='Battery Level')

# グラフの設定
plt.xlabel('Elapsed Time', fontsize=28)
plt.ylabel('Battery Level', fontsize=28)

plt.ylim(0, 109)  # Y軸の範囲を0から100まで設定します

plt.tick_params(axis='both', which='major', labelsize=28)  # 軸のティックのラベルのサイズを変更します

plt.legend(fontsize=24)

# 図をPDFとして保存します
plt.savefig(f'{directory}/battery_info.pdf')

plt.close()

