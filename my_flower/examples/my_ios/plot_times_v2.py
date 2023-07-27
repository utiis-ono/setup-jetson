import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# コマンドライン引数からディレクトリ名と色を取得します
if len(sys.argv) != 3:
    print("Usage: plot_times.py <directory> <color>")
    sys.exit(1)

directory = sys.argv[1]
color = sys.argv[2]

# CSVデータを読み込みます
df = pd.read_csv(f'{directory}/times.csv')

# 0以上の値のみをフィルタリングします
df = df[df['per_time'] >= 0]
df = df[df['total_time'] >= 0]

plt.figure(figsize=(16, 8))

# per_timeについてのエリアプロットを作成します
ax1 = plt.gca()  # get current axis
ax1.fill_between(df['round'], df['per_time'], color='gray', label='Communication round time [s]')
ax1.set_ylabel('Communication round time [s]', fontsize=34)
ax1.set_xlabel('Communication round', fontsize=34)
ax1.set_ylim([0,130])  # per_timeのy軸の範囲を設定します
ax1.set_xlim([1, 100])  # per_timeのx軸の範囲を設定します
ax1.set_xticks([1,20,40,60,80,100]) 
ax1.legend(loc='upper left', fontsize=30)
ax1.tick_params(axis='both', which='major', direction='in', labelsize=30)  # 軸のティックのラベルのサイズを変更します

# 通信ラウンド時間が100の位置に横線を引きます
ax1.axhline(y=100, color='black', linestyle='dashed')

# total_timeについてのエリアプロットを作成します
ax2 = ax1.twinx()  # x軸を共有する新たなy軸を作成します
ax2.set_yticks([2000,4000,6000,8000,10000]) 
ax2.set_ylim([0, 12000])  # per_timeのx軸の範囲を設定します
ax2.fill_between(df['round'], df['total_time'], color=color, alpha=0.5, label='Total time [s]')  # 色を設定
ax2.set_ylabel('Total time [s]', fontsize=34)
ax2.set_ylim(bottom=0)  # total_timeのy軸の範囲を設定します
ax2.legend(loc='lower right', fontsize=30)
ax2.tick_params(axis='both', which='major', direction='in', labelsize=30)  # 軸のティックのラベルのサイズを変更します

# サブプロットの位置を調整します [left, bottom, width, height]
ax1.set_position([0.1, 0.15, 0.77, 0.8])
ax2.set_position([0.1, 0.15, 0.77, 0.8])

# 最後のディレクトリ名を取得します
last_directory_name = os.path.basename(os.path.normpath(directory))

# 図をPDFとして保存します
plt.savefig(f'{directory}/times-{last_directory_name}_v2.pdf')

plt.close()

