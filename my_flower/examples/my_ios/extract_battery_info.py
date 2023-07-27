import json
from datetime import datetime
import sys
import pandas as pd

# コマンドライン引数からJSONファイル名を取得します
if len(sys.argv) != 2:
        print("Usage:extract_battery_info.py <directory>")
        sys.exit(1)

json_file = sys.argv[1] + "benchmark.json"

with open(json_file, 'r') as f:
    # JSONデータをPythonオブジェクト（ここではリスト）に変換します
    data = json.load(f)
    
# データを保存するための空のリストを作成します
results = []

# バッテリーヒストリー部分を取得
battery_history = data.get('batteryHistory', [])

if not battery_history:
    print("No 'batteryHistory' field found in the JSON file.")
    sys.exit()

# 基準となる日時（一行目の日時）をPythonのdatetimeオブジェクトに変換します
base_time = datetime.strptime(battery_history[0]['date'], "%Y-%m-%dT%H:%M:%SZ")

for item in battery_history:
    # 各日時をPythonのdatetimeオブジェクトに変換します
    current_time = datetime.strptime(item['date'], "%Y-%m-%dT%H:%M:%SZ")
    # 経過時間を計算します（秒単位）
    elapsed_time = (current_time - base_time).total_seconds()
    # 結果をリストに追加します
    results.append([item["batteryLevel"], elapsed_time])

# データフレームを作成します
df = pd.DataFrame(results, columns=['Battery Level', 'Elapsed Time'])

# 結果をCSVファイルに保存します
df.to_csv(f'{sys.argv[1]}/battery_info.csv', index=False)
