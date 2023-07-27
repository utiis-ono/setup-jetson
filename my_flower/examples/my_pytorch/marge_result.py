import sys
import os
import pandas as pd

def weighted_average(dataframes, weights_column):
    if len(dataframes) == 0:
        raise ValueError("No dataframes provided.")
    
    final_df = dataframes[0].copy()

    # Round以外のカラムで重み付け平均を計算
    for col in final_df.columns:
        if col == weights_column or col == "Round":
            continue
        final_df[col] = sum(df[col] * df[weights_column] for df in dataframes) / sum(df[weights_column] for df in dataframes)
    
    # サーバごとの重みカラムを追加
    for i, df in enumerate(dataframes):
        server_weight_column = f"{os.path.basename(sys.argv[i + 2])}_weight"
        final_df[server_weight_column] = df[weights_column]
    
    return final_df

def main():
    if len(sys.argv) < 4:
        print("Usage: python merge_federated_results.py output_file server1 server2 ...")
        sys.exit(1)

    output_file = sys.argv[1]
    server_directories = sys.argv[2:]

    dataframes = [pd.read_csv(os.path.join(server_dir, "result_server.csv")) for server_dir in server_directories]

    merged_df = weighted_average(dataframes, weights_column="received")

    # 出力するカラムを選択
    columns_to_output = ["Round", "Accuracy", "Loss", "Precision", "Recall", "F-score"]

    # サーバごとの重みカラムを追加
    for server_dir in server_directories:
        columns_to_output.append(f"{os.path.basename(server_dir)}_weight")

    # 選択したカラムでデータフレームをフィルタリング
    merged_df = merged_df[columns_to_output]

    output_dir = os.path.join("result", "marge")
    os.makedirs(output_dir, exist_ok=True)
    merged_df.to_csv(os.path.join(output_dir, output_file), index=False)

if __name__ == "__main__":
    main()
