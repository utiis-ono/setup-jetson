# conda activate flowerが必要

# server.py  
* サーバ

# client.py  
* クライアント．実機で使う最初使ってた
* クライアントが動く時間を設定できる．

# client_model_selector.py  
* client.pyを改良してデータセットをMNISTかCIFER-10か選択できるようにした．
* client.py使うならこっち使った方がいいかも

# client_sim.py
* シミュレーションだけと実時間計測なのであまり多数起動すると端末1台ごとのスペックが落ちる

# cleint_sim2.py
* シミュレーション用基本的にシミュレーションならこちらを使う

# client_sim3.py
* %によって学習効率が変化することを確かめる時に使う

# run_sim.py
* シミュレーションを実行する時はこれを使う
* server.pyとsum_cluent2.pyを自動起動する

# plot_fig-hogehohe.py
* 図を作る

# data_preproceesing.py
* logからdata整理する

# marge_result.py
* 指定したサーバの結果を統合する

# marge_model.py
* 指定したサーバのモデルを統合する
