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

# client_sim4.py
* CIFAR-100で各地のモデルをマージする時のシミュレーションに使う

# run_sim.py
* シミュレーションを実行する時はこれを使う
* server.pyとsum_cluent2.pyを自動起動する

# run_sim_sifar100.py
* CIFAR100と途中からのモデルを用いてシミュレーションを実行する時はこれを使う
* server_cifar100.pyとsum_cluent4.pyを自動起動する

# plot_fig-hogehohe.py
* 図を作る

# data_preproceesing.py
* logからdata整理する

# marge_result.py
* 指定したサーバの結果を統合する

# marge_model.py
* 指定したサーバのモデルを統合する

# marge-train-cifar100.py
* 作成したモデルをマージしてそこから学習を実行できるコード
* cifa-100のsuperclassをデータセットとして用いることができる
* subclassの一部を削除することができる

# marge-only-cifar100.py
* 作成したモデルをマージしてテストをするだけのコード
* marge-train-chair100.pyを使った方がいい
* cifa-100のsuperclassをデータセットとして用いることができる
