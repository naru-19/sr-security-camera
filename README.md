# sr-security-camera
face detection and face super resolution


## 開発手順
branchのマージとかについて書いてある．<br>
<span style='color:red'>必読</span><br>
https://qiita.com/siida36/items/880d92559af9bd245c34

## 環境構築手順
1. GPUサーバーにdocker環境を構築
```shell
gpu-server$ cd hoge/sr-security-camera/docker
gpu-server$ bash docker_build.sh
gpu-server$ bash docker_run.sh $(適当な名前)
```
2. パッケージのインストール <br>
(この処理はリポジトリがpublicになったらdocker build中にやる)
```shell
container$ cd hoge/sr-security-camera
container$ pip install .
```

3. srgan_naru/pretrainedに
* p-best.pth
* vgg-pre.pickle

をコピー（<span style='color:red'>改善の余地</span>）<br>

### setup.cfg
sr_security_cameraパッケージ<br>
内包するモジュール
* face_detection
* srgan_naru

それぞれのディレクトリの中身をインポートするには
```python
from srgan_naru.utils.imgproc import *
```
みたいに書く．

memo：インストールはリポジトリの最上ディレクトリで
```shell
cd hoge/sr-security-camera
pip install .
```
（urlでのインストールもできるけどパスワードいる）


## 実行方法
* GPUサーバ(docker)側
```shell
cd hoge/sr-security-camera/script
# コードの一部がcuda:0しか使えないようになっているので実実行時にGPU指定
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="3" streamlit run main.py
```
カメラ側を実行する前にブラウザでip:8501にアクセス <br>
（これをしないとstreamlitがrunしない）

* カメラ側
```shell
cd hoge/sr-security-camera/script
python cam_sender.py --host_ip $(ip) --fps $(1~3ぐらい)
```