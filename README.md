# sr-security-camera
face detection and face super resolution


## 開発手順
branchのマージとかについて書いてある．<br>
<span style='color:red'>必読</span><br>
https://qiita.com/siida36/items/880d92559af9bd245c34

## Setup
sr_security_cameraパッケージ<br>
内包するモジュール
* face_detection
* srgan_naru
それぞれのディレクトリの中身をインポートするには
```shell
from srgan_naru.utils.imgproc import *
```
みたいに書く．

### install方法
```shell
pip install .
```
多分urlでのインストールもできるはず