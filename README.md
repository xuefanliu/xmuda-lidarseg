# xmuda-lidarseg

使用point-wise标注的nuscenes-lidarseg数据集  
```
$ python xmuda/data/nuscenes/lidarseg_preprocess.py
$ python xmuda/train_xmuda.py --cfg=configs/nuscenes/usa_singapore/xmuda.yaml
```
注意：在运行train_xmuda.py时先检查`xmuda/data/build.py`中导入的dataloader是否修改为lidarseg版本的
```
from xmuda.data.nuscenes.lidarseg_nuscenes_dataloader import NuScenesSCN
# 原版为 from xmuda.data.nuscenes.nuscenes_dataloader import NuScenesSCN
```
