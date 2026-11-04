# TriCue

**TriCue: Towards Reliable and Complementary Representation Learning for RGB-T Object Detection**

## 1. 安装

参考环境：Python 3.11.13、PyTorch 2.5.0、torchvision 0.20.0、CUDA 11.8。

```bash
git clone https://github.com/78687dfgjn/TriCue.git
cd TriCue
conda create -n tricue python=3.11.13 -y
conda activate tricue
python -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
```

使用仓库自带的双流运行库，不另装PyPI版ultralytics。完整环境版本见 [environment-pip-list.txt](environment-pip-list.txt)。

## 2. 数据与基础权重

数据、权重及输出放在仓库外的 `../repro-work/`。原始数据格式及配对规则见 [数据准备](docs/data.md)。

```bash
python prepare.py rgbt --source ../repro-work/raw/RGBTDronePerson.zip --output ../repro-work/data/rgbt
python prepare.py vtsar --source ../repro-work/raw/VTSaR-Crop --output ../repro-work/data/vtsar
python prepare.py vtuav --source ../repro-work/raw/VTUAV-source --output ../repro-work/data/vtuav

mkdir -p ../repro-work/assets
curl -L --fail https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -o ../repro-work/assets/yolov8n.pt
```

## 3. 训练

| 数据集 | train / val | 默认轮数 | 早停 |
|---|---:|---:|---|
| RGBTDronePerson | 4900 / 1225 | 固定80 | 关闭 |
| VTSaR | 4800 / 533 | 至多200 | Val mAP50，patience=50 |
| VTUAV-det | 11392 / 5378 | 至多200 | Val mAP50，patience=50 |

统一640输入、batch32/nbs64、workers4、SGD、lr0=0.005、lrf=0.01、warmup3、AMP前向及FP32分配/验证。学习率按本次80/200轮线性衰减，最后10轮关闭马赛克；未指定seed时使用0。

```bash
python train.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --pretrained ../repro-work/assets/yolov8n.pt --variant C \
  --project ../repro-work/runs --name rgbt_c_seed0 --seed 0 --device 0
```

跨数据集将 `--dataset`、`--data`及运行名替换为 `vtsar` 或 `vtuav`，预算自动切换。`--variant`可选 `ADD`、`SCR`、`RCMF`、`FSPR`、`SCR_RCMF`、`SCR_FSPR`、`RCMF_FSPR`；完整模型为 `C`。补充种子使用553704、152776324、42；基础和模块初始化保持固定。

输出 `best_map50.pt`（mAP50最佳）、`best.pt`（fitness最佳）、`last.pt`（完整恢复状态），以及逐轮指标和协议记录。早停只依据未舍入mAP50，严格提高才重置耐心。

## 4. 评估与续训

```bash
python evaluate.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --weights ../repro-work/runs/rgbt_c_seed0/weights/best_map50.pt \
  --output ../repro-work/eval/rgbt_c --split val --device 0

python train.py --dataset rgbt --data ../repro-work/data/rgbt/dataset.yaml \
  --resume ../repro-work/runs/rgbt_c_seed0/weights/last.pt \
  --project ../repro-work/runs --device 0
```

测试：`python -m unittest discover -s tests -v`。
