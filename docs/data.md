# 数据准备

统一使用热红外坐标系标签，转换为YOLO格式 `class cx cy width height`。保留空标注图像，越界框裁剪至图像边界，退化框跳过。RGB/TIR共同进行增强与letterbox，输入为六通道、640×640。

输出结构：

```text
prepared/
├── images/train/      # RGB
├── images/val/
├── image/train/       # TIR，目录名为单数image
├── image/val/
├── labels/train/
├── labels/val/
└── dataset.yaml
```

### RGBTDronePerson

输入 `RGBTDronePerson.zip`，其中包含 `train_thermal.json`、`val_thermal.json` 及 `train|val/visible|thermal/`，允许存在共同外层目录。

- 官方训练/评估：4900/1225对。
- 类别顺序：person、rider、crowd。
- 读取thermal JSON，跳过ignore标注；保留原图字节与尺寸。

```bash
python prepare.py rgbt --source ../repro-work/raw/RGBTDronePerson.zip \
  --output ../repro-work/data/rgbt
```

### VTUAV-det

先解开外层下载包，准备以下文件；内层ZIP不必展开：

```text
VTUAV-source/
├── VTUAV_v1.0.zip     # train/rgb、train/ir、test/rgb、test/ir
├── train_ir.json
└── val_ir.json
```

- 官方训练/评估：11392/5378对，person单类。
- 使用IR COCO标签，过滤ignore、iscrowd和非person类别。
- 图像使用INTER_AREA缩放至640×360，保存为PNG；框按原尺寸归一化。
- JSON的val对应ZIP内test，保持官方划分。

```bash
python prepare.py vtuav --source ../repro-work/raw/VTUAV-source \
  --output ../repro-work/data/vtuav
```

### VTSaR

使用 `VTSaR_Crop_640.7z`：

```bash
7z t ../repro-work/raw/VTSaR_Crop_640.7z
7z x ../repro-work/raw/VTSaR_Crop_640.7z -o../repro-work/raw/VTSaR-Crop
python prepare.py vtsar --source ../repro-work/raw/VTSaR-Crop \
  --output ../repro-work/data/vtsar
```

- 官方训练/评估：4800/533对，person单类。
- 按路径中的train/val、rgb/ir/labels定位配对文件。
- 保留640×480 JPEG原字节；标签仅在越界裁剪时重写。
- 缺失配对会报错，多余孤立IR/标签文件不进入样本集。

### 数据复核

处理时自动检查数量、图像可读性、RGB/TIR配对、尺寸、标签范围及跨训练/评估的完全重复图像对，检查记录写入数据目录。也可单独复核：

```bash
python prepare.py rgbt --source ../repro-work/data/rgbt --audit-only
python prepare.py vtuav --source ../repro-work/data/vtuav --audit-only
python prepare.py vtsar --source ../repro-work/data/vtsar --audit-only
```
