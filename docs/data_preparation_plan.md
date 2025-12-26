# 交通设备异常检测数据准备方案

## 1. 项目需求

### 1.1 目标
基于合肥交通设备标注数据，微调 Qwen-VL 模型，实现**交通设备异常检测+定位**功能。

### 1.2 输入输出
- **输入**: 交通场景图片
- **输出**: 检测到的**所有设备**（正常+异常），标注状态、异常原因（如有）、位置坐标(bbox)

### 1.3 检测范围

| 设备类型 | CVAT标签 | 需检测的异常状态 |
|---------|----------|-----------------|
| 交通信号灯 | traffic-signal-system | all-off(全灭)、all-on(全亮)、abnormal(异常) |
| 交通诱导屏 | traffic-guidance-system | black-screen(黑屏)、abnormal(异常) |
| 限高架 | restricted-elevated | abnormal(异常) |
| 机柜 | cabinet | abnormal(异常) |
| 背包箱 | backpack-box | abnormal(异常) |

**注**: 非现场系统(off-site)及其子类（摄像头、闪光灯）暂不纳入异常检测范围。

### 1.4 异常原因定义

根据设备类型和异常状态，定义对应的异常原因描述：

| 设备类型 | 异常状态 | 异常原因描述 |
|---------|---------|-------------|
| **交通信号灯** | all-off | 信号灯不亮，可能存在电源故障、灯泡损坏或控制器故障 |
| | all-on | 信号灯全亮，控制系统可能发生故障 |
| | abnormal | 信号灯显示异常，可能存在灯泡部分损坏或颜色显示错误 |
| **交通诱导屏** | black-screen | 诱导屏黑屏不显示，可能存在电源故障或显示模块损坏 |
| | abnormal | 诱导屏显示异常，可能存在显示内容错乱或部分像素损坏 |
| **限高架** | abnormal | 限高架异常，可能存在结构损坏、标识缺失或倾斜变形 |
| **机柜** | abnormal | 机柜异常，可能存在柜门未关闭、外壳破损或倾斜 |
| **背包箱** | abnormal | 背包箱异常，可能存在箱门未关闭、外壳破损或安装松动 |

---

## 2. 源数据分析

### 2.1 数据位置
```
data/hefei_last_dataset/hefei_stage1_cvat_data/
├── annotations_lcx.xml    # 1722 张
├── annotations_lyy.xml    # 1042 张
├── annotations_wjk.xml    # 797 张
├── annotations_wsj0.xml   # 864 张
├── annotations_wsj2.xml   # 1231 张
├── annotations_xk0.xml    # 1689 张
├── annotations_xk1.xml    # 990 张
├── annotations_xk2.xml    # 1000 张
├── annotations_yjc.xml    # 1021 张
└── hefei-dataset/         # 图片目录
    └── hefei-dataset/
        ├── backpack-box/
        │   ├── bad/       # 异常样本
        │   └── good/      # 正常样本
        ├── cabinet/
        ├── off-site/
        ├── restricted-elevated/
        ├── traffic-guidance-system/
        └── traffic-signal-system/
```

### 2.2 数据统计
- **总图片数**: ~10,356 张
- **状态分布**:
  - normal: 19,930 个标注
  - all-off: 3,814 个 (信号灯全灭)
  - abnormal: 2,477 个 (通用异常)
  - black-screen: 368 个 (诱导屏黑屏)
  - all-on: 85 个 (信号灯全亮)

### 2.3 CVAT XML 格式示例
```xml
<image id="0" name="hefei-dataset/hefei-dataset/backpack-box/bad/backpack-box_bad_0.JPG"
       width="1279" height="1706">
  <box label="backpack-box" xtl="138.10" ytl="91.32" xbr="665.39" ybr="772.06">
    <attribute name="state">abnormal</attribute>
  </box>
</image>
```

---

## 3. 目标数据格式

### 3.1 Qwen-VL 微调格式
采用 Qwen-VL 官方的对话式微调格式（JSON）：

```json
{
  "id": "traffic_anomaly_001",
  "image": "/absolute/path/to/image.jpg",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n请检测图像中的交通设备异常情况..."
    },
    {
      "from": "assistant",
      "value": "检测到以下异常设备：\n1. 背包箱：异常 <box>(108,53),(520,452)</box>"
    }
  ]
}
```

### 3.2 坐标格式说明
- 使用 **归一化坐标**，范围 0-1000
- 格式: `<box>(x1,y1),(x2,y2)</box>`
- 其中 (x1,y1) 是左上角，(x2,y2) 是右下角
- 转换公式: `x_norm = int(x_pixel / width * 1000)`

### 3.3 Prompt 设计

**用户输入 (User)**:
```
<image>
请检测图像中的交通设备。需要检测的设备类型包括：交通信号灯、交通诱导屏、限高架、机柜、背包箱。

请按以下格式输出每个检测到的设备：
1. 设备类型
2. 状态（正常/异常状态）
3. 如果异常，说明可能的原因
4. 位置坐标：<box>(x1,y1),(x2,y2)</box>（坐标范围0-1000）

如果没有检测到任何设备，请回复"未检测到相关设备"。
```

**模型输出 (Assistant)** - 有正常和异常设备时:
```
检测到以下交通设备：

1. 交通信号灯（车行灯）
   - 状态：全灭
   - 原因：信号灯不亮，可能存在电源故障、灯泡损坏或控制器故障
   - 位置：<box>(667,532),(702,593)</box>

2. 交通信号灯（人行灯）
   - 状态：正常
   - 位置：<box>(234,456),(278,534)</box>

3. 背包箱
   - 状态：异常
   - 原因：背包箱异常，可能存在箱门未关闭、外壳破损或安装松动
   - 位置：<box>(405,366),(591,488)</box>

4. 机柜
   - 状态：正常
   - 位置：<box>(123,600),(198,780)</box>
```

**模型输出 (Assistant)** - 全部正常时:
```
检测到以下交通设备：

1. 交通信号灯（车行灯）
   - 状态：正常
   - 位置：<box>(345,234),(412,345)</box>

2. 机柜
   - 状态：正常
   - 位置：<box>(567,456),(634,678)</box>
```

**模型输出 (Assistant)** - 未检测到设备时:
```
未检测到相关设备。
```

### 3.4 实际检测效果样例

以下是一个真实标注图片的检测效果展示：

#### 示例图片信息
- **文件**: `backpack-box_bad_101.JPG`
- **尺寸**: 1279 × 1706
- **场景**: 交通路口，包含背包箱（箱门打开）、远处信号灯组、球机摄像头等

#### CVAT原始标注

| 设备类型 | 子类型 | 状态 | 像素坐标 (xtl,ytl,xbr,ybr) |
|---------|-------|------|---------------------------|
| traffic-signal-system | 车行灯 | **all-off** | (852.98, 909.50, 863.40, 933.83) |
| traffic-signal-system | 车行灯 | normal | (835.47, 917.43, 849.27, 927.23) |
| traffic-signal-system | 车行灯 | normal | (822.97, 910.03, 833.07, 933.23) |
| traffic-signal-system | 车行灯 | normal | (791.94, 909.71, 801.70, 933.36) |
| traffic-signal-system | 人行灯 | normal | (1128.59, 984.31, 1136.35, 999.83) |
| backpack-box | - | normal | (362.37, 442.07, 702.74, 734.71) |
| Dome-Camera | - | - | (727.29, 137.90, 759.87, 176.92) |
| off-site | - | - | (719.35, 1.87, 760.86, 180.57) |
| ignore | - | - | 多个小目标区域 |

#### 转换后的模型输出

```
检测到以下交通设备：

1. 交通信号灯（车行灯）
   - 状态：全灭
   - 原因：信号灯不亮，可能存在电源故障、灯泡损坏或控制器故障
   - 位置：<box>(666,533),(675,547)</box>

2. 交通信号灯（车行灯）
   - 状态：正常
   - 位置：<box>(653,537),(664,543)</box>

3. 交通信号灯（车行灯）
   - 状态：正常
   - 位置：<box>(643,533),(651,547)</box>

4. 交通信号灯（车行灯）
   - 状态：正常
   - 位置：<box>(619,533),(627,547)</box>

5. 交通信号灯（人行灯）
   - 状态：正常
   - 位置：<box>(882,577),(888,586)</box>

6. 背包箱
   - 状态：正常
   - 位置：<box>(283,259),(549,430)</box>
```

#### 数据处理说明

| 原始标注 | 处理方式 | 输出 |
|---------|---------|------|
| traffic-signal-system (all-off) | 主设备+异常 → 输出状态+原因 | ✅ 输出 |
| traffic-signal-system (normal) | 主设备+正常 → 输出状态 | ✅ 输出 |
| backpack-box (normal) | 主设备+正常 → 输出状态 | ✅ 输出 |
| Dome-Camera | 非主设备 → 跳过 | ❌ 不输出 |
| off-site | 非主设备 → 跳过 | ❌ 不输出 |
| ignore | 忽略区域 → 跳过 | ❌ 不输出 |

#### 完整JSON样例

```json
{
  "id": "traffic_detection_4",
  "image": "data/hefei_last_dataset/hefei_stage1_cvat_data/hefei-dataset/hefei-dataset/backpack-box/bad/backpack-box_bad_101.JPG",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n请检测图像中的交通设备。需要检测的设备类型包括：交通信号灯、交通诱导屏、限高架、机柜、背包箱。\n\n请按以下格式输出每个检测到的设备：\n1. 设备类型\n2. 状态（正常/异常状态）\n3. 如果异常，说明可能的原因\n4. 位置坐标：<box>(x1,y1),(x2,y2)</box>（坐标范围0-1000）\n\n如果没有检测到任何设备，请回复\"未检测到相关设备\"。"
    },
    {
      "from": "assistant",
      "value": "检测到以下交通设备：\n\n1. 交通信号灯（车行灯）\n   - 状态：全灭\n   - 原因：信号灯不亮，可能存在电源故障、灯泡损坏或控制器故障\n   - 位置：<box>(666,533),(675,547)</box>\n\n2. 交通信号灯（车行灯）\n   - 状态：正常\n   - 位置：<box>(653,537),(664,543)</box>\n\n3. 交通信号灯（车行灯）\n   - 状态：正常\n   - 位置：<box>(643,533),(651,547)</box>\n\n4. 交通信号灯（车行灯）\n   - 状态：正常\n   - 位置：<box>(619,533),(627,547)</box>\n\n5. 交通信号灯（人行灯）\n   - 状态：正常\n   - 位置：<box>(882,577),(888,586)</box>\n\n6. 背包箱\n   - 状态：正常\n   - 位置：<box>(283,259),(549,430)</box>"
    }
  ]
}
```

---

## 4. 实施步骤

### 4.1 数据处理流程

```
┌─────────────────┐
│  CVAT XML 文件  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  解析XML标注    │  提取: 图片路径、尺寸、bbox、类别、状态
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  筛选异常样本   │  只保留包含异常状态的图片
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  (可选)添加正常样本 │  按比例加入正常样本，平衡数据
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  坐标归一化     │  像素坐标 -> 0-1000
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  生成对话格式   │  组装 user/assistant 对话
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  划分数据集     │  train/val/test (8:1:1)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  输出JSON文件   │  train.json, val.json, test.json
└─────────────────┘
```

### 4.2 具体步骤

| 步骤 | 说明 | 输出 |
|-----|------|-----|
| 1 | 解析所有 CVAT XML 文件 | 内存中的标注数据结构 |
| 2 | 统计分析各类别/状态分布 | stats.json |
| 3 | 筛选包含异常设备的图片 | 异常样本列表 |
| 4 | (可选) 采样部分正常样本 | 平衡后的样本列表 |
| 5 | 转换为 Qwen-VL 对话格式 | 对话数据列表 |
| 6 | 按 8:1:1 划分数据集 | train/val/test |
| 7 | 验证图片路径有效性 | 过滤无效样本 |
| 8 | 保存为 JSON 文件 | 最终数据集 |

### 4.3 关键参数

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `include_normal` | True | 是否包含正常样本 |
| `normal_ratio` | 0.3 | 正常样本比例（相对于异常样本数） |
| `train_ratio` | 0.8 | 训练集比例 |
| `val_ratio` | 0.1 | 验证集比例 |
| `coord_scale` | 1000 | 坐标归一化范围 |

---

## 5. 输出文件结构

```
data/hefei_last_dataset/qwenvl_format/
├── train.json          # 训练集
├── val.json            # 验证集
├── test.json           # 测试集
└── stats.json          # 数据统计信息
```

### 5.1 完整样例 (train.json 中的一条)

```json
{
  "id": "traffic_anomaly_183",
  "image": "data/hefei_last_dataset/hefei_stage1_cvat_data/hefei-dataset/hefei-dataset/backpack-box/bad/backpack-box_bad_0.JPG",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n请检测图像中的交通设备。需要检测的设备类型包括：交通信号灯、交通诱导屏、限高架、机柜、背包箱。\n\n请按以下格式输出每个检测到的设备：\n1. 设备类型\n2. 状态（正常/异常状态）\n3. 如果异常，说明可能的原因\n4. 位置坐标：<box>(x1,y1),(x2,y2)</box>（坐标范围0-1000）\n\n如果没有检测到任何设备，请回复\"未检测到相关设备\"。"
    },
    {
      "from": "assistant",
      "value": "检测到以下交通设备：\n\n1. 背包箱\n   - 状态：异常\n   - 原因：背包箱异常，可能存在箱门未关闭、外壳破损或安装松动\n   - 位置：<box>(108,53),(520,452)</box>"
    }
  ]
}
```

### 5.2 混合样例（正常+异常设备）

```json
{
  "id": "traffic_anomaly_456",
  "image": "data/.../traffic-signal-system/bad/signal_bad_123.JPG",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n请检测图像中的交通设备..."
    },
    {
      "from": "assistant",
      "value": "检测到以下交通设备：\n\n1. 交通信号灯（车行灯）\n   - 状态：全灭\n   - 原因：信号灯不亮，可能存在电源故障、灯泡损坏或控制器故障\n   - 位置：<box>(234,156),(289,245)</box>\n\n2. 交通信号灯（人行灯）\n   - 状态：正常\n   - 位置：<box>(567,178),(612,256)</box>\n\n3. 机柜\n   - 状态：正常\n   - 位置：<box>(100,500),(180,720)</box>"
    }
  ]
}
```

### 5.3 全部正常样例

```json
{
  "id": "traffic_anomaly_789",
  "image": "data/.../cabinet/good/cabinet_good_45.JPG",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n请检测图像中的交通设备..."
    },
    {
      "from": "assistant",
      "value": "检测到以下交通设备：\n\n1. 机柜\n   - 状态：正常\n   - 位置：<box>(345,234),(456,567)</box>\n\n2. 背包箱\n   - 状态：正常\n   - 位置：<box>(123,345),(234,456)</box>"
    }
  ]
}
```

### 5.4 stats.json 示例

```json
{
  "total_images": 10356,
  "filtered_images": 2500,
  "train_count": 2000,
  "val_count": 250,
  "test_count": 250,
  "category_counts": {
    "traffic-signal-system": 15000,
    "backpack-box": 3500,
    "cabinet": 2800,
    "traffic-guidance-system": 1200,
    "restricted-elevated": 800
  },
  "abnormal_counts": {
    "traffic-signal-system": 3899,
    "backpack-box": 1200,
    "cabinet": 500,
    "traffic-guidance-system": 368,
    "restricted-elevated": 200
  }
}
```

---

## 6. 注意事项

### 6.1 数据质量
- **忽略 ignore 标签**: 过小或模糊的目标不参与训练
- **验证图片存在**: 跳过路径无效的样本
- **去重处理**: 同一图片可能在多个XML中出现

### 6.2 样本平衡
- 异常样本远少于正常样本，需要适当采样
- 建议异常:正常 ≈ 1:0.3 避免模型偏向"无异常"

### 6.3 坐标处理
- CVAT坐标为像素坐标（浮点数）
- 需要根据图片实际尺寸归一化到 0-1000
- 处理边界情况：坐标可能超出图片范围

### 6.4 信号灯特殊处理
- 信号灯有子类型（车行灯/人行灯），可在输出中标注
- 信号灯有方向属性（横向/竖向），可选是否输出

---

## 7. 后续微调步骤 (参考)

数据准备完成后，使用以下方式微调 Qwen-VL:

```bash
# 使用 LLaMA-Factory 或官方脚本
python train.py \
  --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
  --data_path data/hefei_last_dataset/qwenvl_format/train.json \
  --eval_data_path data/hefei_last_dataset/qwenvl_format/val.json \
  --output_dir outputs/traffic_anomaly_detector \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --learning_rate 1e-5 \
  --bf16 True
```

---

## 8. 执行命令

确认方案后，执行以下命令进行数据转换：

```bash
# 默认：包含所有有效样本（正常+异常设备）
python scripts/cvat_to_qwenvl.py \
  --cvat-dir data/hefei_last_dataset/hefei_stage1_cvat_data \
  --output-dir data/hefei_last_dataset/qwenvl_format \
  --train-ratio 0.8 \
  --val-ratio 0.1

# 可选：如果正常样本太多，可以采样部分
python scripts/cvat_to_qwenvl.py \
  --cvat-dir data/hefei_last_dataset/hefei_stage1_cvat_data \
  --output-dir data/hefei_last_dataset/qwenvl_format \
  --normal-sample-ratio 0.5  # 只保留50%的纯正常样本

# 可选：只保留有异常的样本
python scripts/cvat_to_qwenvl.py \
  --cvat-dir data/hefei_last_dataset/hefei_stage1_cvat_data \
  --output-dir data/hefei_last_dataset/qwenvl_format \
  --only-abnormal
```

### 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--cvat-dir` | 必填 | CVAT XML文件目录 |
| `--output-dir` | 必填 | 输出目录 |
| `--only-abnormal` | False | 是否只保留有异常设备的图片 |
| `--normal-sample-ratio` | 1.0 | 纯正常样本的采样比例 |
| `--train-ratio` | 0.8 | 训练集比例 |
| `--val-ratio` | 0.1 | 验证集比例 |
| `--seed` | 42 | 随机种子 |
