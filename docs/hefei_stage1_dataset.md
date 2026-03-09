# Hefei Stage-1 数据集统计

数据来源：`data/hefei_last_dataset/hefei_stage1_cvat_data`
转换脚本：`scripts/data/convert_cvat_to_sft_2.py`
输出目录：`data/hefei_last_dataset/sft_output/`

---

## 1. 总体规模

| 指标 | 数值 |
|------|------|
| 原始 XML 文件数 | 9（每位标注员一个） |
| 总图片数 | 10,356 |
| 总 bbox 标注数 | 26,674 |
| 平均每张图 bbox 数 | 2.58 |
| 最大单张 bbox 数 | 16 |
| 无目标图片（empty） | 571（5.5%） |

---

## 2. 训练 / 验证 / 测试 划分

划分方式：**按类别 × 状态分层抽样（stratified split）**，保证每个分层各占 80 / 10 / 10。

| Split | 图片数 | bbox 数 | 空场景数 |
|-------|--------|---------|---------|
| train | 8,287  | 21,124  | 432 |
| val   | 1,036  | 2,847   | 74  |
| test  | 1,033  | 2,703   | 65  |
| **合计** | **10,356** | **26,674** | **571** |

---

## 3. 设备类别分布

### 3.1 按设备类型

| device_type | train | val | test | 合计 |
|-------------|-------|-----|------|------|
| traffic_signal | 10,195 | 1,475 | 1,376 | **13,046** |
| backpack_box | 4,252 | 571 | 517 | **5,340** |
| cabinet | 3,959 | 438 | 481 | **4,878** |
| guidance_screen | 1,475 | 195 | 178 | **1,848** |
| height_limit_bar | 1,243 | 168 | 151 | **1,562** |

### 3.2 traffic_signal 子类型

| sub_type | 数量 | 占比 |
|----------|------|------|
| vehicle_signal（机动车信号灯） | 9,448 | 72.4% |
| pedestrian_signal（行人信号灯） | 3,598 | 27.6% |

---

## 4. 状态（state）分布

| device_type | state | train | val | test | 合计 |
|-------------|-------|-------|-----|------|------|
| traffic_signal | normal | 7,010 | 975 | 878 | **8,863** |
| traffic_signal | all-off | 2,892 | 470 | 452 | **3,814** |
| traffic_signal | abnormal | 238 | 21 | 25 | **284** |
| traffic_signal | all-on | 55 | 9 | 21 | **85** |
| backpack_box | normal | 3,516 | 482 | 428 | **4,426** |
| backpack_box | abnormal | 736 | 89 | 89 | **914** |
| cabinet | normal | 3,206 | 327 | 382 | **3,915** |
| cabinet | abnormal | 753 | 111 | 99 | **963** |
| guidance_screen | normal | 949 | 126 | 115 | **1,190** |
| guidance_screen | black-screen | 286 | 44 | 38 | **368** |
| guidance_screen | abnormal | 240 | 25 | 25 | **290** |
| height_limit_bar | normal | 1,220 | 167 | 149 | **1,536** |
| height_limit_bar | abnormal | 23 | 1 | 2 | **26** |

> **注意：** `height_limit_bar/abnormal` 仅 26 条，样本极度稀少，评估时需注意指标可靠性。

---

## 5. 分层分布（每层精确 80/10/10）

| 分层（category/state） | 总数 | train | val | test |
|------------------------|------|-------|-----|------|
| backpack-box/bad | 861 | 689 | 86 | 86 |
| backpack-box/good | 861 | 689 | 86 | 86 |
| cabinet/bad | 912 | 730 | 91 | 91 |
| cabinet/good | 777 | 622 | 78 | 77 |
| off-site | 990 | 792 | 99 | 99 |
| restricted-elevated/bad | 20 | 16 | 2 | 2 |
| restricted-elevated/good | 1,001 | 801 | 100 | 100 |
| traffic-guidance-system/bad | 295 | 236 | 30 | 29 |
| traffic-guidance-system/good | 747 | 598 | 75 | 74 |
| traffic-signal/Pedestrian/bad | 797 | 638 | 80 | 79 |
| traffic-signal/Pedestrian/good | 1,231 | 985 | 123 | 123 |
| traffic-signal/Vehicle/bad | 864 | 691 | 86 | 87 |
| traffic-signal/Vehicle/good | 1,000 | 800 | 100 | 100 |

---

## 6. 图片尺寸分布

| 指标 | 宽（width） | 高（height） |
|------|-------------|--------------|
| 最小 | 502 px | 370 px |
| 最大 | 4,608 px | 4,624 px |
| 平均 | 1,419 px | 1,802 px |

---

## 7. 标注员来源

| XML 文件 | 图片数 |
|----------|--------|
| annotations_lcx | 1,722 |
| annotations_xk0 | 1,689 |
| annotations_wsj2 | 1,231 |
| annotations_xk2 | 1,000 |
| annotations_xk1 | 990 |
| annotations_wsj0 | 864 |
| annotations_yjc | 1,021 |
| annotations_lyy | 1,042 |
| annotations_wjk | 797 |

---

## 8. 注意事项

- **类别不均衡**：`traffic_signal` 占总 bbox 的 48.9%，`height_limit_bar` 仅占 5.9%，训练时可考虑加权损失或过采样。
- **异常状态稀少**：`height_limit_bar/abnormal` 仅 26 条、`traffic_signal/all-on` 仅 85 条，异常检测精度参考价值有限。
- **off-site 图片**：990 张 `off-site` 类图片在标注中被忽略（IGNORE label），产生空 bbox 场景，占全部空场景的主体。
- **图片分辨率差异大**：从 502×370 到 4608×4624，训练时建议统一 resize 策略。

---

## 9. 复现命令

```bash
python3 scripts/data/convert_cvat_to_sft_2.py \
  --xml_dir data/hefei_last_dataset/hefei_stage1_cvat_data \
  --output_dir data/hefei_last_dataset/sft_output \
  --image_root data/hefei_last_dataset/hefei_stage1_cvat_data \
  --keep_meta \
  --train_ratio 0.8 \
  --val_ratio 0.1
```
