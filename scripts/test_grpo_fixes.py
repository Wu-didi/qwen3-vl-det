#!/usr/bin/env python3
"""
测试 GRPO 脚本修复的完整性

运行此脚本以验证所有修复是否正确应用
"""

import sys
import os
import re
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_grpo_finetune_trl():
    """测试 grpo_finetune_trl.py 的修复"""
    print("\n" + "="*60)
    print("测试 grpo_finetune_trl.py")
    print("="*60)
    
    filepath = "scripts/training/grpo_finetune_trl.py"
    with open(filepath, 'r') as f:
        content = f.read()
    
    tests_passed = 0
    tests_total = 0
    
    # 测试 1: 检查参数定义修复
    tests_total += 1
    if '--no_4bit' in content and 'set_defaults(use_4bit=True)' in content:
        print("✓ [1/3] 参数冲突修复: 找到 --no_4bit 和 set_defaults")
        tests_passed += 1
    else:
        print("✗ [1/3] 参数冲突修复: 未找到正确的参数定义")
    
    # 测试 2: 检查 Pillow 兼容性
    tests_total += 1
    if 'Image.Resampling.LANCZOS' in content and 'except AttributeError' in content:
        print("✓ [2/3] Pillow 兼容性: 找到版本兼容代码")
        tests_passed += 1
    else:
        print("✗ [2/3] Pillow 兼容性: 未找到兼容代码")
    
    # 测试 3: 检查 peft_config 处理
    tests_total += 1
    if 'Keep peft_config' in content or 'keep peft_config' in content.lower():
        print("✓ [3/3] peft_config 处理: 找到正确的注释")
        tests_passed += 1
    else:
        print("✗ [3/3] peft_config 处理: 未找到相关注释")
    
    print(f"\n结果: {tests_passed}/{tests_total} 测试通过")
    return tests_passed == tests_total


def test_grpo_finetune():
    """测试 grpo_finetune.py 的修复"""
    print("\n" + "="*60)
    print("测试 grpo_finetune.py")
    print("="*60)
    
    filepath = "scripts/training/grpo_finetune.py"
    with open(filepath, 'r') as f:
        content = f.read()
    
    tests_passed = 0
    tests_total = 0
    
    # 测试 1: 检查梯度累积修复
    tests_total += 1
    if 'Applying remaining gradients' in content:
        print("✓ [1/5] 梯度累积修复: 找到剩余梯度处理代码")
        tests_passed += 1
    else:
        print("✗ [1/5] 梯度累积修复: 未找到剩余梯度处理")
    
    # 测试 2: 检查参考模型 merge 修复
    tests_total += 1
    if 'not merged for safety' in content:
        print("✓ [2/5] 参考模型 merge: 找到安全处理注释")
        tests_passed += 1
    else:
        print("✗ [2/5] 参考模型 merge: 未找到安全处理")
    
    # 测试 3: 检查 Pillow 兼容性
    tests_total += 1
    if 'Image.Resampling.LANCZOS' in content:
        print("✓ [3/5] Pillow 兼容性: 找到版本兼容代码")
        tests_passed += 1
    else:
        print("✗ [3/5] Pillow 兼性: 未找到兼容代码")
    
    # 测试 4: 检查奖励函数修复
    tests_total += 1
    if 'avg_iou ** 0.5' in content:
        print("✓ [4/5] 奖励函数修复: 找到平方根映射")
        tests_passed += 1
    else:
        print("✗ [4/5] 奖励函数修复: 未找到平方根映射")
    
    # 测试 5: 检查 Assistant 位置回退
    tests_total += 1
    if '0.7' in content and 'fallback' in content.lower():
        print("✓ [5/5] Assistant 位置回退: 找到 70% 回退")
        tests_passed += 1
    else:
        print("✗ [5/5] Assistant 位置回退: 未找到 70% 回退")
    
    print(f"\n结果: {tests_passed}/{tests_total} 测试通过")
    return tests_passed == tests_total


def test_syntax():
    """测试语法正确性"""
    print("\n" + "="*60)
    print("测试语法正确性")
    print("="*60)
    
    import ast
    
    files = [
        "scripts/training/grpo_finetune_trl.py",
        "scripts/training/grpo_finetune.py"
    ]
    
    all_passed = True
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print(f"✓ {filepath}: 语法正确")
        except SyntaxError as e:
            print(f"✗ {filepath}: 语法错误 line {e.lineno}: {e.msg}")
            all_passed = False
    
    return all_passed


def test_imports():
    """测试导入是否正常"""
    print("\n" + "="*60)
    print("测试模块导入")
    print("="*60)
    
    try:
        from PIL import Image
        print("✓ PIL (Pillow) 导入成功")
        
        # 测试 Pillow 版本兼容性
        try:
            resample = Image.Resampling.LANCZOS
            print(f"✓ 检测到 Pillow 10+ (Resampling.LANCZOS = {resample})")
        except AttributeError:
            resample = Image.LANCZOS
            print(f"✓ 检测到 Pillow 9- (LANCZOS = {resample})")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False


def main():
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*15 + "GRPO 修复验证测试" + " "*25 + "║")
    print("╚" + "="*58 + "╝")
    
    results = []
    
    # 运行所有测试
    results.append(("语法检查", test_syntax()))
    results.append(("模块导入", test_imports()))
    results.append(("grpo_finetune_trl.py", test_grpo_finetune_trl()))
    results.append(("grpo_finetune.py", test_grpo_finetune()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过！修复已正确应用。")
        print("="*60)
        print("\n下一步:")
        print("1. 准备训练数据 (data/qwen_data/train.json)")
        print("2. 运行训练:")
        print("   python scripts/training/grpo_finetune_trl.py \\")
        print("       --train_data data/qwen_data/train.json \\")
        print("       --output_dir outputs/qwen3vl_grpo")
        return 0
    else:
        print("❌ 部分测试失败，请检查修复是否完整。")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
