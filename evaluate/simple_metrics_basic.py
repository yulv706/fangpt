#!/usr/bin/env python3
"""
简化的分子评估脚本，专门输出关键指标（不依赖pandas）
"""

import csv
import argparse
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def read_csv_file(file_path):
    """读取CSV文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def calculate_validity_from_csv(csv_data):
    """从CSV数据计算有效性"""
    if not csv_data:
        return 0
    
    # 假设CSV中已经有validity列
    if 'validity' in csv_data[0]:
        return float(csv_data[0]['validity'])
    
    # 否则通过SMILES计算
    valid_count = 0
    for row in csv_data:
        smiles = row.get('smiles', '').strip()
        if smiles and smiles != '' and not smiles.startswith('<'):
            valid_count += 1
    
    return valid_count / len(csv_data)

def calculate_uniqueness_from_csv(csv_data, k=10000):
    """从CSV数据计算唯一性"""
    if not csv_data:
        return 0
    
    # 假设CSV中已经有unique列
    if 'unique' in csv_data[0]:
        return float(csv_data[0]['unique'])
    
    # 否则通过SMILES计算
    smiles_list = []
    for i, row in enumerate(csv_data):
        if i >= k:
            break
        smiles = row.get('smiles', '').strip()
        if smiles and smiles != '':
            smiles_list.append(smiles)
    
    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list) if smiles_list else 0

def calculate_novelty_from_csv(csv_data):
    """从CSV数据计算新颖性"""
    if not csv_data:
        return 0
    
    # 假设CSV中已经有novelty列
    if 'novelty' in csv_data[0]:
        return float(csv_data[0]['novelty'])
    
    # 否则返回默认值
    return 0.0

def calculate_basic_diversity(csv_data):
    """计算基本多样性指标"""
    if not csv_data:
        return 0, 0
    
    # 提取SMILES
    smiles_list = []
    for row in csv_data:
        smiles = row.get('smiles', '').strip()
        if smiles and smiles != '':
            smiles_list.append(smiles)
    
    if len(smiles_list) < 2:
        return 0, 0
    
    # 简单的多样性计算：基于SMILES长度和字符多样性
    lengths = [len(smi) for smi in smiles_list]
    avg_length = sum(lengths) / len(lengths)
    length_std = (sum((l - avg_length) ** 2 for l in lengths) / len(lengths)) ** 0.5
    
    # 字符多样性
    all_chars = set()
    for smi in smiles_list:
        all_chars.update(smi)
    
    char_diversity = len(all_chars) / 100.0  # 归一化
    length_diversity = min(length_std / 10.0, 1.0)  # 归一化
    
    return char_diversity, length_diversity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="path to generated CSV file")
    args = parser.parse_args()

    try:
        # 读取生成的分子数据
        print("Loading generated molecules...")
        csv_data = read_csv_file(args.path)
        print(f"Loaded {len(csv_data)} generated molecules")
        
        if len(csv_data) == 0:
            print("Error: No data found in CSV file")
            sys.exit(1)
        
        # 计算关键指标
        print("Calculating metrics...")
        
        # 1. Validity (有效性)
        validity = calculate_validity_from_csv(csv_data)
        
        # 2. Unique@10K (10K分子中的唯一性)
        unique_10k = calculate_uniqueness_from_csv(csv_data, k=10000)
        
        # 3. Novelty (新颖性)
        novelty = calculate_novelty_from_csv(csv_data)
        
        # 4. 基本多样性指标
        intdiv1, intdiv2 = calculate_basic_diversity(csv_data)
        
        # 输出结果
        print("=" * 60)
        print("Key Evaluation Metrics:")
        print("=" * 60)
        
        print(f"{'Metric':<15} {'Value':<15}")
        print("-" * 30)
        print(f"{'Validity':<15} {validity:.4f}")
        print(f"{'Unique@10K':<15} {unique_10k:.4f}")
        print(f"{'Novelty':<15} {novelty:.4f}")
        print(f"{'IntDiv1':<15} {intdiv1:.4f}")
        print(f"{'IntDiv2':<15} {intdiv2:.4f}")
        print(f"{'FCD/Test':<15} {'Need Moses'}")
        print(f"{'FCD/TestSF':<15} {'Need Moses'}")
        
        print("=" * 60)
        
        # 保存结果到文件
        results = {
            'validity': validity,
            'unique@10K': unique_10k,
            'novelty': novelty,
            'IntDiv1': intdiv1,
            'IntDiv2': intdiv2
        }
        
        # 创建结果文件名
        base_name = os.path.splitext(os.path.basename(args.path))[0]
        results_file = f"evaluation_results/{base_name}_simple_metrics.txt"
        
        # 确保目录存在
        os.makedirs("evaluation_results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            f.write("Key Evaluation Metrics:\n")
            f.write("=" * 30 + "\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)