import argparse
import pandas as pd
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED
import numpy as np

def calculate_basic_metrics(smiles_list):
    """计算基本的分子指标"""
    results = {
        'total_molecules': len(smiles_list),
        'valid_molecules': 0,
        'unique_molecules': 0,
        'avg_molecular_weight': 0,
        'avg_logp': 0,
        'avg_qed': 0,
        'validity_rate': 0,
        'uniqueness_rate': 0
    }
    
    valid_smiles = []
    valid_mols = []
    
    # 验证分子
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
                valid_mols.append(mol)
        except:
            continue
    
    results['valid_molecules'] = len(valid_smiles)
    results['validity_rate'] = len(valid_smiles) / len(smiles_list) if smiles_list else 0
    
    # 计算唯一性
    unique_smiles = list(set(valid_smiles))
    results['unique_molecules'] = len(unique_smiles)
    results['uniqueness_rate'] = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0
    
    if valid_mols:
        # 计算分子描述符
        molecular_weights = []
        logp_values = []
        qed_values = []
        
        for mol in valid_mols:
            try:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                qed = QED.qed(mol)
                
                molecular_weights.append(mw)
                logp_values.append(logp)
                qed_values.append(qed)
            except:
                continue
        
        if molecular_weights:
            results['avg_molecular_weight'] = np.mean(molecular_weights)
            results['avg_logp'] = np.mean(logp_values)
            results['avg_qed'] = np.mean(qed_values)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="name of the generated dataset")
    args = parser.parse_args()

    try:
        # 读取CSV文件
        data = pd.read_csv(args.path)
        print(f"成功读取文件: {args.path}")
        print(f"数据形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
        
        # 检查是否有'smiles'列
        if 'smiles' not in data.columns:
            print("错误: CSV文件中没有'smiles'列")
            sys.exit(1)
        
        # 获取SMILES数据
        smiles_data = data['smiles'].tolist()
        print(f"SMILES数据数量: {len(smiles_data)}")
        
        # 计算基本指标
        print("开始计算基本分子指标...")
        metrics = calculate_basic_metrics(smiles_data)
        
        print(f"文件路径: {args.path}")
        print("="*50)
        print("分子生成指标结果:")
        print("="*50)
        print(f"总分子数: {metrics['total_molecules']}")
        print(f"有效分子数: {metrics['valid_molecules']}")
        print(f"唯一分子数: {metrics['unique_molecules']}")
        print(f"有效性率: {metrics['validity_rate']:.4f}")
        print(f"唯一性率: {metrics['uniqueness_rate']:.4f}")
        print(f"平均分子量: {metrics['avg_molecular_weight']:.2f}")
        print(f"平均LogP: {metrics['avg_logp']:.2f}")
        print(f"平均QED: {metrics['avg_qed']:.4f}")
        print('*'*50)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {args.path}")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)