import moses
import pandas as pd
import argparse
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="path to generated CSV file")
    args = parser.parse_args()

    try:
        # 读取生成的分子数据
        print("Loading generated molecules...")
        data = pd.read_csv(args.path)
        print(f"Loaded {len(data)} generated molecules")
        
        # 读取Moses训练数据集
        print("Loading Moses training dataset...")
        datasets_path = os.path.join(project_root, 'datasets', 'moses2.csv')
        moses_data = pd.read_csv(datasets_path)
        moses_data = moses_data.dropna(axis=0).reset_index(drop=True)
        moses_data.columns = moses_data.columns.str.lower()
        
        # 提取训练集、测试集和scaffold测试集SMILES
        train_data = moses_data[moses_data['split'] == 'train']
        test_data = moses_data[moses_data['split'] == 'test']
        test_scaffolds_data = moses_data[moses_data['split'] == 'test_scaffolds']
        
        train_smiles = train_data['smiles'].tolist()
        test_smiles = test_data['smiles'].tolist()
        test_scaffolds_smiles = test_scaffolds_data['smiles'].tolist()
        
        print(f"Loaded {len(train_smiles)} training molecules")
        print(f"Loaded {len(test_smiles)} test molecules")
        print(f"Loaded {len(test_scaffolds_smiles)} test scaffold molecules")
        
        # 提取生成的SMILES
        gen_smiles = data['smiles'].tolist()
        gen_smiles = [str(s) for s in gen_smiles if isinstance(s, str) and s.strip() != '']
        print(f"Processing {len(gen_smiles)} generated SMILES")
        
        if len(gen_smiles) == 0:
            print("Error: No valid SMILES found in generated data")
            sys.exit(1)
        
        # 运行Moses评估，提供所有必需的数据集
        print("Running Moses evaluation...")
        test_results = moses.get_all_metrics(
            gen_smiles, 
            train=train_smiles,
            test=test_smiles,
            test_scaffolds=test_scaffolds_smiles,
            device='cuda'
        )
        
        # 输出结果
        print("=" * 50)
        print("MOSES Evaluation Results:")
        print("=" * 50)
        for key, value in test_results.items():
            print(f"{key}: {value}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during Moses evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)