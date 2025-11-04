import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import pandas as pd
import argparse

from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import Chem
from generate.utils import check_novelty, canonic_smiles

# RDKit兼容性处理
try:
    from rdkit.Chem.rdMolDescriptors import MorganGenerator
    USE_NEW_MORGAN = True
except ImportError:
    # 旧版本RDKit使用GetMorganFingerprintAsBitVect
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    USE_NEW_MORGAN = False

def calculate_un(results, moses):

    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
    unique_smiles = list(set(canon_smiles))
    novel_ratio = check_novelty(unique_smiles, set(moses[moses['split']=='train']['smiles']))   # replace 'source' with 'split' for moses

    return len(unique_smiles)/len(results), novel_ratio


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=False)
    parser.add_argument('--path', type=str, help="path of csv", required=True)

    args = parser.parse_args()


    data = pd.read_csv(args.path)
    # 使用相对于脚本位置的数据集路径
    datasets_path = os.path.join(project_root, 'datasets', 'moses2.csv')
    moses = pd.read_csv(datasets_path)
    moses = moses.dropna(axis=0).reset_index(drop=True)
    moses.columns = moses.columns.str.lower()


    if 'scaffold_cond' in data.columns:


        data['scaffold_cond'] = data['scaffold_cond'].apply(lambda x: x.replace('<',''))
        data['mol_scaf'] = data['smiles'].apply(lambda x: MurckoScaffoldSmiles(x))
        
        # 兼容不同版本的RDKit
        if USE_NEW_MORGAN:
            morgan_gen = MorganGenerator(radius=2, fpSize=2048)
            data['fp'] = data['mol_scaf'].apply(lambda x: morgan_gen.GetFingerprint(Chem.MolFromSmiles(x)))
            data['cond_fp'] = data['scaffold_cond'].apply(lambda x: morgan_gen.GetFingerprint(Chem.MolFromSmiles(x)))
        else:
            # 旧版本RDKit
            data['fp'] = data['mol_scaf'].apply(lambda x: GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), radius=2, nBits=2048))
            data['cond_fp'] = data['scaffold_cond'].apply(lambda x: GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), radius=2, nBits=2048))
        data['similarity'] = -1

        for idx, row in data.iterrows():
            data.loc[idx, 'similarity'] = TanimotoSimilarity(row['fp'], row['cond_fp'])

        # Fraction of valid molecules having tanimoto similarity of conditional scaffold and generated mol scaffold as 1
        x = data['scaffold_cond'].value_counts()
        y = data[data['similarity'] == 1]['scaffold_cond'].value_counts()
        print(y.divide(x))

        new_df = []
        for cond in data['scaffold_cond'].unique():
            scaffold_samples = len(data[data['scaffold_cond'] == cond].reset_index(drop = True))
            results = data[(data['scaffold_cond'] == cond) & (data['similarity'] > 0.8)].reset_index(drop = True)
            val = len(results) / scaffold_samples
            previous_validity = results['validity'][0]
            uniqueness, novelty = calculate_un(results, moses)
            results['validity'] = val * previous_validity
            results['unique'] = uniqueness
            results['novelty'] = novelty
            new_df.append(results)

        data = pd.concat(new_df).reset_index(drop = True)

        avg_validity = data.groupby('scaffold_cond')['validity'].mean()
        avg_unique = data.groupby('scaffold_cond')['unique'].mean()
        avg_novelty = data.groupby('scaffold_cond')['novelty'].mean()

        print('Validity \n')
        print(avg_validity)
        print('\n Uniqueness \n')
        print(avg_unique)
        print('\n Novelty \n')
        print(avg_novelty)

        if len(args.props) == 1:
            data['difference'] = abs(data['condition'] - data[args.props[0]])
            print(f'\n Mean Absolute Difference: {args.props[0]} \n')
            print(data.groupby('scaffold_cond')['difference'].mean())
            print(f'\n Standard Deviation of the Difference: {args.props[0]} \n')
            print(data.groupby('scaffold_cond')['difference'].std())
        elif len(args.props) > 1:
            for idx, p in enumerate(args.props):
                data[f'{p}_condition'] = data['condition'].apply(lambda x: tuple(float(s) for s in x.strip("()").split(","))[idx])

                data['difference'] = abs(data[f'{p}_condition'] - data[p])
                print(f'\n Mean Absolute Difference: {p} \n')
                print(data.groupby('scaffold_cond')['difference'].mean())
                print(f'\n Standard Deviation of the Difference: {p} \n')
                print(data.groupby('scaffold_cond')['difference'].std())


    else:

        avg_validity = data['validity'].mean()
        avg_unique = data['unique'].mean()
        avg_novelty = data['novelty'].mean()

        print('Validity \n')
        print(avg_validity)
        print('\n Uniqueness \n')
        print(avg_unique)
        print('\n Novelty \n')
        print(avg_novelty)

        # 只有在有条件生成时才计算属性差异
        if len(args.props) > 0 and 'condition' in data.columns:
            if len(args.props) == 1:
                data['difference'] = abs(data['condition'] - data[args.props[0]])
                print(f'\n Mean Absolute Difference: {args.props[0]} \n')
                print(data['difference'].mean())
                print(f'\n Standard Deviation of the Difference: {args.props[0]} \n')
                print(data['difference'].std())
            else:
                for idx, p in enumerate(args.props):
                    data[f'{p}_condition'] = data['condition'].apply(lambda x: tuple(float(s) for s in x.strip("()").split(","))[idx])

                    data['difference'] = abs(data[f'{p}_condition'] - data[p])
                    print(f'\n Mean Absolute Difference: {p} \n')
                    print(data['difference'].mean())
                    print(f'\n Standard Deviation of the Difference: {p} \n')
                    print(data['difference'].std())
        elif len(args.props) == 0:
            print('\n This is unconditional generation evaluation - no property conditions to compare.')
        else:
            print('\n Warning: Properties specified but no condition column found in data.')
