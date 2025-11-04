import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from dataset import SmileDataset
import math
from utils import SmilesEnumerator
import re
from rdkit import Chem

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str,
                        help="name for wandb run", required=False)
    parser.add_argument('--debug', action='store_true',
                        default=False, help='debug')
    # in moses dataset, on average, there are only 5 molecules per scaffold
    parser.add_argument('--scaffold', action='store_true',
                        default=False, help='condition on scaffold')
    parser.add_argument('--lstm', action='store_true',
                        default=False, help='use lstm for transforming scaffold')
    parser.add_argument('--data_name', type=str, default='moses2',
                        help="name of the dataset to train on", required=False)
    # parser.add_argument('--property', type=str, default = 'qed', help="which property to use for condition", required=False)
    parser.add_argument('--props', nargs="+", default=['qed'],
                        help="properties to be used for condition", required=False)
    parser.add_argument('--num_props', type=int, default = 0, help="number of properties to use for condition", required=False)
    parser.add_argument('--atom_list', nargs="+", default=[],
                        help="atom symbols to be used for conditioning (e.g. Pd Fe)", required=False)
    parser.add_argument('--atom_column', type=str, default=None,
                        help="optional column in the dataset providing target atom labels", required=False)
    # parser.add_argument('--prop1_unique', type=int, default = 0, help="unique values in that property", required=False)
    parser.add_argument('--n_layer', type=int, default=8,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=512,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int,
                        default=6e-4, help="learning rate", required=False)
    parser.add_argument('--lstm_layers', type=int, default=0,
                        help="number of layers in lstm", required=False)

    args = parser.parse_args()

    set_seed(42)

    wandb.init(project="lig_gpt", name=args.run_name)

    data = pd.read_csv('datasets/' + args.data_name + '.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    # data = data.sample(frac = 0.1).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    # 🔧 优化：统一处理不同数据集的列名差异
    if 'moses' in args.data_name:
        # Moses数据集使用'split'列
        train_data = data[data['split'] == 'train'].reset_index(drop=True)
        val_data = data[data['split'] == 'test'].reset_index(drop=True)  # Moses用test作为验证集
    else:
        # GuacaMol数据集使用'source'列
        train_data = data[data['source'] == 'train'].reset_index(drop=True)
        val_data = data[data['source'] == 'val'].reset_index(drop=True)   # GuacaMol用val作为验证集

    # train_data = train_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)
    # val_data = val_data.sample(frac = 0.1, random_state = 42).reset_index(drop=True)

    token_pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    token_regex = re.compile(token_pattern)

    smiles = train_data['smiles'].tolist()
    vsmiles = val_data['smiles'].tolist()
    raw_train_smiles = list(smiles)
    raw_val_smiles = list(vsmiles)

    prop_columns = [col.lower() for col in args.props] if args.props else []
    if prop_columns:
        missing_props = [col for col in prop_columns if col not in train_data.columns]
        if missing_props:
            raise ValueError(f"Property columns not found: {missing_props}")
        prop = train_data[prop_columns].astype(float).values.tolist()
        vprop = val_data[prop_columns].astype(float).values.tolist()
    else:
        prop = None
        vprop = None

    inferred_props = len(prop_columns) if prop_columns else 0
    num_props = args.num_props if args.num_props else inferred_props

    if args.scaffold:
        print("启用脚手架条件生成，正在处理脚手架数据...")
        scaffold = train_data['scaffold_smiles'].fillna('<').tolist()
        vscaffold = val_data['scaffold_smiles'].fillna('<').tolist()
    else:
        print("未启用脚手架条件生成，使用无脚手架配置...")
        scaffold = ['<'] * len(smiles)
        vscaffold = ['<'] * len(vsmiles)

    def normalize_symbol(symbol):
        if symbol is None:
            return ''
        symbol = str(symbol).strip().replace('[', '').replace(']', '')
        if not symbol:
            return ''
        if len(symbol) == 1:
            return symbol.upper()
        return symbol[0].upper() + symbol[1:].lower()

    atom_symbols = [normalize_symbol(atom) for atom in args.atom_list if atom and normalize_symbol(atom)]
    seen_atoms = set()
    normalized_atoms = []
    for atom in atom_symbols:
        if atom not in seen_atoms:
            normalized_atoms.append(atom)
            seen_atoms.add(atom)
    atom_symbols = normalized_atoms
    atom_to_idx = {atom: idx for idx, atom in enumerate(atom_symbols)}
    atom_column = args.atom_column.lower() if args.atom_column else None

    def encode_atom_vector(smiles_str):
        if not atom_symbols:
            return []
        vector = [0.0] * len(atom_symbols)
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None:
            return vector
        for atom in mol.GetAtoms():
            sym = normalize_symbol(atom.GetSymbol())
            if sym in atom_to_idx:
                vector[atom_to_idx[sym]] = 1.0
        return vector

    def encode_atom_label(label):
        if not atom_symbols:
            return []
        vector = [0.0] * len(atom_symbols)
        if label is None or pd.isna(label):
            return vector
        if isinstance(label, (list, tuple, set)):
            tokens = label
        else:
            tokens = re.split(r'[,\s;|]+', str(label))
        for token in tokens:
            norm = normalize_symbol(token)
            if norm and norm in atom_to_idx:
                vector[atom_to_idx[norm]] = 1.0
        return vector

    def build_atom_features(smiles_list, label_list=None):
        if not atom_symbols:
            return None
        features = []
        labels = label_list or [None] * len(smiles_list)
        for smi, label in zip(smiles_list, labels):
            vector = encode_atom_vector(smi)
            label_vector = encode_atom_label(label)
            if label_vector and any(label_vector):
                vector = label_vector
            features.append(vector)
        return features

    train_atom_features = None
    val_atom_features = None
    if atom_symbols:
        if atom_column and atom_column not in train_data.columns:
            raise ValueError(f"Atom label column '{atom_column}' not found in dataset.")
        train_labels = train_data[atom_column].tolist() if atom_column else None
        val_labels = val_data[atom_column].tolist() if atom_column else None
        train_atom_features = build_atom_features(raw_train_smiles, train_labels)
        val_atom_features = build_atom_features(raw_val_smiles, val_labels)
        print(f"启用原子条件生成，目标原子: {atom_symbols}")
    else:
        print("未启用原子条件生成")

    lens = [len(token_regex.findall(i.strip()))
            for i in (smiles + vsmiles)]
    max_len = max(lens)
    print('Max len: ', max_len)

    if args.scaffold:
        lens = [len(token_regex.findall(i.strip()))
                for i in (scaffold + vscaffold)]
        scaffold_max_len = max(lens)
        print('Scaffold max len: ', scaffold_max_len)
    else:
        scaffold_max_len = 0
        print("无脚手架配置：scaffold_maxlen=0")

    smiles = [i + str('<') * (max_len - len(token_regex.findall(i.strip())))
                for i in smiles]
    vsmiles = [i + str('<') * (max_len - len(token_regex.findall(i.strip())))
                for i in vsmiles]

    scaffold = [i + str('<') * (scaffold_max_len -
                                len(token_regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [i + str('<') * (scaffold_max_len -
                                len(token_regex.findall(i.strip()))) for i in vscaffold]

    # 词汇表保持不变
    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']

    # 🔧 修复：数据集创建时正确处理脚手架参数
    train_dataset = SmileDataset(args, smiles, whole_string, max_len,
                                   prop=prop, aug_prob=0,
                                   scaffold=scaffold,
                                   scaffold_maxlen=scaffold_max_len,
                                   atom_features=train_atom_features)
    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len,
                                   prop=vprop, aug_prob=0,
                                   scaffold=vscaffold,
                                   scaffold_maxlen=scaffold_max_len,
                                   atom_features=val_atom_features)

    # 🔧 修复：模型配置时正确设置脚手架参数
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, 
                     num_props=num_props,  # args.num_props,
                     n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                     scaffold=args.scaffold,  # 使用用户指定的scaffold参数
                     scaffold_maxlen=scaffold_max_len,  # 只有启用scaffold时才非零
                     lstm=args.lstm, lstm_layers=args.lstm_layers,
                     atom_cond=bool(atom_symbols), atom_vocab_size=len(atom_symbols))
    model = GPT(mconf)

    # 🔧 打印配置信息以便验证
    print(f"\n=== 模型配置 ===")
    print(f"vocab_size: {train_dataset.vocab_size}")
    print(f"max_len: {train_dataset.max_len}")
    print(f"num_props: {num_props}")
    print(f"scaffold: {args.scaffold}")
    print(f"scaffold_maxlen: {scaffold_max_len}")
    print(f"n_layer: {args.n_layer}")
    print(f"n_head: {args.n_head}")
    print(f"n_embd: {args.n_embd}")
    
    # 计算预期的掩码大小
    expected_mask_size = train_dataset.max_len + int(bool(num_props)) + scaffold_max_len
    print(f"预期掩码大小: {expected_mask_size}x{expected_mask_size}")
    print(f"掩码计算: {train_dataset.max_len} (序列) + {int(bool(num_props))} (属性) + {scaffold_max_len} (脚手架) = {expected_mask_size}")

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len, final_tokens=args.max_epochs*len(train_data)*max_len,
                            num_workers=10, ckpt_path=f'weights/{args.run_name}.pt', block_size=train_dataset.max_len, generate=False)
    
    # 🔧 新增：保存训练配置信息
    training_config = {
        'props': args.props,
        'num_props': num_props,
        'scaffold': args.scaffold,
        'scaffold_maxlen': scaffold_max_len,
        'lstm': args.lstm,
        'lstm_layers': args.lstm_layers,
        'data_name': args.data_name,
        'vocab_size': train_dataset.vocab_size,
        'block_size': train_dataset.max_len,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
        'atom_cond': bool(atom_symbols),
        'atom_list': atom_symbols,
        'atom_column': atom_column
    }
    
    trainer = Trainer(model, train_dataset, valid_dataset,
                        tconf, train_dataset.stoi, train_dataset.itos, training_config)
    df = trainer.train(wandb)

    if df is not None:
        df.to_csv(f'{args.run_name}.csv', index=False)
    else:
        print(f"Training completed successfully. Model saved to {tconf.ckpt_path}")
    
    print(f"\n=== 训练完成 ===")
    print(f"模型权重保存到: {tconf.ckpt_path}")
    if args.scaffold:
        print("⚠️  这是一个脚手架条件生成模型")
        print(f"生成时需要使用: --scaffold --scaffold_maxlen {scaffold_max_len}")
    else:
        print("✅ 这是一个无条件/属性条件生成模型")
        print("生成时可以直接使用，无需额外的脚手架参数") 
