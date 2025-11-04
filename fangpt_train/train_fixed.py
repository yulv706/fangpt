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

    smiles = train_data['smiles']
    vsmiles = val_data['smiles']

    # prop = train_data[['qed']]
    # vprop = val_data[['qed']]

    prop = train_data[args.props].values.tolist()
    vprop = val_data[args.props].values.tolist()
    num_props = args.num_props

    # 🔧 修复：只有在启用scaffold时才处理脚手架数据
    if args.scaffold:
        print("启用脚手架条件生成，正在处理脚手架数据...")
        scaffold = train_data['scaffold_smiles']
        vscaffold = val_data['scaffold_smiles']
        
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        # 计算分子序列的最大长度
        lens = [len(regex.findall(i.strip()))
                  for i in (list(smiles.values) + list(vsmiles.values))]
        max_len = max(lens)
        print('Max len: ', max_len)

        # 计算脚手架的最大长度
        lens = [len(regex.findall(i.strip()))
                for i in (list(scaffold.values) + list(vscaffold.values))]
        scaffold_max_len = max(lens)
        print('Scaffold max len: ', scaffold_max_len)

        # 对分子序列进行padding
        smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                    for i in smiles]
        vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                    for i in vsmiles]

        # 对脚手架序列进行padding
        scaffold = [i + str('<')*(scaffold_max_len -
                                    len(regex.findall(i.strip()))) for i in scaffold]
        vscaffold = [i + str('<')*(scaffold_max_len -
                                    len(regex.findall(i.strip()))) for i in vscaffold]
        
        print(f"脚手架条件生成配置：scaffold_maxlen={scaffold_max_len}")
        
    else:
        print("未启用脚手架条件生成，使用无脚手架配置...")
        scaffold = None
        vscaffold = None
        scaffold_max_len = 0
        
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        # 只计算分子序列的最大长度
        lens = [len(regex.findall(i.strip()))
                  for i in (list(smiles.values) + list(vsmiles.values))]
        max_len = max(lens)
        print('Max len: ', max_len)

        # 只对分子序列进行padding
        smiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                    for i in smiles]
        vsmiles = [i + str('<')*(max_len - len(regex.findall(i.strip())))
                    for i in vsmiles]
        
        print("无脚手架配置：scaffold_maxlen=0")

    # 词汇表保持不变
    whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[B]', '[C+]', '[C-]', '[CH+]', '[CH-]', '[CH2+]', '[CH2]', '[CH]', '[F+]', '[H]', '[I+]', '[IH2]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]', '[N]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]', '[PH+]', '[PH2+]', '[PH]', '[S+]', '[S-]', '[SH+]', '[SH]', '[Se+]', '[SeH+]', '[SeH]', '[Se]', '[Si-]', '[SiH-]', '[SiH2]', '[SiH]', '[Si]', '[b-]', '[bH-]', '[c+]', '[c-]', '[cH+]', '[cH-]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[sH+]', '[se+]', '[se]', 'b', 'c', 'n', 'o', 'p', 's']

    # 🔧 修复：数据集创建时正确处理脚手架参数
    if args.scaffold:
        train_dataset = SmileDataset(args, smiles, whole_string, max_len, 
                                   prop=prop, aug_prob=0, 
                                   scaffold=scaffold, 
                                   scaffold_maxlen=scaffold_max_len)
        valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, 
                                   prop=vprop, aug_prob=0, 
                                   scaffold=vscaffold, 
                                   scaffold_maxlen=scaffold_max_len)
    else:
        # 无脚手架时，创建dummy scaffold以保持数据集兼容性
        dummy_scaffold_train = ['<'] * len(smiles)
        dummy_scaffold_valid = ['<'] * len(vsmiles)
        train_dataset = SmileDataset(args, smiles, whole_string, max_len, 
                                   prop=prop, aug_prob=0, 
                                   scaffold=dummy_scaffold_train, 
                                   scaffold_maxlen=scaffold_max_len)
        valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, 
                                   prop=vprop, aug_prob=0, 
                                   scaffold=dummy_scaffold_valid, 
                                   scaffold_maxlen=scaffold_max_len)

    # 🔧 修复：模型配置时正确设置脚手架参数
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len, 
                     num_props=num_props,  # args.num_props,
                     n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                     scaffold=args.scaffold,  # 使用用户指定的scaffold参数
                     scaffold_maxlen=scaffold_max_len,  # 只有启用scaffold时才非零
                     lstm=args.lstm, lstm_layers=args.lstm_layers)
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
        'n_embd': args.n_embd
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