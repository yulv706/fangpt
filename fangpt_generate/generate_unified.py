#!/usr/bin/env python3
"""
统一分子生成脚本
自动检测模型配置，支持无条件、属性条件、脚手架条件和混合条件生成
"""

from utils import check_novelty, sample, canonic_smiles
from dataset import SmileDataset
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import moses
import json
import warnings
import os
import sys

# 抑制RDKit的弃用警告
warnings.filterwarnings("ignore", message=".*DEPRECATION WARNING.*")
from rdkit.Chem import RDConfig
from rdkit import Chem

# 导入SA Score
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.rdMolDescriptors import CalcTPSA

def get_mol(smiles_or_mol):
    """将SMILES字符串转换为RDKit分子对象"""
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def normalize_symbol(symbol):
    if symbol is None:
        return ''
    symbol = str(symbol).strip().replace('[', '').replace(']', '')
    if not symbol:
        return ''
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def dedupe_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def parse_atom_condition(atom_symbols, condition_args, on_value=1.0, off_value=0.0):
    if not atom_symbols:
        return None
    if not condition_args:
        return [on_value] * len(atom_symbols)
    try:
        values = [float(v) for v in condition_args]
        if len(values) != len(atom_symbols):
            raise ValueError
        return values
    except ValueError:
        normalized_targets = {normalize_symbol(v) for v in condition_args if normalize_symbol(v)}
        if not normalized_targets or normalized_targets == {'None'}:
            return [off_value] * len(atom_symbols)
        return [on_value if sym in normalized_targets else off_value for sym in atom_symbols]

def detect_model_config(weight_path):
    """自动检测模型配置"""
    print("🔍 正在自动检测模型配置...")
    
    checkpoint_data = torch.load(weight_path, map_location='cpu', weights_only=False)
    
    # 🔧 检查是否为新格式（包含训练配置）
    if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
        print("🆕 检测到新格式权重文件（包含训练配置）")
        checkpoint = checkpoint_data['model_state_dict']
        training_config = checkpoint_data.get('training_config', {})
        
        # 直接使用保存的训练配置
        if training_config:
            print("✅ 使用保存的训练配置信息")
            # 🔧 修复：只有当num_props > 0时才显示props，避免混淆
            num_props = training_config.get('num_props', 0)
            props_list = training_config.get('props', []) if num_props > 0 else []
            
            atom_cond = training_config.get('atom_cond', False)
            atom_list = training_config.get('atom_list', []) if atom_cond else []
            atom_vocab = training_config.get('atom_vocab_size', len(atom_list)) if atom_cond else 0
            config = {
                'vocab_size': training_config.get('vocab_size', 26),
                'n_embd': training_config.get('n_embd', 256),
                'block_size': training_config.get('block_size', 54),
                'n_layer': training_config.get('n_layer', 8),
                'n_head': training_config.get('n_head', 8),
                'num_props': num_props,
                'has_props': num_props > 0,
                'scaffold_maxlen': training_config.get('scaffold_maxlen', 0),
                'uses_scaffold': training_config.get('scaffold', False),
                'has_lstm': training_config.get('lstm', False),
                'lstm_layers': training_config.get('lstm_layers', 2),
                'props': props_list,
                'data_name': training_config.get('data_name', 'moses2'),
                'atom_cond': atom_cond,
                'atom_list': atom_list,
                'atom_vocab_size': atom_vocab,
                'mask_size': training_config.get('block_size', 54) + 
                           int(num_props > 0) + 
                           training_config.get('scaffold_maxlen', 0) + 
                           (1 if atom_cond else 0)
            }
            return config
    else:
        print("🔄 检测到旧格式权重文件（仅包含模型权重）")
        checkpoint = checkpoint_data
    
    # 旧格式的启发式检测逻辑
    # 基本参数
    vocab_size = checkpoint['tok_emb.weight'].shape[0]
    n_embd = checkpoint['tok_emb.weight'].shape[1]
    block_size = checkpoint['pos_emb'].shape[1]
    
    # 获取掩码大小
    mask_keys = [k for k in checkpoint.keys() if 'attn.mask' in k]
    if mask_keys:
        mask_size = checkpoint[mask_keys[0]].shape[-1]
    else:
        mask_size = block_size
    
    # 层数
    layer_keys = [k for k in checkpoint.keys() if k.startswith('blocks.')]
    layer_nums = set()
    for key in layer_keys:
        parts = key.split('.')
        if len(parts) >= 2 and parts[1].isdigit():
            layer_nums.add(int(parts[1]))
    n_layer = max(layer_nums) + 1 if layer_nums else 8
    
    # 注意力头数（简化推断）
    n_head = 8 if n_embd % 8 == 0 else 1
    
    # 检查条件生成特征
    has_prop_nn = any('prop_nn' in k for k in checkpoint.keys())
    has_lstm = any('lstm' in k.lower() for k in checkpoint.keys())
    
    # 分析掩码配置：mask_size = block_size + int(bool(num_props)) + scaffold_maxlen
    extra_size = mask_size - block_size
    
    if has_prop_nn:
        num_props = 1  # 简化假设单属性
        scaffold_maxlen = extra_size - 1
    else:
        num_props = 0
        scaffold_maxlen = extra_size
    
    # 确保scaffold_maxlen合理
    if scaffold_maxlen < 0:
        scaffold_maxlen = 0
    
    config = {
        'vocab_size': vocab_size,
        'n_embd': n_embd,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'num_props': num_props,
        'has_props': has_prop_nn,
        'scaffold_maxlen': scaffold_maxlen,
        'uses_scaffold': scaffold_maxlen > 0,
        'has_lstm': has_lstm,
        'mask_size': mask_size,
        'props': [],  # 旧格式无法推断具体属性
        'data_name': 'moses2' if vocab_size == 26 else 'guacamol2',
        'atom_cond': False,
        'atom_list': [],
        'atom_vocab_size': 0
    }

    return config

def create_dummy_scaffold(scaffold_maxlen, stoi):
    """创建填充的虚拟脚手架"""
    if scaffold_maxlen <= 0:
        return None
    dummy_scaffold = '<' * scaffold_maxlen
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    return torch.tensor([stoi[s] for s in regex.findall(dummy_scaffold)], dtype=torch.long)

def generate_molecules(model, stoi, itos, block_size, batch_size, gen_iter, 
                      prop_tensor=None, scaffold_tensor=None, atom_tensor=None, context="C"):
    """生成分子的核心函数"""
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    molecules = []
    
    for i in tqdm(range(gen_iter), desc="生成中"):
        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(batch_size, 1).to('cuda')
        
        # 准备属性和脚手架输入
        p = prop_tensor.repeat(batch_size, 1).to('cuda') if prop_tensor is not None else None
        sca = scaffold_tensor[None, ...].repeat(batch_size, 1).to('cuda') if scaffold_tensor is not None else None
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        # 生成
        atom = None
        if atom_tensor is not None:
            atom = atom_tensor[None, ...].repeat(batch_size, 1).to('cuda')

        with torch.no_grad():
            y = sample(model, x, block_size, temperature=1, sample=True, top_k=None, prop=p, scaffold=sca, atom_cond=atom)
        
        # 解码分子
        for gen_mol in y:
            completion = ''.join([itos[int(i)] for i in gen_mol])
            completion = completion.replace('<', '')
            mol = get_mol(completion)
            if mol:
                molecules.append(mol)
    
    return molecules

def calculate_metrics(molecules, results_df, data, data_name, batch_size, gen_iter):
    """计算生成指标"""
    # 计算去重和新颖性
    canon_smiles = [canonic_smiles(s) for s in results_df['smiles']]
    unique_smiles = list(set(canon_smiles))
    
    if 'moses' in data_name:
        novel_ratio = check_novelty(unique_smiles, set(data[data['split'] == 'train']['smiles']))
    else:
        novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))
    
    # 计算分子性质
    results_df['qed'] = results_df['molecule'].apply(lambda x: QED.qed(x))
    results_df['sas'] = results_df['molecule'].apply(lambda x: sascorer.calculateScore(x))
    results_df['logp'] = results_df['molecule'].apply(lambda x: Crippen.MolLogP(x))
    results_df['tpsa'] = results_df['molecule'].apply(lambda x: CalcTPSA(x))
    
    # 计算比率
    validity = len(results_df) / (batch_size * gen_iter)
    uniqueness = len(unique_smiles) / len(results_df) if len(results_df) > 0 else 0
    novelty = novel_ratio / 100
    
    results_df['validity'] = np.round(validity, 3)
    results_df['unique'] = np.round(uniqueness, 3)
    results_df['novelty'] = np.round(novelty, 3)
    
    return results_df, validity, uniqueness, novelty

def main():
    parser = argparse.ArgumentParser(description='统一分子生成脚本')
    
    # 必需参数
    parser.add_argument('--model_weight', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--csv_name', type=str, required=True, help='输出CSV文件名')
    
    # 生成参数
    parser.add_argument('--data_name', type=str, default='moses2', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
    parser.add_argument('--gen_size', type=int, default=10000, help='生成分子总数')
    
    # 条件生成参数
    parser.add_argument('--props', nargs='+', default=[], help='属性条件')
    parser.add_argument('--scaffold', action='store_true', help='启用脚手架条件生成')
    parser.add_argument('--lstm', action='store_true', help='使用LSTM处理脚手架')
    parser.add_argument('--lstm_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--atom_list', nargs='+', default=None, help='训练时使用的原子列表（按顺序）')
    parser.add_argument('--atom_condition', nargs='+', default=None, help='原子条件向量或需要激活的原子符号')
    parser.add_argument('--atom_on_value', type=float, default=1.0, help='激活原子的取值')
    parser.add_argument('--atom_off_value', type=float, default=0.0, help='未激活原子的取值')
    
    # 可选的手动配置参数
    parser.add_argument('--vocab_size', type=int, default=None, help='词汇表大小（自动检测）')
    parser.add_argument('--block_size', type=int, default=None, help='序列长度（自动检测）')
    parser.add_argument('--n_layer', type=int, default=None, help='层数（自动检测）')
    parser.add_argument('--n_head', type=int, default=None, help='注意力头数（自动检测）')
    parser.add_argument('--n_embd', type=int, default=None, help='嵌入维度（自动检测）')
    
    args = parser.parse_args()

    # 自动检测模型配置
    detected_config = detect_model_config(args.model_weight)
    
    print("📋 检测到的模型配置:")
    for key, value in detected_config.items():
        print(f"  {key}: {value}")
    
    # 使用检测到的配置，允许命令行参数覆盖
    vocab_size = args.vocab_size or detected_config['vocab_size']
    block_size = args.block_size or detected_config['block_size']
    n_layer = args.n_layer or detected_config['n_layer']
    n_head = args.n_head or detected_config['n_head']
    n_embd = args.n_embd or detected_config['n_embd']

    detected_atom_list = detected_config.get('atom_list', [])
    atom_candidates = args.atom_list if args.atom_list is not None else detected_atom_list
    atom_symbols = dedupe_preserve_order([normalize_symbol(a) for a in atom_candidates or [] if normalize_symbol(a)])
    atom_cond_enabled = detected_config.get('atom_cond', bool(atom_symbols))
    if atom_cond_enabled and not atom_symbols and detected_atom_list:
        atom_symbols = dedupe_preserve_order([normalize_symbol(a) for a in detected_atom_list if normalize_symbol(a)])
        atom_cond_enabled = bool(atom_symbols)
    elif not atom_symbols:
        atom_cond_enabled = False

    atom_condition_values = parse_atom_condition(
        atom_symbols,
        args.atom_condition,
        on_value=args.atom_on_value,
        off_value=args.atom_off_value
    ) if atom_cond_enabled else None
    atom_condition_tensor = torch.tensor(atom_condition_values, dtype=torch.float) if atom_condition_values is not None else None
    
    # 加载数据和词汇表
    data = pd.read_csv(f'datasets/{args.data_name}.csv')
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()
    
    # 根据检测到的vocab_size选择正确的词汇表
    detected_vocab_size = detected_config['vocab_size']
    vocab_file_candidates = [
        f'{args.data_name}_stoi.json',
        'guacamol2_stoi.json',
        'moses2_stoi.json'
    ]
    
    stoi = None
    for vocab_file in vocab_file_candidates:
        if os.path.exists(vocab_file):
            test_stoi = json.load(open(vocab_file, 'r'))
            if len(test_stoi) == detected_vocab_size:
                stoi = test_stoi
                print(f"✅ 找到匹配的词汇表: {vocab_file} (大小: {len(stoi)})")
                break
    
    if stoi is None:
        print(f"❌ 警告：无法找到匹配的词汇表，使用默认的 {args.data_name}_stoi.json")
        stoi = json.load(open(f'{args.data_name}_stoi.json', 'r'))
    
    itos = {i: ch for ch, i in stoi.items()}
    
    print(f"\n📖 数据集信息:")
    print(f"  数据集: {args.data_name}")
    print(f"  词汇表大小: {len(itos)}")
    if atom_cond_enabled:
        print(f"  原子条件: {atom_symbols}")
        if atom_condition_values is not None:
            print(f"  原子向量: {atom_condition_values}")
    else:
        print("  原子条件: 未启用")
    
    # 🔧 新增：处理属性条件自动检测
    # 准备属性条件值
    if 'guacamol' in args.data_name:
        single_prop_values = {
            'qed': [0.3, 0.5, 0.7], 
            'sas': [2.0, 3.0, 4.0], 
            'logp': [2.0, 4.0, 6.0], 
            'tpsa': [40.0, 80.0, 120.0]
        }
    else:
        single_prop_values = {
            'qed': [0.6, 0.725, 0.85], 
            'sas': [2.0, 2.75, 3.5], 
            'logp': [1.0, 2.0, 3.0], 
            'tpsa': [30.0, 60.0, 90.0]
        }
    
    def create_prop_conditions(props_list):
        """创建属性条件列表"""
        if len(props_list) == 1:
            # 单属性：直接使用数值列表
            return single_prop_values.get(props_list[0], [0.5])
        else:
            # 多属性：生成组合向量
            prop_values_lists = []
            for prop in props_list:
                prop_values_lists.append(single_prop_values.get(prop, [0.5]))
            
            # 取每个属性的对应值构成向量
            prop_conditions = []
            for i in range(min(len(lst) for lst in prop_values_lists)):
                prop_vector = [lst[i] for lst in prop_values_lists]
                prop_conditions.append(prop_vector)
            return prop_conditions
    
    # 确定生成条件
    prop_conditions = None
    if args.props:
        # 用户指定了属性
        prop_conditions = create_prop_conditions(args.props)
    elif detected_config.get('props') and detected_config.get('num_props', 0) > 0:
        # 从模型配置中检测到属性（新格式），并且模型确实有属性支持
        detected_props = detected_config['props']
        print(f"🔍 从模型配置中检测到属性: {detected_props}")
        prop_conditions = create_prop_conditions(detected_props)
        args.props = detected_props  # 更新args以便后续使用
        print(f"✅ 自动设置属性条件: {detected_props}")
    elif detected_config.get('num_props', 0) == 0:
        # 无条件模型，不设置任何属性条件
        print("🔍 检测到无条件模型，不使用属性条件")
        args.props = []  # 确保为空列表
    
    # 配置模型
    num_props = len(args.props) if args.props else 0
    model_uses_scaffold = detected_config['uses_scaffold']
    scaffold_maxlen = detected_config['scaffold_maxlen']
    
    # 如果用户明确要求使用脚手架但模型不支持，给出警告
    if args.scaffold and not model_uses_scaffold:
        print("⚠️  警告：用户要求脚手架生成，但模型未配置脚手架支持")
        print("   将尝试使用无脚手架模式进行生成")
        args.scaffold = False
    
    print(f"\n🎯 生成配置:")
    print(f"  属性条件: {args.props if args.props else '无'}")
    print(f"  脚手架条件: {'是' if args.scaffold else '否'}")
    print(f"  模型脚手架支持: {'是' if model_uses_scaffold else '否'}")
    print(f"  脚手架最大长度: {scaffold_maxlen}")
    
    # 创建模型
    mconf = GPTConfig(vocab_size, block_size, num_props=num_props,
                     n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                     scaffold=model_uses_scaffold, scaffold_maxlen=scaffold_maxlen,
                     lstm=args.lstm, lstm_layers=args.lstm_layers,
                     atom_cond=atom_cond_enabled and bool(atom_symbols), atom_vocab_size=len(atom_symbols))
    model = GPT(mconf)
    
    # 加载权重
    checkpoint_data = torch.load(args.model_weight, map_location='cpu', weights_only=False)
    if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
        # 新格式：包含训练配置
        model_state_dict = checkpoint_data['model_state_dict']
    else:
        # 旧格式：直接是state_dict
        model_state_dict = checkpoint_data
    
    model.load_state_dict(model_state_dict)
    model.to('cuda')
    print('✅ 模型加载成功')
    
    # 准备生成参数
    gen_iter = math.ceil(args.gen_size / args.batch_size)
    context = "C"
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    # 🔧 注意：属性条件已在上面的自动检测逻辑中处理
    
    scaffold_conditions = None
    if args.scaffold:
        # 预定义的脚手架条件
        base_scaffolds = [
            'O=C(Cc1ccccc1)NCc1ccccc1', 
            'c1cnc2[nH]ccc2c1', 
            'c1ccc(-c2ccnnc2)cc1', 
            'c1ccc(-n2cnc3ccccc32)cc1', 
            'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1'
        ]
        scaffold_conditions = [s + '<' * (scaffold_maxlen - len(regex.findall(s))) for s in base_scaffolds]
    
    # 确定生成模式
    if not prop_conditions and not scaffold_conditions:
        generation_mode = "无条件生成"
    elif prop_conditions and not scaffold_conditions:
        generation_mode = f"属性条件生成 ({','.join(args.props)})"
    elif not prop_conditions and scaffold_conditions:
        generation_mode = "脚手架条件生成"
    else:
        generation_mode = f"混合条件生成 ({','.join(args.props)} + 脚手架)"
    
    print(f"\n🚀 开始{generation_mode}...")
    
    # 执行生成
    all_results = []
    
    # 确定循环条件
    prop_loop = prop_conditions or [None]
    scaffold_loop = scaffold_conditions or [None]
    
    for prop_cond in prop_loop:
        for scaffold_cond in scaffold_loop:
            # 准备条件描述
            cond_desc = []
            if prop_cond is not None:
                if len(args.props) == 1:
                    cond_desc.append(f"属性={prop_cond}")
                else:
                    # 多属性：显示属性名和对应值
                    prop_str = ", ".join([f"{prop}={val}" for prop, val in zip(args.props, prop_cond)])
                    cond_desc.append(f"属性=({prop_str})")
            if scaffold_cond is not None:
                cond_desc.append(f"脚手架={scaffold_cond[:20]}...")
            
            desc = ", ".join(cond_desc) if cond_desc else "无条件"
            print(f"\n生成条件: {desc}")
            
            # 准备输入张量
            prop_tensor = None
            if prop_cond is not None:
                if len(args.props) == 1:
                    prop_tensor = torch.tensor([[prop_cond]])
                else:
                    # 多属性：prop_cond 已经是向量，直接构造张量
                    prop_tensor = torch.tensor([prop_cond])
            
            scaffold_tensor = None
            if scaffold_cond is not None:
                scaffold_tensor = torch.tensor([stoi[s] for s in regex.findall(scaffold_cond)], dtype=torch.long)
            elif model_uses_scaffold:
                # 模型需要脚手架输入但用户未提供，使用虚拟脚手架
                scaffold_tensor = create_dummy_scaffold(scaffold_maxlen, stoi)
            
            # 生成分子
            molecules = generate_molecules(
                model, stoi, itos, block_size, args.batch_size, gen_iter,
                prop_tensor, scaffold_tensor, atom_condition_tensor, context
            )
            
            print(f"有效分子数: {len(molecules)}")
            
            # 创建结果DataFrame
            if molecules:
                mol_dict = [{'molecule': mol, 'smiles': Chem.MolToSmiles(mol)} for mol in molecules]
                results_df = pd.DataFrame(mol_dict)
                
                # 添加条件信息
                if prop_cond is not None:
                    if len(args.props) == 1:
                        results_df['condition'] = prop_cond
                    else:
                        # 多属性：保存为字符串格式
                        prop_str = "_".join([f"{prop}={val}" for prop, val in zip(args.props, prop_cond)])
                        results_df['condition'] = prop_str
                
                if scaffold_cond is not None:
                    results_df['scaffold_cond'] = scaffold_cond
                
                # 计算指标
                results_df, validity, uniqueness, novelty = calculate_metrics(
                    molecules, results_df, data, args.data_name, args.batch_size, gen_iter
                )
                
                print(f'有效性: {validity:.3f}')
                print(f'唯一性: {uniqueness:.3f}')
                print(f'新颖性: {novelty:.3f}')
                
                all_results.append(results_df)
    
    # 合并所有结果
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        
        # 确保输出目录存在
        if '/' not in args.csv_name:
            os.makedirs('generated_csvs', exist_ok=True)
            output_path = os.path.join('generated_csvs', args.csv_name)
        else:
            output_path = args.csv_name
        
        # 保存结果
        final_results.to_csv(f'{output_path}.csv', index=False)
        
        # 计算总体指标
        canon_smiles = [canonic_smiles(s) for s in final_results['smiles']]
        unique_smiles = list(set(canon_smiles))
        if 'moses' in args.data_name:
            novel_ratio = check_novelty(unique_smiles, set(data[data['split'] == 'train']['smiles']))
        else:
            novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))
        
        total_expected = args.batch_size * gen_iter * len(prop_loop) * len(scaffold_loop)
        
        print('\n=== 🎉 最终统计 ===')
        print(f'生成模式: {generation_mode}')
        print(f'总分子数: {len(final_results)}')
        print(f'总体有效性: {len(final_results)/total_expected:.3f}')
        print(f'总体唯一性: {len(unique_smiles)/len(final_results):.3f}')
        print(f'总体新颖性: {novel_ratio/100:.3f}')
        print(f'结果已保存到: {output_path}.csv')
    else:
        print("❌ 未生成任何有效分子")

if __name__ == '__main__':
    main() 
