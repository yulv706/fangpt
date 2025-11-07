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

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from rdkit.Chem.rdMolDescriptors import CalcTPSA


def get_mol(smiles_or_mol):
    """Loads SMILES/molecule into RDKit's object."""
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


def parse_atom_condition_args(atom_symbols, condition_args, on_value=1.0, off_value=0.0):
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


def load_stoi_with_fallback(data_name, expected_vocab):
    candidates = [f'{data_name}_stoi.json', 'guacamol2_stoi.json', 'moses2_stoi.json']
    inspected = []
    for path in candidates:
        if path in inspected or not os.path.exists(path):
            continue
        inspected.append(path)
        with open(path, 'r') as handle:
            vocab = json.load(handle)
        if len(vocab) == expected_vocab:
            print(f"Using vocabulary from {path} (size={len(vocab)})")
            return vocab
    tried = ', '.join(inspected) if inspected else 'none'
    raise ValueError(f"Unable to find vocabulary file matching size {expected_vocab}. Tried: {tried}")


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, help="path of model weights", required=True)
        parser.add_argument('--scaffold', action='store_true', default=False, help='condition on scaffold')
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--csv_name', type=str, help="name to save the generated mols in csv format", required=True)
        parser.add_argument('--data_name', type=str, default='moses2', help="name of the dataset to train on", required=False)
        parser.add_argument('--batch_size', type=int, default=512, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default=10000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default=26, help="token vocabulary size", required=False)
        parser.add_argument('--block_size', type=int, default=54, help="sequence length", required=False)
        parser.add_argument('--props', nargs="+", default=[], help="properties to be used for condition", required=False)
        parser.add_argument('--atom_list', nargs="+", default=None, help="atom symbols used during training (order-sensitive)")
        parser.add_argument('--atom_condition', nargs="+", default=None, help="atom conditioning vector or target atom symbols")
        parser.add_argument('--atom_on_value', type=float, default=1.0, help="value assigned to active atom targets")
        parser.add_argument('--atom_off_value', type=float, default=0.0, help="value assigned to inactive atom targets")
        parser.add_argument('--n_layer', type=int, default=8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default=8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default=256, help="embedding dimension", required=False)
        parser.add_argument('--lstm_layers', type=int, default=2, help="number of layers in lstm", required=False)

        args = parser.parse_args()

        checkpoint_data = torch.load(args.model_weight, map_location='cpu', weights_only=False)
        training_config = {}
        if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
            model_state_dict = checkpoint_data['model_state_dict']
            training_config = checkpoint_data.get('training_config', {}) or {}
        else:
            model_state_dict = checkpoint_data

        if 'vocab_size' in training_config:
            args.vocab_size = training_config['vocab_size']
        if 'block_size' in training_config:
            args.block_size = training_config['block_size']
        if 'n_layer' in training_config:
            args.n_layer = training_config['n_layer']
        if 'n_head' in training_config:
            args.n_head = training_config['n_head']
        if 'n_embd' in training_config:
            args.n_embd = training_config['n_embd']
        if 'lstm' in training_config:
            args.lstm = training_config['lstm']
        if 'lstm_layers' in training_config:
            args.lstm_layers = training_config['lstm_layers']
        if 'scaffold' in training_config:
            args.scaffold = training_config['scaffold']
        if training_config.get('props'):
            args.props = training_config.get('props')
        if 'data_name' in training_config and training_config['data_name']:
            args.data_name = training_config['data_name']

        num_props_from_config = training_config.get('num_props') if training_config else None
        stored_atom_list = training_config.get('atom_list', []) if training_config else []
        atom_candidates = args.atom_list if args.atom_list is not None else stored_atom_list
        atom_symbols = dedupe_preserve_order([normalize_symbol(a) for a in atom_candidates or [] if normalize_symbol(a)])
        atom_cond_enabled = training_config.get('atom_cond', bool(atom_symbols)) if training_config else bool(atom_symbols)
        if atom_cond_enabled and not atom_symbols:
            atom_symbols = dedupe_preserve_order([normalize_symbol(a) for a in stored_atom_list if normalize_symbol(a)])

        atom_condition_values = parse_atom_condition_args(
            atom_symbols,
            args.atom_condition,
            on_value=args.atom_on_value,
            off_value=args.atom_off_value
        ) if atom_cond_enabled else None

        context = "C"

        data = pd.read_csv(f'datasets/{args.data_name}.csv')
        data = data.dropna(axis=0).reset_index(drop=True)
        data.columns = data.columns.str.lower()

        if 'moses' in args.data_name:
            smiles = data[data['split'] != 'test_scaffolds']['smiles']
            scaf = data[data['split'] != 'test_scaffolds']['scaffold_smiles']
        else:
            smiles = data[data['source'] != 'test']['smiles']
            scaf = data[data['source'] != 'test']['scaffold_smiles']

        pattern = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)

        if training_config.get('scaffold_maxlen') is not None:
            scaffold_max_len = training_config['scaffold_maxlen']
        elif 'moses' in args.data_name:
            scaffold_max_len = 48
        elif 'guacamol' in args.data_name:
            scaffold_max_len = 100
        else:
            scaffold_max_len = 0

        if not args.scaffold:
            scaffold_max_len = 0

        stoi = load_stoi_with_fallback(args.data_name, args.vocab_size)
        itos = {i: ch for ch, i in stoi.items()}

        print(itos)
        print(len(itos))
        if atom_symbols:
            print(f"Atom conditioning symbols: {atom_symbols}")
            if atom_condition_values is not None:
                print(f"Atom condition vector: {atom_condition_values}")
        else:
            print("Atom conditioning disabled")

        num_props = num_props_from_config if num_props_from_config is not None else len(args.props)
        mconf = GPTConfig(
            args.vocab_size,
            args.block_size,
            num_props=num_props,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            scaffold=args.scaffold,
            scaffold_maxlen=scaffold_max_len,
            lstm=args.lstm,
            lstm_layers=args.lstm_layers,
            atom_cond=atom_cond_enabled and bool(atom_symbols),
            atom_vocab_size=len(atom_symbols)
        )
        model = GPT(mconf)

        model.load_state_dict(model_state_dict)
        model.to('cuda')
        print('Model loaded')

        gen_iter = math.ceil(args.gen_size / args.batch_size)

        atom_condition_tensor = None
        if atom_condition_values is not None:
            atom_condition_tensor = torch.tensor(atom_condition_values, dtype=torch.float)

        if 'guacamol' in args.data_name:
            prop2value = {
                'qed': [0.3, 0.5, 0.7],
                'sas': [2.0, 3.0, 4.0],
                'logp': [2.0, 4.0, 6.0],
                'tpsa': [40.0, 80.0, 120.0],
                'tpsa_logp': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0], [40.0, 6.0], [80.0, 6.0], [120.0, 6.0]],
                'sas_logp': [[2.0, 2.0], [2.0, 4.0], [2.0, 6.0], [3.0, 2.0], [3.0, 4.0], [3.0, 6.0], [4.0, 2.0], [4.0, 4.0], [4.0, 6.0]],
                'tpsa_sas': [[40.0, 2.0], [80.0, 2.0], [120.0, 2.0], [40.0, 3.0], [80.0, 3.0], [120.0, 3.0], [40.0, 4.0], [80.0, 4.0], [120.0, 4.0]],
                'tpsa_logp_sas': [[40.0, 2.0, 2.0], [40.0, 2.0, 4.0], [40.0, 6.0, 4.0], [40.0, 6.0, 2.0], [80.0, 6.0, 4.0], [80.0, 2.0, 4.0], [80.0, 2.0, 2.0], [80.0, 6.0, 2.0]]
            }
        else:
            prop2value = {
                'qed': [0.6, 0.725, 0.85],
                'sas': [2.0, 2.75, 3.5],
                'logp': [1.0, 2.0, 3.0],
                'tpsa': [30.0, 60.0, 90.0],
                'tpsa_logp': [[40.0, 2.0], [80.0, 2.0], [40.0, 4.0], [80.0, 4.0]],
                'sas_logp': [[2.0, 1.0], [2.0, 3.0], [3.5, 1.0], [3.5, 3.0]],
                'tpsa_sas': [[40.0, 2.0], [80.0, 2.0], [40.0, 3.5], [80.0, 3.5]],
                'tpsa_logp_sas': [[40.0, 1.0, 2.0], [40.0, 1.0, 3.5], [40.0, 3.0, 2.0], [40.0, 3.0, 3.5], [80.0, 1.0, 2.0], [80.0, 1.0, 3.5], [80.0, 3.0, 2.0], [80.0, 3.0, 3.5]]
            }

        prop_condition = None
        if args.props:
            key = '_'.join(args.props)
            prop_condition = prop2value.get(key)

        scaf_condition = None
        if args.scaffold:
            base_scaffolds = [
                'O=C(Cc1ccccc1)NCc1ccccc1',
                'c1cnc2[nH]ccc2c1',
                'c1ccc(-c2ccnnc2)cc1',
                'c1ccc(-n2cnc3ccccc32)cc1',
                'O=C(c1cc[nH]c1)N1CCN(c2ccccc2)CC1'
            ]
            scaf_condition = [sc + '<' * (scaffold_max_len - len(regex.findall(sc))) for sc in base_scaffolds]

        all_dfs = []
        unknown_token_ids = set()
        count = 0

        if prop_condition is None and scaf_condition is None:
            molecules = []
            count += 1
            for _ in tqdm(range(gen_iter)):
                    x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
                    p = None
                    sca = None
                    atom_batch = None
                    if atom_condition_tensor is not None:
                        atom_batch = atom_condition_tensor.unsqueeze(0).repeat(args.batch_size, 1).to('cuda')
                    y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p, scaffold=sca, atom_cond=atom_batch)
                    for gen_mol in y:
                            token_ids = [int(i) for i in gen_mol]
                            missing = [idx for idx in token_ids if idx not in itos]
                            if missing:
                                unknown_token_ids.update(missing)
                                continue
                            completion = ''.join([itos[idx] for idx in token_ids]).replace('<', '')
                            mol = get_mol(completion)
                            if mol:
                                molecules.append(mol)

            if molecules:
                mol_dict = [{'molecule': mol, 'smiles': Chem.MolToSmiles(mol)} for mol in molecules]
                results = pd.DataFrame(mol_dict)
                canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                unique_smiles = list(set(canon_smiles))
                if 'moses' in args.data_name:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['split'] == 'train']['smiles']))
                else:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))

                print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

                results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
                results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
                results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
                results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
                results['unique'] = np.round(len(unique_smiles) / len(results), 3)
                results['novelty'] = np.round(novel_ratio / 100, 3)
                all_dfs.append(results)
            else:
                print("Warning: no valid molecules generated in the unconditional configuration.")

        elif prop_condition is not None and scaf_condition is None:
            count = 0
            for c in prop_condition:
                molecules = []
                count += 1
                for _ in tqdm(range(gen_iter)):
                        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
                        if len(args.props) == 1:
                                p = torch.tensor([[c]], dtype=torch.float, device='cuda').repeat(args.batch_size, 1)
                        else:
                                p = torch.tensor([c], dtype=torch.float, device='cuda').repeat(args.batch_size, 1).unsqueeze(1)
                        sca = None
                        atom_batch = None
                        if atom_condition_tensor is not None:
                                atom_batch = atom_condition_tensor.unsqueeze(0).repeat(args.batch_size, 1).to('cuda')
                        y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p, scaffold=sca, atom_cond=atom_batch)
                        for gen_mol in y:
                                token_ids = [int(i) for i in gen_mol]
                                missing = [idx for idx in token_ids if idx not in itos]
                                if missing:
                                        unknown_token_ids.update(missing)
                                        continue
                                completion = ''.join([itos[idx] for idx in token_ids]).replace('<', '')
                                mol = get_mol(completion)
                                if mol:
                                        molecules.append(mol)

                if not molecules:
                        print(f"Warning: no valid molecules generated for property condition {c}.")
                        continue

                mol_dict = [{'molecule': mol, 'smiles': Chem.MolToSmiles(mol)} for mol in molecules]
                results = pd.DataFrame(mol_dict)
                canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                unique_smiles = list(set(canon_smiles))
                if 'moses' in args.data_name:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['split'] == 'train']['smiles']))
                else:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))

                print(f'Condition: {c}')
                print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

                if len(args.props) == 1:
                        results['condition'] = c
                elif len(args.props) == 2:
                        results['condition'] = str((c[0], c[1]))
                else:
                        results['condition'] = str(tuple(c))

                results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
                results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
                results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
                results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
                results['unique'] = np.round(len(unique_smiles) / len(results), 3)
                results['novelty'] = np.round(novel_ratio / 100, 3)
                all_dfs.append(results)

        elif prop_condition is None and scaf_condition is not None:
            count = 0
            for j in scaf_condition:
                molecules = []
                count += 1
                for _ in tqdm(range(gen_iter)):
                    x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
                    p = None
                    sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
                    atom_batch = None
                    if atom_condition_tensor is not None:
                        atom_batch = atom_condition_tensor.unsqueeze(0).repeat(args.batch_size, 1).to('cuda')
                    y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p, scaffold=sca, atom_cond=atom_batch)
                    for gen_mol in y:
                        token_ids = [int(i) for i in gen_mol]
                        missing = [idx for idx in token_ids if idx not in itos]
                        if missing:
                            unknown_token_ids.update(missing)
                            continue
                        completion = ''.join([itos[idx] for idx in token_ids]).replace('<', '')
                        mol = get_mol(completion)
                        if mol:
                            molecules.append(mol)

                if not molecules:
                    print(f"Warning: no valid molecules generated for scaffold condition {j}.")
                    continue

                mol_dict = [{'molecule': mol, 'smiles': Chem.MolToSmiles(mol)} for mol in molecules]
                results = pd.DataFrame(mol_dict)
                canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                unique_smiles = list(set(canon_smiles))
                if 'moses' in args.data_name:
                    novel_ratio = check_novelty(unique_smiles, set(data[data['split'] == 'train']['smiles']))
                else:
                    novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))

                print(f'Scaffold: {j}')
                print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
                print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
                print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

                results['scaffold_cond'] = j
                results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
                results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
                results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
                results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
                results['unique'] = np.round(len(unique_smiles) / len(results), 3)
                results['novelty'] = np.round(novel_ratio / 100, 3)
                all_dfs.append(results)

        elif prop_condition is not None and scaf_condition is not None:
            count = 0
            for j in scaf_condition:
                for c in prop_condition:
                    molecules = []
                    count += 1
                    for _ in tqdm(range(gen_iter)):
                        x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
                        if len(args.props) == 1:
                            p = torch.tensor([[c]], dtype=torch.float, device='cuda').repeat(args.batch_size, 1)
                        else:
                            p = torch.tensor([c], dtype=torch.float, device='cuda').repeat(args.batch_size, 1).unsqueeze(1)
                        sca = torch.tensor([stoi[s] for s in regex.findall(j)], dtype=torch.long)[None, ...].repeat(args.batch_size, 1).to('cuda')
                        atom_batch = None
                        if atom_condition_tensor is not None:
                            atom_batch = atom_condition_tensor.unsqueeze(0).repeat(args.batch_size, 1).to('cuda')
                        y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=None, prop=p, scaffold=sca, atom_cond=atom_batch)
                        for gen_mol in y:
                            token_ids = [int(i) for i in gen_mol]
                            missing = [idx for idx in token_ids if idx not in itos]
                            if missing:
                                unknown_token_ids.update(missing)
                                continue
                            completion = ''.join([itos[idx] for idx in token_ids]).replace('<', '')
                            mol = get_mol(completion)
                            if mol:
                                molecules.append(mol)

                    if not molecules:
                        print(f"Warning: no valid molecules generated for property {c} with scaffold {j}.")
                        continue

                    mol_dict = [{'molecule': mol, 'smiles': Chem.MolToSmiles(mol)} for mol in molecules]
                    results = pd.DataFrame(mol_dict)
                    canon_smiles = [canonic_smiles(s) for s in results['smiles']]
                    unique_smiles = list(set(canon_smiles))
                    if 'moses' in args.data_name:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['split'] == 'train']['smiles']))
                    else:
                        novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))

                    print(f'Condition: {c}')
                    print(f'Scaffold: {j}')
                    print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter), 3))
                    print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
                    print('Novelty ratio: ', np.round(novel_ratio / 100, 3))

                    if len(args.props) == 1:
                        results['condition'] = c
                    elif len(args.props) == 2:
                        results['condition'] = str((c[0], c[1]))
                    else:
                        results['condition'] = str(tuple(c))

                    results['scaffold_cond'] = j
                    results['qed'] = results['molecule'].apply(lambda x: QED.qed(x))
                    results['sas'] = results['molecule'].apply(lambda x: sascorer.calculateScore(x))
                    results['logp'] = results['molecule'].apply(lambda x: Crippen.MolLogP(x))
                    results['tpsa'] = results['molecule'].apply(lambda x: CalcTPSA(x))
                    results['validity'] = np.round(len(results) / (args.batch_size * gen_iter), 3)
                    results['unique'] = np.round(len(unique_smiles) / len(results), 3)
                    results['novelty'] = np.round(novel_ratio / 100, 3)
                    all_dfs.append(results)

        if unknown_token_ids:
            print(f"Warning: skipped tokens with unknown indices: {sorted(unknown_token_ids)}. Check vocabulary alignment.")

        if not all_dfs:
            print("No valid molecules were generated; skipping CSV export.")
            sys.exit(0)

        results = pd.concat(all_dfs, ignore_index=True)
        if '/' not in args.csv_name:
            os.makedirs('generated_csvs', exist_ok=True)
            args.csv_name = os.path.join('generated_csvs', args.csv_name)
        results.to_csv(args.csv_name + '.csv', index=False)

        canon_smiles = [canonic_smiles(s) for s in results['smiles']]
        unique_smiles = list(set(canon_smiles))
        if 'moses' in args.data_name:
                novel_ratio = check_novelty(unique_smiles, set(data[data['split'] == 'train']['smiles']))
        else:
                novel_ratio = check_novelty(unique_smiles, set(data[data['source'] == 'train']['smiles']))

        print('Valid ratio: ', np.round(len(results) / (args.batch_size * gen_iter * count), 3))
        print('Unique ratio: ', np.round(len(unique_smiles) / len(results), 3))
        print('Novelty ratio: ', np.round(novel_ratio / 100, 3))
