#!/usr/bin/env python3
"""
统一模型配置检测脚本
精确检测PyTorch模型权重文件(.pt)的训练配置并提供生成命令建议
"""

import torch
import argparse
import os
import json
from collections import OrderedDict

class ModelConfigDetector:
    """模型配置检测器"""
    
    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.checkpoint = None
        self.config = {}
        
    def load_checkpoint(self):
        """加载权重文件"""
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"权重文件不存在: {self.weight_path}")
        
        try:
            checkpoint_data = torch.load(self.weight_path, map_location='cpu', weights_only=False)
            print(f"✅ 成功加载权重文件: {self.weight_path}")
            print(f"📁 文件大小: {os.path.getsize(self.weight_path) / (1024*1024):.2f} MB")
            
            # 🔧 检查是否为新格式（包含训练配置）
            if isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data:
                print("🆕 检测到新格式权重文件（包含训练配置）")
                self.checkpoint = checkpoint_data['model_state_dict']
                self.training_config = checkpoint_data.get('training_config', {})
                print(f"📋 训练配置信息: {self.training_config}")
            else:
                print("🔄 检测到旧格式权重文件（仅包含模型权重）")
                self.checkpoint = checkpoint_data
                self.training_config = {}
            
            print(f"🔢 参数总数: {len(self.checkpoint)}")
        except Exception as e:
            raise RuntimeError(f"无法加载权重文件: {str(e)}")
    
    def detect_training_config_from_saved(self):
        """从保存的训练配置中获取信息"""
        if hasattr(self, 'training_config') and self.training_config:
            # 直接从保存的配置中读取
            for key, value in self.training_config.items():
                self.config[key] = value
            
            # 设置便于检测的标志
            self.config['has_props'] = self.config.get('num_props', 0) > 0
            self.config['uses_scaffold'] = self.config.get('scaffold', False)
            self.config['has_lstm'] = self.config.get('lstm', False)
            
            return True
        return False
    
    def detect_basic_config(self):
        """检测基本配置参数"""
        # 词汇表大小和嵌入维度
        if 'tok_emb.weight' in self.checkpoint:
            self.config['vocab_size'] = self.checkpoint['tok_emb.weight'].shape[0]
            self.config['n_embd'] = self.checkpoint['tok_emb.weight'].shape[1]
        
        # 序列长度
        if 'pos_emb' in self.checkpoint:
            self.config['block_size'] = self.checkpoint['pos_emb'].shape[1]
        
        # 层数
        layer_keys = [k for k in self.checkpoint.keys() if k.startswith('blocks.')]
        if layer_keys:
            layer_nums = set()
            for key in layer_keys:
                parts = key.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_nums.add(int(parts[1]))
            self.config['n_layer'] = max(layer_nums) + 1 if layer_nums else 0
        
        # 注意力头数推断
        if 'n_embd' in self.config:
            # 常见的头数配置
            possible_heads = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]
            for heads in possible_heads:
                if self.config['n_embd'] % heads == 0:
                    self.config['n_head'] = heads
                    break
            if 'n_head' not in self.config:
                self.config['n_head'] = 8  # 默认值
        
        # 注意力掩码大小
        mask_keys = [k for k in self.checkpoint.keys() if 'attn.mask' in k]
        if mask_keys:
            self.config['mask_size'] = self.checkpoint[mask_keys[0]].shape[-1]
    
    def detect_conditional_features(self):
        """检测条件生成特征"""
        # 检测属性层
        prop_keys = [k for k in self.checkpoint.keys() if 'prop_nn' in k]
        self.config['has_props'] = len(prop_keys) > 0
        
        if self.config['has_props']:
            if 'prop_nn.weight' in self.checkpoint:
                self.config['num_props'] = self.checkpoint['prop_nn.weight'].shape[1]
            else:
                self.config['num_props'] = 1  # 默认
        else:
            self.config['num_props'] = 0
        
        # 检测脚手架层
        scaffold_keys = [k for k in self.checkpoint.keys() if 'scaffold' in k.lower()]
        self.config['has_scaffold_layers'] = len(scaffold_keys) > 0
        
        # 检测LSTM层
        lstm_keys = [k for k in self.checkpoint.keys() if 'lstm' in k.lower()]
        self.config['has_lstm'] = len(lstm_keys) > 0
        
        if self.config['has_lstm']:
            # 推断LSTM层数
            lstm_layer_keys = [k for k in lstm_keys if 'weight_ih_l' in k or 'weight_hh_l' in k]
            if lstm_layer_keys:
                layer_nums = set()
                for key in lstm_layer_keys:
                    if 'weight_ih_l' in key or 'weight_hh_l' in key:
                        layer_num = key.split('_')[-1]
                        if layer_num.isdigit():
                            layer_nums.add(int(layer_num))
                self.config['lstm_layers'] = max(layer_nums) + 1 if layer_nums else 2
            else:
                self.config['lstm_layers'] = 2  # 默认
        else:
            self.config['lstm_layers'] = 0
    
    def infer_training_config(self):
        """基于掩码大小推断训练时的配置"""
        if 'mask_size' not in self.config or 'block_size' not in self.config:
            return
        
        mask_size = self.config['mask_size']
        block_size = self.config['block_size']
        extra_size = mask_size - block_size
        
        # 根据掩码公式: mask_size = block_size + int(bool(num_props)) + scaffold_maxlen
        prop_contribution = 1 if self.config['has_props'] else 0
        inferred_scaffold_maxlen = extra_size - prop_contribution
        
        # 验证推断的合理性
        if inferred_scaffold_maxlen < 0:
            inferred_scaffold_maxlen = 0
        elif inferred_scaffold_maxlen > 200:  # 不合理的大值
            if self.config['vocab_size'] == 94:  # GuacaMol
                inferred_scaffold_maxlen = 100  # 根据之前的分析
            else:
                inferred_scaffold_maxlen = 48  # Moses默认
        
        self.config['scaffold_maxlen'] = inferred_scaffold_maxlen
        self.config['uses_scaffold'] = inferred_scaffold_maxlen > 0
        
        # 验证配置一致性
        expected_mask_size = block_size + prop_contribution + inferred_scaffold_maxlen
        self.config['config_consistent'] = (expected_mask_size == mask_size)
    
    def detect_dataset(self):
        """推断数据集类型"""
        if 'vocab_size' in self.config and 'block_size' in self.config:
            vocab_size = self.config['vocab_size']
            block_size = self.config['block_size']
            
            # 根据词汇表大小和序列长度推断数据集
            if vocab_size == 94 and block_size == 100:
                self.config['dataset'] = 'guacamol2'
            elif vocab_size == 94 and block_size == 54:
                self.config['dataset'] = 'moses2'  # Moses也可能使用94词汇表
            elif vocab_size == 26:
                self.config['dataset'] = 'moses2'
            else:
                # 根据序列长度推断
                if block_size >= 90:
                    self.config['dataset'] = 'guacamol2'
                else:
                    self.config['dataset'] = 'moses2'
    
    def analyze(self):
        """完整分析模型配置"""
        print("🔍 开始分析模型配置...")
        print("=" * 60)
        
        self.load_checkpoint()
        
        # 🔧 优先从保存的训练配置中获取信息
        if self.detect_training_config_from_saved():
            print("✅ 成功从保存的训练配置中获取完整信息")
        else:
            print("⚠️  使用启发式方法推断模型配置（可能不够准确）")
            self.detect_basic_config()
            self.detect_conditional_features()
            self.infer_training_config()
            self.detect_dataset()
        
        return self.config
    
    def print_config(self):
        """打印配置信息"""
        print("\n📋 检测到的模型配置:")
        print("-" * 40)
        
        # 基本配置
        basic_params = ['vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd', 'mask_size']
        for param in basic_params:
            if param in self.config:
                print(f"  {param}: {self.config[param]}")
        
        print(f"  dataset: {self.config.get('dataset', 'unknown')}")
        
        # 条件生成配置
        print("\n🎯 条件生成配置:")
        print(f"  has_props: {self.config.get('has_props', False)}")
        print(f"  num_props: {self.config.get('num_props', 0)}")
        print(f"  uses_scaffold: {self.config.get('uses_scaffold', False)}")
        print(f"  scaffold_maxlen: {self.config.get('scaffold_maxlen', 0)}")
        print(f"  has_lstm: {self.config.get('has_lstm', False)}")
        if self.config.get('has_lstm'):
            print(f"  lstm_layers: {self.config.get('lstm_layers', 2)}")
        
        # 配置一致性
        print(f"\n✅ 配置一致性检查:")
        consistent = self.config.get('config_consistent', False)
        print(f"  掩码大小匹配: {'✅' if consistent else '❌'}")
        
        if not consistent:
            print("  ⚠️  检测到配置不一致，可能是训练脚本的bug导致")
        
        # 掩码分析
        if 'mask_size' in self.config and 'block_size' in self.config:
            mask_size = self.config['mask_size']
            block_size = self.config['block_size']
            prop_contrib = 1 if self.config.get('has_props') else 0
            scaffold_len = self.config.get('scaffold_maxlen', 0)
            
            print(f"\n🔍 掩码大小分析:")
            print(f"  掩码大小: {mask_size}")
            print(f"  计算公式: {block_size} (序列) + {prop_contrib} (属性) + {scaffold_len} (脚手架) = {block_size + prop_contrib + scaffold_len}")
    
    def generate_commands(self):
        """生成使用建议"""
        print(f"\n🚀 推荐的生成命令:")
        print("=" * 50)
        
        # 推荐使用统一生成脚本（自动检测配置）
        print("🌟 推荐使用统一生成脚本（自动检测模型配置）:")
        print("-" * 45)
        
        unified_cmd = [
            "python generate/generate_unified.py",
            f"--model_weight {self.weight_path}",
            "--csv_name your_output_name",
            "--gen_size 1000",
            "--batch_size 32"
        ]
        
        # 根据检测到的配置添加条件参数
        if self.config.get('has_props'):
            # 🔧 如果有准确的属性信息，直接使用
            if 'props' in self.config and self.config['props']:
                props_str = ' '.join(self.config['props'])
                unified_cmd.append(f"--props {props_str}")
                print(f"  ✅ 检测到训练时使用的属性: {props_str}")
            else:
                print("  ⚠️  检测到属性条件模型，但无法从权重文件中确定具体属性类型")
                print("  📝 请根据训练时使用的属性手动指定，例如：")
                print("      --props qed    (药物相似性)")
                print("      --props sas    (合成可达性)")  
                print("      --props logp   (脂水分配系数)")
                print("      --props tpsa   (极性表面积)")
                unified_cmd.append("--props YOUR_PROPERTY_TYPE")
        
        if self.config.get('uses_scaffold'):
            unified_cmd.append("--scaffold")
        
        if self.config.get('has_lstm'):
            unified_cmd.append("--lstm")
        
        # 添加数据集参数
        if 'dataset' in self.config:
            unified_cmd.append(f"--data_name {self.config['dataset']}")
        
        print(" \\\n  ".join(unified_cmd))
        print("\n  ✨ 统一脚本会自动检测模型配置，无需手动指定架构参数")
        
        # 如果用户需要使用原始生成脚本（手动配置）
        if os.path.exists('generate/generate.py'):
            print(f"\n💡 如果需要使用原始生成脚本（手动配置）:")
            print("-" * 40)
            
            manual_cmd = ["python generate/generate.py"]
            manual_cmd.append(f"--model_weight {self.weight_path}")
            manual_cmd.append("--csv_name your_output_name")
            
            # 数据集
            if 'dataset' in self.config:
                manual_cmd.append(f"--data_name {self.config['dataset']}")
            
            # 基本参数（需要手动指定）
            basic_params = ['vocab_size', 'block_size', 'n_layer', 'n_head', 'n_embd']
            for param in basic_params:
                if param in self.config:
                    manual_cmd.append(f"--{param} {self.config[param]}")
            
            # 条件参数
            if self.config.get('has_props'):
                if 'props' in self.config and self.config['props']:
                    props_str = ' '.join(self.config['props'])
                    manual_cmd.append(f"--props {props_str}")
                else:
                    manual_cmd.append("--props YOUR_PROPERTY_TYPE")
            
            if self.config.get('uses_scaffold'):
                manual_cmd.append("--scaffold")
            
            if self.config.get('has_lstm'):
                manual_cmd.append("--lstm")
                manual_cmd.append(f"--lstm_layers {self.config.get('lstm_layers', 2)}")
            
            # 生成参数
            manual_cmd.append("--gen_size 1000")
            manual_cmd.append("--batch_size 32")
            
            print(" \\\n  ".join(manual_cmd))
            print("\n  ⚠️  原始脚本需要手动指定所有架构参数")
            
        # 添加属性类型说明
        if self.config.get('has_props'):
            print(f"\n📋 常用属性类型说明:")
            print("-" * 30)
            print("  qed  : 药物相似性 (Drug-likeness)")
            print("  sas  : 合成可达性 (Synthetic Accessibility)")
            print("  logp : 脂水分配系数 (Lipophilicity)")
            print("  tpsa : 极性表面积 (Topological Polar Surface Area)")
            print("\n  💡 请根据您的训练命令中使用的 --props 参数来选择")

def main():
    parser = argparse.ArgumentParser(description='统一模型配置检测工具')
    parser.add_argument('weight_path', type=str, help='权重文件路径(.pt)')
    parser.add_argument('--json', action='store_true', help='输出JSON格式的配置')
    parser.add_argument('--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    try:
        detector = ModelConfigDetector(args.weight_path)
        config = detector.analyze()
        
        detector.print_config()
        detector.generate_commands()
        
        if args.json:
            print(f"\n📄 JSON配置输出:")
            print(json.dumps(config, indent=2, ensure_ascii=False))
        
        if args.verbose:
            print(f"\n🔍 详细权重信息:")
            print("-" * 40)
            for key, value in detector.checkpoint.items():
                shape_str = str(value.shape) if hasattr(value, 'shape') else str(type(value))
                print(f"  {key:<40} {shape_str}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 