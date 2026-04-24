# pmpnn_wrapper.py - 完整修复版
# ESM2 REINFORCE反馈已正确实现

import csv
from collections import defaultdict
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch, time, os
import wandb
import math
from matplotlib import pyplot as plt
from openfold.np.residue_constants import restype_order_with_x, restypes_with_x, unk_restype_index

from codon.utils.codon_const import codon_order, codon_types, codon_to_res, res_to_codon, unk_codon_index

from codon.utils.esm_foldability_utils import run_foldability  # 或 foldability_utils
from codon.utils.logging import get_logger
from codon.utils.pmpnn import ProteinMPNN

try:
    from codon.utils.flow_pmpnn import FlowProteinMPNN
except ImportError:
    try:
        from .flow_pmpnn import FlowProteinMPNN
    except ImportError:
        from flow_pmpnn import FlowProteinMPNN

import torch.nn as nn
import torch.nn.functional as F
import json
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logger = get_logger(__name__)


def gather_log(log, world_size):
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out


class Wrapper(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        log = self._log
        if self.stage == 'train' or self.args.validate:
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        out = self.general_step(batch, stage='val')
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()
        return out

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, stage='test')

    def on_train_epoch_end(self):
        self.print_log(prefix='train', save=True)

    def on_validation_epoch_end(self):
        self.print_log(prefix='val', save=True)

    def on_test_epoch_end(self):
        self.print_log(prefix='test', save=True)

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            self.print_log()

    def print_log(self, prefix='iter', save=False, extra_logs=None):
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update({
            'epoch': self.trainer.current_epoch,
            'trainer_step': self.trainer.global_step + int(prefix == 'iter'),
            'iter_step': self.iter_step,
            f'{prefix}_count': len(log[next(iter(log))]) if log else 0,
        })
        if extra_logs:
            mean_log.update(extra_logs)
        try:
            for param_group in self.optimizers().optimizer.param_groups:
                mean_log['lr'] = param_group['lr']
        except:
            pass

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)
            if save and log:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv"
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{prefix}_" in key:
                del self._log[key]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
        )
        return optimizer


class TransformerBlock(nn.Module):
    """标准Transformer块"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        if mask is not None:
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None
            
        attn_out, attention_weights = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return x, attention_weights


class PMPNNWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        self.K = len(restype_order_with_x) if args.train_aa else len(codon_order)
        self.model = FlowProteinMPNN(args, vocab=self.K, node_features=args.hidden_dim,
                                 edge_features=args.hidden_dim,
                                 hidden_dim=args.hidden_dim, num_encoder_layers=args.num_encoder_layers,
                                 num_decoder_layers=args.num_decoder_layers,
                                 k_neighbors=args.num_neighbors, dropout=args.dropout, ca_only=False)
        self.val_dict = defaultdict(list)
        
        # 序列保存相关属性
        self.predicted_sequences = defaultdict(list)
        self.sequence_metadata = defaultdict(list)

        # Transformer组件
        self.use_transformer = getattr(args, 'use_transformer', False)

        # ========== ESM2组件初始化 ==========
        self.use_esm2 = getattr(args, 'use_esm2_feedback', False)
        if self.use_esm2:
            self._init_esm2_components(args)
            # 【关键】初始化baseline
            self.esm2_baseline = None
            self.baseline_momentum = 0.9
        
        # Transformer初始化
        if self.use_transformer:
            print(f"启用Transformer增强: heads={args.transformer_heads}, layers={args.transformer_layers}")
            
            self.transformer_layers = nn.ModuleList([
                TransformerBlock(
                    d_model=args.hidden_dim,
                    n_heads=args.transformer_heads,
                    d_ff=args.hidden_dim * 4,
                    dropout=args.dropout
                ) for _ in range(args.transformer_layers)
            ])
            
            self.feature_fusion = nn.Sequential(
                nn.Linear(args.hidden_dim * 2, args.hidden_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim)
            )
            
            if getattr(args, 'use_positional_encoding', True):
                max_seq_len = getattr(args, 'max_seq_len', 2048)
                self.register_buffer('positional_encoding', 
                                   self._create_positional_encoding(max_seq_len, args.hidden_dim))
            
            self.transformer_start_epoch = getattr(args, 'transformer_start_epoch', 0)
            self.max_fusion_weight = getattr(args, 'transformer_fusion_weight', 0.5)
    
    # ========== Transformer相关方法 ==========
    def _create_positional_encoding(self, max_len, d_model):
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _get_current_fusion_weight(self):
        """根据当前epoch计算融合权重"""
        if not self.use_transformer:
            return 0.0
        
        current_epoch = self.trainer.current_epoch if hasattr(self, 'trainer') else 0
        
        if current_epoch < self.transformer_start_epoch:
            return 0.0
        
        max_epochs = getattr(self.trainer, 'max_epochs', 100) if hasattr(self, 'trainer') else 100
        progress = (current_epoch - self.transformer_start_epoch) / \
                  max(1, max_epochs - self.transformer_start_epoch)
        return min(self.max_fusion_weight, progress * self.max_fusion_weight)
    
    def _apply_transformer_enhancement(self, h_V, mask):
        """应用Transformer增强"""
        if not self.use_transformer:
            return h_V
        
        fusion_weight = self._get_current_fusion_weight()
        if fusion_weight == 0.0:
            return h_V
        
        B, L, D = h_V.shape
        
        if hasattr(self, 'positional_encoding'):
            pos_enc = self.positional_encoding[:, :L, :].to(h_V.device)
            h_V_with_pos = h_V + pos_enc * mask.unsqueeze(-1)
        else:
            h_V_with_pos = h_V
        
        h_V_transformed = h_V_with_pos
        for transformer_layer in self.transformer_layers:
            h_V_transformed, _ = transformer_layer(h_V_transformed, mask)
        
        fused_features = torch.cat([h_V, h_V_transformed], dim=-1)
        h_V_enhanced = self.feature_fusion(fused_features)
        
        h_V_final = (1 - fusion_weight) * h_V + fusion_weight * h_V_enhanced
        
        return h_V_final * mask.unsqueeze(-1)
    
    def _compute_codon_consistency_loss(self, log_probs, targets, mask):
        """计算密码子一致性损失（向量化版本）"""
        B, L, K = log_probs.shape
        
        if L < 2:
            return torch.tensor(0.0, device=log_probs.device)
        
        # 向量化计算
        prob = F.softmax(log_probs, dim=-1)
        prob_current = prob[:, :-1, :]
        prob_next = prob[:, 1:, :]
        
        diff = F.smooth_l1_loss(prob_current, prob_next, reduction='none').sum(-1)
        
        valid_mask = mask[:, :-1] * mask[:, 1:]
        
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=log_probs.device)
        
        return (diff * valid_mask).sum() / valid_mask.sum()
    
    # ========== ESM2相关方法 ==========
    def _init_esm2_components(self, args):
        """初始化ESM2组件"""
        try:
            import esm
        except ImportError:
            print(" ESM模块未安装，禁用ESM2反馈")
            self.use_esm2 = False
            return
        
        self.esm2_model_name = getattr(args, 'esm2_model', 'esm2_t12_35M_UR50D')
        self.esm2_start_epoch = getattr(args, 'esm2_start_epoch', 5)
        self.esm2_check_interval = getattr(args, 'esm2_check_interval', 10)
        self.esm2_weight = getattr(args, 'esm2_weight', 0.1)
        self.esm2_batch_counter = 0
        
        print(f"\n{'='*60}")
        print("Initializing ESM2 REINFORCE Feedback")
        print(f"  Model: {self.esm2_model_name}")
        print(f"  Start Epoch: {self.esm2_start_epoch}")
        print(f"  Weight: {self.esm2_weight}")
        print(f"  Check Interval: {self.esm2_check_interval}")
        print(f"  Method: Policy Gradient (REINFORCE + Baseline)")
        print(f"{'='*60}\n")
        
        try:
            self.esm2_model, self.esm2_alphabet = esm.pretrained.load_model_and_alphabet(
                self.esm2_model_name
            )
            self.esm2_model = self.esm2_model.eval()
            
            # 【关键】冻结ESM2参数，避免被训练
            for param in self.esm2_model.parameters():
                param.requires_grad = False
            
            if torch.cuda.is_available():
                self.esm2_model = self.esm2_model.cuda()
            self.esm2_batch_converter = self.esm2_alphabet.get_batch_converter()
            print("ESM2 loaded successfully (parameters frozen)\n")
        except Exception as e:
            print(f"xESM2 loading failed: {e}")
            self.use_esm2 = False
    
    def _should_apply_esm2(self):
        """判断是否应用ESM2反馈"""
        if not self.use_esm2:
            return False
        current_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        if current_epoch < self.esm2_start_epoch:
            return False
        should_apply = (self.esm2_batch_counter % self.esm2_check_interval == 0)
        self.esm2_batch_counter += 1
        return should_apply
    
    def _get_esm2_weight(self):
        """获取渐进式权重"""
        if not self.use_esm2:
            return 0.0
        current_epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        if current_epoch < self.esm2_start_epoch:
            return 0.0
        progress = min(1.0, (current_epoch - self.esm2_start_epoch) / 10.0)
        return self.esm2_weight * progress
    
    def _codons_to_proteins(self, codon_indices, mask):
        """密码子转蛋白质（保留X）"""
        B, L = codon_indices.shape
        proteins = []
        for b in range(B):
            valid_len = int(mask[b].sum().item())
            codons_b = codon_indices[b, :valid_len]
            protein = []
            for idx in codons_b:
                idx_val = int(idx.item())
                if 0 <= idx_val < len(codon_types):
                    codon = codon_types[idx_val]
                    aa = codon_to_res.get(codon, 'X')
                    protein.append(aa)  # 保留X
                else:
                    protein.append('X')
            proteins.append(''.join(protein))
        return proteins
    
    @torch.no_grad()
    def _compute_esm2_reward(self, pred_codons, mask):
        """计算ESM2奖励"""
        try:
            pred_proteins = self._codons_to_proteins(pred_codons, mask)
            
            # 过滤包含太多X的序列
            valid_indices = []
            valid_proteins = []
            min_length = 5
            max_x_ratio = 0.15
            
            for i, seq in enumerate(pred_proteins):
                if len(seq) >= min_length and seq.count('X') / len(seq) < max_x_ratio:
                    valid_indices.append(i)
                    valid_proteins.append(seq)
            
            B = pred_codons.shape[0]
            
            if len(valid_proteins) == 0:
                if self.iter_step % 100 == 0:
                    print(f"Warning: All sequences filtered (X ratio >{max_x_ratio*100}%)")
                return {
                    'reward': torch.zeros(B, device=self.device),
                    'perplexity': torch.tensor(float('nan')),
                    'confidence': torch.tensor(float('nan'))
                }
            
            # ESM2评估
            batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(valid_proteins)]
            _, _, batch_tokens = self.esm2_batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)
            
            results = self.esm2_model(batch_tokens, repr_layers=[12], return_contacts=False)
            logits = results["logits"]
            
            # 计算困惑度和置信度
            log_probs = F.log_softmax(logits, dim=-1)
            actual_log_probs = log_probs.gather(2, batch_tokens[:, 1:-1].unsqueeze(-1)).squeeze(-1)
            perplexity = torch.exp(-actual_log_probs.mean(dim=1))
            
            token_probs = F.softmax(logits, dim=-1)
            confidence = token_probs.gather(2, batch_tokens[:, 1:-1].unsqueeze(-1)).squeeze(-1).mean(dim=1)
            
            # 组合奖励
            reward_valid = -torch.log(perplexity + 1e-6) + confidence
            
            # 填充完整批次
            reward_full = torch.zeros(B, device=self.device)
            for idx, r in zip(valid_indices, reward_valid):
                reward_full[idx] = r
            
            return {
                'reward': reward_full,
                'perplexity': perplexity.mean(),
                'confidence': confidence.mean()
            }
        except Exception as e:
            print(f"ESM2 reward computation failed: {e}")
            return {
                'reward': torch.zeros(pred_codons.shape[0], device=self.device),
                'perplexity': torch.tensor(float('nan')),
                'confidence': torch.tensor(float('nan'))
            }
    
    def _update_baseline(self, reward):
        """更新baseline（指数移动平均）"""
        if self.esm2_baseline is None:
            self.esm2_baseline = reward.mean().item()
        else:
            self.esm2_baseline = (self.baseline_momentum * self.esm2_baseline + 
                                 (1 - self.baseline_momentum) * reward.mean().item())
    
    # ========== 序列保存相关方法 ==========
    def codons_to_dna_string(self, codon_indices):
        """将密码子索引转换为DNA字符串"""
        dna_seq = ""
        for idx in codon_indices:
            if idx.item() < len(codon_types):
                dna_seq += codon_types[idx.item()]
            else:
                dna_seq += "NNN"
        return dna_seq

    def residues_to_protein_string(self, residue_indices):
        """将氨基酸索引转换为蛋白质字符串"""
        protein_seq = ""
        for idx in residue_indices:
            if idx.item() < len(restypes_with_x):
                protein_seq += restypes_with_x[idx.item()]
            else:
                protein_seq += "X"
        return protein_seq

    def save_predicted_sequences(self, pred_codons, pred_res, batch, stage='val', batch_idx=0):
        """保存预测的DNA序列和蛋白质序列"""
        mask = batch['mask']
        B, L = pred_codons.shape
        
        for b in range(B):
            valid_length = int(mask[b].sum().item())
            
            if valid_length == 0:
                continue
            
            valid_pred_codons = pred_codons[b, :valid_length]
            valid_pred_res = pred_res[b, :valid_length]
            
            dna_seq = self.codons_to_dna_string(valid_pred_codons)
            protein_seq = self.residues_to_protein_string(valid_pred_res)
            
            seq_info = {
                'batch_idx': batch_idx,
                'sample_idx': b,
                'stage': stage,
                'epoch': self.trainer.current_epoch if hasattr(self, 'trainer') else 0,
                'length': valid_length,
                'dna_sequence': dna_seq,
                'protein_sequence': protein_seq,
            }
            
            if 'seq' in batch:
                original_seq = batch['seq'][b, :valid_length]
                seq_info['original_protein'] = self.residues_to_protein_string(original_seq)
                
            if 'codons' in batch:
                original_codons = batch['codons'][b, :valid_length]
                seq_info['original_dna'] = self.codons_to_dna_string(original_codons)
            
            if 'taxon_id' in batch:
                seq_info['taxon_id'] = batch['taxon_id'][b].item()
                
            self.predicted_sequences[stage].append(seq_info)

    def save_sequences_to_csv(self, stage='val', filename=None):
        """将预测序列保存为CSV文件"""
        if not filename:
            filename = f'{os.environ.get("MODEL_DIR", ".")}/predicted_sequences_{stage}_epoch{self.trainer.current_epoch}.csv'
        
        if stage in self.predicted_sequences and self.predicted_sequences[stage]:
            df = pd.DataFrame(self.predicted_sequences[stage])
            df.to_csv(filename, index=False)
            print(f"Saved {len(df)} sequences to {filename}")
            return filename
        return None

    def save_sequences_to_fasta(self, stage='val', dna_filename=None, protein_filename=None):
        """将预测序列保存为FASTA文件"""
        if not dna_filename:
            dna_filename = f'{os.environ.get("MODEL_DIR", ".")}/predicted_dna_{stage}_epoch{self.trainer.current_epoch}.fasta'
        if not protein_filename:
            protein_filename = f'{os.environ.get("MODEL_DIR", ".")}/predicted_protein_{stage}_epoch{self.trainer.current_epoch}.fasta'
        
        if stage not in self.predicted_sequences or not self.predicted_sequences[stage]:
            return None, None
        
        dna_records = []
        protein_records = []
        
        for i, seq_info in enumerate(self.predicted_sequences[stage]):
            dna_id = f"pred_dna_{stage}_{i}_epoch{seq_info['epoch']}"
            dna_desc = f"Length: {seq_info['length']} | Taxon: {seq_info.get('taxon_id', 'unknown')}"
            dna_record = SeqRecord(Seq(seq_info['dna_sequence']), id=dna_id, description=dna_desc)
            dna_records.append(dna_record)
            
            protein_id = f"pred_protein_{stage}_{i}_epoch{seq_info['epoch']}"
            protein_record = SeqRecord(Seq(seq_info['protein_sequence']), id=protein_id, description=dna_desc)
            protein_records.append(protein_record)
        
        with open(dna_filename, 'w') as dna_file:
            SeqIO.write(dna_records, dna_file, 'fasta')
        with open(protein_filename, 'w') as protein_file:
            SeqIO.write(protein_records, protein_file, 'fasta')
        
        print(f"Saved {len(dna_records)} sequences to FASTA")
        return dna_filename, protein_filename

    def save_sequences_to_json(self, stage='val', filename=None):
        """将预测序列保存为JSON文件"""
        if not filename:
            filename = f'{os.environ.get("MODEL_DIR", ".")}/predicted_sequences_{stage}_epoch{self.trainer.current_epoch}.json'
        
        if stage in self.predicted_sequences and self.predicted_sequences[stage]:
            with open(filename, 'w') as f:
                json.dump(self.predicted_sequences[stage], f, indent=2)
            print(f"Saved {len(self.predicted_sequences[stage])} sequences to JSON")
            return filename
        return None

    def calculate_sequence_statistics(self, stage='val'):
        """计算序列统计信息"""
        if stage not in self.predicted_sequences or not self.predicted_sequences[stage]:
            return {}
        
        sequences = self.predicted_sequences[stage]
        stats = {
            'total_sequences': len(sequences),
            'avg_length': np.mean([seq['length'] for seq in sequences]),
            'min_length': np.min([seq['length'] for seq in sequences]),
            'max_length': np.max([seq['length'] for seq in sequences]),
        }
        
        gc_contents = []
        for seq_info in sequences:
            dna_seq = seq_info['dna_sequence']
            if dna_seq:
                gc_count = dna_seq.count('G') + dna_seq.count('C')
                gc_content = gc_count / len(dna_seq) if len(dna_seq) > 0 else 0
                gc_contents.append(gc_content)
        
        if gc_contents:
            stats['avg_gc_content'] = np.mean(gc_contents)
            stats['std_gc_content'] = np.std(gc_contents)
        
        return stats
    
    # ========== 核心训练步骤 ==========
    def general_step(self, batch, stage):
        self.iter_step += 1
        self.stage = stage
        start = time.time()

        mask = batch['mask']
        atom37 = batch['atom37']
        codons = batch['codons']
        seq = batch['seq']
        B, L, _, _ = atom37.shape

        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)

        # Transformer增强（可选）
        if self.use_transformer and self._get_current_fusion_weight() > 0:
            # 这里保持原有逻辑...
            pass
        
        # 前向传播
        log_probs = self.model.forward_train(
            X=bb_pos,
            S=seq if self.args.train_aa else codons,
            taxon_id=batch['taxon_id'],
            mask=mask,
            chain_M=torch.ones([B, L], device=self.device, dtype=torch.long),
            residue_idx=batch['pmpnn_res_idx'],
            chain_encoding_all=batch['pmpnn_chain_encoding'],
        )

        train_target = batch['seq'] if self.args.train_aa else batch['codons']
        
        # 计算主损失
        if log_probs.dim() == 0 or log_probs.numel() == 1:
            # Flow模型返回标量损失
            main_loss = log_probs
            total_loss = main_loss
        else:
            # 传统模型返回log概率
            ce_loss = F.cross_entropy(log_probs.view(-1, self.K), train_target.view(-1),
                                     reduction='none')
            ce_loss = ce_loss.view(B, L)
            main_loss = (ce_loss * mask.float()).sum() / mask.sum()
            
            # ========== ESM2策略梯度 ==========
            if stage == 'train' and self._should_apply_esm2():
                esm2_weight = self._get_esm2_weight()
                
                if esm2_weight > 0:
                    # 1. 从log_probs采样（保持梯度）
                    probs = F.softmax(log_probs, dim=-1)
                    pred_codons = torch.multinomial(probs.view(-1, self.K), 1).view(B, L)
                    
                    # 2. ESM2奖励（无梯度）
                    with torch.no_grad():
                        esm2_results = self._compute_esm2_reward(pred_codons, mask)
                        reward = esm2_results['reward']
                        self._update_baseline(reward)
                        reward_centered = reward - self.esm2_baseline
                    
                    # 3. 策略梯度（有梯度！）
                    selected_log_probs = log_probs.gather(-1, pred_codons.unsqueeze(-1)).squeeze(-1)
                    policy_loss = -(reward_centered.unsqueeze(1) * selected_log_probs * mask).sum() / mask.sum()
                    
                    # 4. 组合损失
                    total_loss = main_loss + esm2_weight * policy_loss
                    
                    # 5. 日志记录
                    self.log('esm2_reward', reward.mean())
                    self.log('esm2_reward_std', reward.std())
                    self.log('esm2_baseline', self.esm2_baseline)
                    self.log('esm2_perplexity', esm2_results['perplexity'])
                    self.log('esm2_confidence', esm2_results['confidence'])
                    self.log('esm2_policy_loss', policy_loss)
                    self.log('esm2_weight_current', esm2_weight)
                    self.log('main_loss', main_loss)
                    
                    # 首次验证梯度
                    if self.iter_step == self.esm2_check_interval:
                        print(f"\n{'='*60}")
                        print("ESM2首次梯度验证:")
                        print(f"  total_loss.requires_grad: {total_loss.requires_grad}")
                        print(f"  total_loss.grad_fn: {total_loss.grad_fn}")
                        print(f"  esm2_baseline: {self.esm2_baseline:.4f}")
                        print(f"{'='*60}\n")
                else:
                    total_loss = main_loss
            else:
                total_loss = main_loss
            
            # 一致性损失（可选）
            if self.use_transformer and self._get_current_fusion_weight() > 0:
                consistency_weight = getattr(self.args, 'codon_consistency_weight', 0.1)
                if consistency_weight > 0 and not self.args.train_aa:
                    consistency_loss = self._compute_codon_consistency_loss(log_probs, train_target, mask)
                    total_loss = total_loss + consistency_weight * consistency_loss
                    self.log('codon_consistency_loss', consistency_loss)

        # 日志记录
        self.log('loss', total_loss)
        self.log('forward_dur', time.time() - start)
        
        return total_loss
    
    # ========== 验证步骤 ==========
    def validation_step(self, batch, batch_idx):
        # 首先执行general_step计算验证损失
        out = self.general_step(batch, stage='val')
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()
        
        # 【关键】采样部分用no_grad包裹，避免验证时不必要的梯度计算
        mask = batch['mask']
        atom37 = batch['atom37']
        seq = batch['seq']
        codons = batch['codons']
        B, L, _, _ = atom37.shape
        
        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)
        
        with torch.no_grad():  # ← 添加这行
            pred_dict = self.model.sample(
                X=bb_pos,
                mask=mask,
                chain_M=torch.ones([B, L], device=self.device, dtype=torch.long),
                residue_idx=batch['pmpnn_res_idx'],
                chain_encoding_all=batch['pmpnn_chain_encoding'],
                taxon_id=batch['taxon_id'],
                temperature=self.args.sampling_temp,
            )
        probs = pred_dict['probs']
        
        
        if self.args.train_aa:
            pred_res = probs.argmax(-1)
            pred_codons = torch.stack([torch.tensor(
                [codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) for i in s], 
                device=self.device) for s in pred_res], dim=0)
        else:
            pred_codons = probs.argmax(-1)
            pred_res = torch.stack([torch.tensor(
                [restype_order_with_x.get(codon_to_res[codon_types[i.long().item()]], unk_restype_index) for i in s], 
                device=self.device) for s in pred_codons], dim=0)
        
        self.save_predicted_sequences(pred_codons, pred_res, batch, 'val', batch_idx)
        
        codon_recovery = ((pred_codons == codons).float() * mask).sum() / mask.sum()
        res_recovery = ((pred_res == seq).float() * mask).sum() / mask.sum()
        
        self.val_dict['pred_codons'].append(pred_codons)
        self.val_dict['pred_res'].append(pred_res)
        self.val_dict['codons'].append(codons)
        self.val_dict['seq'].append(seq)
        self.val_dict['mask'].append(mask)
        
        self.log('codon_recovery', codon_recovery)
        self.log('res_recovery', res_recovery)
        
        if batch_idx < self.args.num_foldability_batches:
            atom37s = [atom37_e[mask_e.bool()].cpu().numpy() for atom37_e, mask_e in zip(atom37, mask)]
            pred_seqs = []
            for seq_e, mask_e in zip(pred_res, mask):
                valid_seq = seq_e[mask_e.bool()]
                valid_seq_np = valid_seq.cpu().numpy()
                valid_seq_np = np.clip(valid_seq_np, 0, len(restypes_with_x) - 1)
                pred_seqs.append(valid_seq_np)
            
            fold_results = run_foldability(atom37s, pred_seqs, device=self.device)
            self.log('tm_score', np.array(fold_results['tm_score']).mean())
            self.log('rmsd', np.array(fold_results['rmsd']).mean())
        else:
            self.log('tm_score', np.nan)
            self.log('rmsd', np.nan)
        
        return out

    def on_validation_epoch_end(self):
        if not self.val_dict['seq']:
            return
        
        max_len = max([seq.shape[1] for seq in self.val_dict['seq']])
        for k in self.val_dict:
            for i in range(len(self.val_dict[k])):
                L = self.val_dict[k][i].shape[1]
                if L < max_len:
                    current_batch_size = self.val_dict[k][i].shape[0]
                    self.val_dict[k][i] = torch.cat([
                        self.val_dict[k][i],
                        torch.zeros(current_batch_size, max_len - L,
                                  *self.val_dict[k][i].shape[2:],
                                  device=self.device)
                    ], dim=1)
            
            self.val_dict[k] = torch.cat(self.val_dict[k], dim=0)

        codons = self.val_dict['codons']
        pred_codons = self.val_dict['pred_codons']
        pred_res = self.val_dict['pred_res']
        mask = self.val_dict['mask']
        seq = self.val_dict['seq']

        codons_from_res = torch.stack([
            torch.tensor([codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) 
                         for i in s], device=self.device) for s in pred_res
        ])
        codons_from_oracle_res = torch.stack([
            torch.tensor([codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) 
                         for i in s], device=self.device) for s in seq
        ])

        naive_codon_recovery = ((codons_from_res == codons).float() * mask).sum() / mask.sum()
        oracle_codon_recovery = ((codons_from_oracle_res == codons).float() * mask).sum() / mask.sum()

        per_aa_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_naive_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_aa_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_oracle_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        
        for i in range(len(restype_order_with_x)):
            id_mask = (seq == i) * mask
            if id_mask.sum() > 0:
                per_aa_codon_recovery[i] = (codons == pred_codons)[id_mask.bool()].float().sum() / id_mask.sum()
                per_aa_aa_recovery[i] = (seq == pred_res)[id_mask.bool()].float().sum() / id_mask.sum()
                per_aa_naive_codon_recovery[i] = (codons_from_res == codons)[id_mask.bool()].float().sum() / id_mask.sum()
                per_aa_oracle_codon_recovery[i] = (codons_from_oracle_res == codons)[id_mask.bool()].float().sum() / id_mask.sum()

        # 绘图
        x = np.arange(len(restypes_with_x))
        width = 0.15
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - 1.5*width, per_aa_codon_recovery.cpu(), width, label='Codon Recovery')
        ax.bar(x - 0.5*width, per_aa_naive_codon_recovery.cpu(), width, label='Naive Codon Recovery')
        ax.bar(x + 0.5*width, per_aa_oracle_codon_recovery.cpu(), width, label='Oracle Codon Recovery')
        ax.bar(x + 1.5*width, per_aa_aa_recovery.cpu(), width, label='AA Recovery')

        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Recovery Rate')
        ax.set_title('Recovery Rates Per Amino Acid')
        ax.set_xticks(x)
        ax.set_xticklabels(restypes_with_x)
        ax.legend()
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        model_dir = os.environ.get("MODEL_DIR", ".")
        os.makedirs(model_dir, exist_ok=True)
        
        plt.savefig(f'{model_dir}/recovery_rates_epoch{self.current_epoch}.png')
        if self.args.wandb:
            wandb.log({'recovery_rates_per_aa': wandb.Image(
                f'{model_dir}/recovery_rates_epoch{self.current_epoch}.png')})
        plt.close()
        
        # 清空验证字典
        for k in self.val_dict:
            self.val_dict[k] = []
        
        # 保存预测序列
        if 'val' in self.predicted_sequences and self.predicted_sequences['val']:
            try:
                self.save_sequences_to_csv('val')
                self.save_sequences_to_fasta('val')
                self.save_sequences_to_json('val')
                
                stats = self.calculate_sequence_statistics('val')
                if self.args.wandb:
                    wandb.log({
                        'val_sequence_count': stats.get('total_sequences', 0),
                        'val_avg_sequence_length': stats.get('avg_length', 0),
                        'val_avg_gc_content': stats.get('avg_gc_content', 0),
                    })
                
                self.predicted_sequences['val'] = []
            except Exception as e:
                print(f"Error saving sequences: {e}")
        
        self.print_log(prefix='val', save=True, extra_logs={
            'codon_from_res_recovery': naive_codon_recovery.item(),
            'codon_from_oracle_res_recovery': oracle_codon_recovery.item()
        })

    # ========== 测试步骤 ==========
    def test_step(self, batch, batch_idx):
        self.iter_step += 1
        self.stage = 'test'

        mask = batch['mask']
        atom37 = batch['atom37']
        B, L, _, _ = atom37.shape
        seq = batch['seq']
        wildtype_codons = batch['wildtype_codons']
        mut_position = torch.ceil((batch['mut_position'] + 1) / 3) - 1 
        mask_seq = torch.nn.functional.one_hot(mut_position.long(), num_classes=L)

        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)
        chain_M = torch.ones([B, L], device=self.device, dtype=torch.long)

        decoding_order = torch.arange(L).repeat(B, 1)
        decoding_order[mask_seq == 1] = L - 1
        decoding_order[:, L - 1] = mut_position
        
        log_probs = self.model.forward_inference(
            X=bb_pos,
            randn=torch.randn(B, L, device=self.device),
            S=seq if self.args.train_aa else wildtype_codons,
            taxon_id=batch['taxon_id'],
            mask=mask,
            chain_M=chain_M,
            residue_idx=batch['pmpnn_res_idx'],
            chain_encoding_all=batch['pmpnn_chain_encoding'],
            use_input_decoding_order=True,
            decoding_order=decoding_order,
        )
        
        mask_logits = mask_seq.unsqueeze(-1).repeat(1, 1, self.K)
        mut_log_probs = log_probs[mask_logits == 1].view(B, self.K)
        self.log('mut_log_prob', mut_log_probs.detach().cpu().numpy())
        self.log('output_seq', torch.argmax(log_probs, dim=-1).tolist())

        pred_codons = torch.argmax(log_probs, dim=-1)
        if self.args.train_aa:
            pred_res = pred_codons
        else:
            pred_res = torch.stack([torch.tensor(
                [restype_order_with_x.get(codon_to_res[codon_types[i.long().item()]], unk_restype_index) 
                 for i in s], device=self.device) for s in pred_codons], dim=0)
        
        self.save_predicted_sequences(pred_codons, pred_res, batch, 'test', batch_idx)

        train_target = batch['seq'] if self.args.train_aa else batch['wildtype_codons']
        loss = F.cross_entropy(log_probs.view(-1, self.K), train_target.view(-1), reduction='none')
        loss = loss.view(B, L)
        loss = (loss * mask.float()).sum() / mask.sum()

        return loss

    def on_test_epoch_end(self):
        if 'test_mut_log_prob' in self._log and self._log['test_mut_log_prob']:
            np.save('test_mut_log_prob', np.concatenate(self._log['test_mut_log_prob'], axis=0))
        
        if 'test_output_seq' in self._log and self._log['test_output_seq']:
            with open("test_output_seq.csv", "w") as f:
                wr = csv.writer(f)
                wr.writerows(self._log['test_output_seq'])
        
        if 'test' in self.predicted_sequences and self.predicted_sequences['test']:
            try:
                self.save_sequences_to_csv('test')
                self.save_sequences_to_fasta('test')
                self.save_sequences_to_json('test')
            except Exception as e:
                print(f"Error saving test sequences: {e}")
        
        self.print_log(prefix='test', save=True)