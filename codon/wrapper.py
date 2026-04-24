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
from codon.utils.foldability_utils import run_foldability
from codon.utils.logging import get_logger
from codon.utils.pmpnn import ProteinMPNN

import torch.nn as nn
import torch.nn.functional as F

# 添加序列保存所需的导入
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
        return self.test_step(batch, batch_idx)

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
        # 创建注意力掩码
        if mask is not None:
            key_padding_mask = ~mask.bool()  # True的位置被忽略
        else:
            key_padding_mask = None
            
        # 自注意力 + 残差连接
        attn_out, attention_weights = self.attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        # 应用掩码
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        
        return x, attention_weights


class PMPNNWrapper(Wrapper):
    def __init__(self, args):
        super().__init__(args)
        self.K = len(restype_order_with_x) if args.train_aa else len(codon_order)
        self.model = ProteinMPNN(args, vocab=self.K, node_features=args.hidden_dim,
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
        
        if self.use_transformer:
            print(f"启用Transformer增强: heads={args.transformer_heads}, layers={args.transformer_layers}")
            
            # 自注意力层
            self.transformer_layers = nn.ModuleList([
                TransformerBlock(
                    d_model=args.hidden_dim,
                    n_heads=args.transformer_heads,
                    d_ff=args.hidden_dim * 4,
                    dropout=args.dropout
                ) for _ in range(args.transformer_layers)
            ])
            
            # 特征融合层
            self.feature_fusion = nn.Sequential(
                nn.Linear(args.hidden_dim * 2, args.hidden_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden_dim, args.hidden_dim)
            )
            
            # 位置编码
            if getattr(args, 'use_positional_encoding', True):
                max_seq_len = getattr(args, 'max_seq_len', 2048)
                self.register_buffer('positional_encoding', 
                                   self._create_positional_encoding(max_seq_len, args.hidden_dim))
            
            # 渐进式训练参数
            self.transformer_start_epoch = getattr(args, 'transformer_start_epoch', 0)
            self.max_fusion_weight = getattr(args, 'transformer_fusion_weight', 0.5)
            
    def _create_positional_encoding(self, max_len, d_model):
        """创建正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def _get_current_fusion_weight(self):
        """根据当前epoch计算融合权重"""
        if not self.use_transformer:
            return 0.0
        
        current_epoch = self.trainer.current_epoch if hasattr(self, 'trainer') else 0
        
        if current_epoch < self.transformer_start_epoch:
            return 0.0
        
        # 线性增加融合权重
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
        
        # 1. 添加位置编码
        if hasattr(self, 'positional_encoding'):
            pos_enc = self.positional_encoding[:, :L, :].to(h_V.device)
            h_V_with_pos = h_V + pos_enc * mask.unsqueeze(-1)
        else:
            h_V_with_pos = h_V
        
        # 2. 应用Transformer层
        h_V_transformed = h_V_with_pos
        for transformer_layer in self.transformer_layers:
            h_V_transformed, _ = transformer_layer(h_V_transformed, mask)
        
        # 3. 特征融合
        fused_features = torch.cat([h_V, h_V_transformed], dim=-1)
        h_V_enhanced = self.feature_fusion(fused_features)
        
        # 4. 加权融合
        h_V_final = (1 - fusion_weight) * h_V + fusion_weight * h_V_enhanced
        
        # 5. 应用掩码
        return h_V_final * mask.unsqueeze(-1)
    
    def _compute_codon_consistency_loss(self, log_probs, targets, mask):
        """计算密码子一致性损失"""
        B, L, K = log_probs.shape
        consistency_loss = 0.0
        count = 0
        
        # 计算相邻位置的预测分布相似性
        for i in range(L - 1):
            valid_current = mask[:, i].bool()
            valid_next = mask[:, i + 1].bool()
            valid_pair = valid_current & valid_next
            
            if valid_pair.any():
                prob_current = F.softmax(log_probs[valid_pair, i], dim=-1)
                prob_next = F.softmax(log_probs[valid_pair, i + 1], dim=-1)
                
                # 使用平滑L1损失
                consistency_loss += F.smooth_l1_loss(prob_current, prob_next, reduction='mean')
                count += 1
        
        return consistency_loss / max(count, 1)

    def codons_to_dna_string(self, codon_indices):
        """将密码子索引转换为DNA字符串"""
        dna_seq = ""
        for idx in codon_indices:
            if idx.item() < len(codon_types):
                dna_seq += codon_types[idx.item()]
            else:
                dna_seq += "NNN"  # 未知密码子用NNN表示
        return dna_seq

    def residues_to_protein_string(self, residue_indices):
        """将氨基酸索引转换为蛋白质字符串"""
        protein_seq = ""
        for idx in residue_indices:
            if idx.item() < len(restypes_with_x):
                protein_seq += restypes_with_x[idx.item()]
            else:
                protein_seq += "X"  # 未知氨基酸用X表示
        return protein_seq

    def save_predicted_sequences(self, pred_codons, pred_res, batch, stage='val', batch_idx=0):
        """保存预测的DNA序列和蛋白质序列"""
        mask = batch['mask']
        B, L = pred_codons.shape
        
        for b in range(B):
            # 获取有效长度
            valid_length = mask[b].sum().item()
            
            if valid_length == 0:
                continue
            
            # 提取有效的预测序列
            valid_pred_codons = pred_codons[b, :valid_length]
            valid_pred_res = pred_res[b, :valid_length]
            
            # 转换为实际的序列字符串
            dna_seq = self.codons_to_dna_string(valid_pred_codons)
            protein_seq = self.residues_to_protein_string(valid_pred_res)
            
            # 保存序列信息
            seq_info = {
                'batch_idx': batch_idx,
                'sample_idx': b,
                'stage': stage,
                'epoch': self.trainer.current_epoch if hasattr(self, 'trainer') else 0,
                'length': valid_length,
                'dna_sequence': dna_seq,
                'protein_sequence': protein_seq,
            }
            
            # 添加原始序列（如果有）
            if 'seq' in batch:
                original_seq = batch['seq'][b, :valid_length]
                seq_info['original_protein'] = self.residues_to_protein_string(original_seq)
                
            if 'codons' in batch:
                original_codons = batch['codons'][b, :valid_length]
                seq_info['original_dna'] = self.codons_to_dna_string(original_codons)
            
            # 添加其他元数据
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
        else:
            print(f"No sequences to save for stage: {stage}")
            return None

    def save_sequences_to_fasta(self, stage='val', dna_filename=None, protein_filename=None):
        """将预测序列保存为FASTA文件"""
        if not dna_filename:
            dna_filename = f'{os.environ.get("MODEL_DIR", ".")}/predicted_dna_{stage}_epoch{self.trainer.current_epoch}.fasta'
        if not protein_filename:
            protein_filename = f'{os.environ.get("MODEL_DIR", ".")}/predicted_protein_{stage}_epoch{self.trainer.current_epoch}.fasta'
        
        if stage not in self.predicted_sequences or not self.predicted_sequences[stage]:
            print(f"No sequences to save for stage: {stage}")
            return None, None
        
        # 保存DNA序列和蛋白质序列
        dna_records = []
        protein_records = []
        
        for i, seq_info in enumerate(self.predicted_sequences[stage]):
            # DNA FASTA记录
            dna_id = f"pred_dna_{stage}_{i}_epoch{seq_info['epoch']}_batch{seq_info['batch_idx']}_sample{seq_info['sample_idx']}"
            dna_desc = f"Predicted DNA sequence | Length: {seq_info['length']} | Taxon: {seq_info.get('taxon_id', 'unknown')}"
            dna_record = SeqRecord(Seq(seq_info['dna_sequence']), id=dna_id, description=dna_desc)
            dna_records.append(dna_record)
            
            # 蛋白质FASTA记录
            protein_id = f"pred_protein_{stage}_{i}_epoch{seq_info['epoch']}_batch{seq_info['batch_idx']}_sample{seq_info['sample_idx']}"
            protein_desc = f"Predicted protein sequence | Length: {seq_info['length']} | Taxon: {seq_info.get('taxon_id', 'unknown')}"
            protein_record = SeqRecord(Seq(seq_info['protein_sequence']), id=protein_id, description=protein_desc)
            protein_records.append(protein_record)
        
        # 写入FASTA文件
        with open(dna_filename, 'w') as dna_file:
            SeqIO.write(dna_records, dna_file, 'fasta')
        with open(protein_filename, 'w') as protein_file:
            SeqIO.write(protein_records, protein_file, 'fasta')
        
        print(f"Saved {len(dna_records)} DNA sequences to {dna_filename}")
        print(f"Saved {len(protein_records)} protein sequences to {protein_filename}")
        
        return dna_filename, protein_filename

    def save_sequences_to_json(self, stage='val', filename=None):
        """将预测序列保存为JSON文件"""
        if not filename:
            filename = f'{os.environ.get("MODEL_DIR", ".")}/predicted_sequences_{stage}_epoch{self.trainer.current_epoch}.json'
        
        if stage in self.predicted_sequences and self.predicted_sequences[stage]:
            with open(filename, 'w') as f:
                json.dump(self.predicted_sequences[stage], f, indent=2)
            print(f"Saved {len(self.predicted_sequences[stage])} sequences to {filename}")
            return filename
        else:
            print(f"No sequences to save for stage: {stage}")
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
        
        # GC含量统计
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

    def general_step(self, batch, stage):
        self.iter_step += 1
        self.stage = stage
        start = time.time()

        mask = batch['mask']  # (B, L)
        atom37 = batch['atom37']  # (B, L, 37, 3)
        codons = batch['codons']  # (B, L, 65)
        seq = batch['seq']  # (B, L, 21)
        B, L, _, _ = atom37.shape

        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... 
        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)  # (B, L, 4, 3)

        # 条件性获取中间特征进行Transformer增强
        if self.use_transformer and self._get_current_fusion_weight() > 0:
            # 检查模型是否支持返回隐藏状态
            if hasattr(self.model, 'forward_train') and 'return_hidden' in self.model.forward_train.__code__.co_varnames:
                try:
                    log_probs, h_V_encoder, h_V_decoder = self.model.forward_train(
                        X=bb_pos,
                        S=seq if self.args.train_aa else codons,
                        taxon_id=batch['taxon_id'],
                        mask=mask,
                        chain_M=torch.ones([B, L], device=self.device, dtype=torch.long),
                        residue_idx=batch['pmpnn_res_idx'],
                        chain_encoding_all=batch['pmpnn_chain_encoding'],
                        return_hidden=True
                    )
                    
                    # 应用Transformer增强
                    h_V_enhanced = self._apply_transformer_enhancement(h_V_encoder, mask)
                    
                    # 生成增强的logits
                    if hasattr(self.model, 'W_out'):
                        enhanced_logits = self.model.W_out(h_V_enhanced)
                        
                        # 融合原始和增强的logits
                        fusion_weight = self._get_current_fusion_weight()
                        fused_logits = (1 - fusion_weight) * log_probs + fusion_weight * enhanced_logits
                        log_probs = F.log_softmax(fused_logits, dim=-1)
                        
                        # 记录融合权重
                        self.log('transformer_fusion_weight', fusion_weight)
                    
                except Exception as e:
                    print(f"Transformer enhancement failed: {e}")
                    # 如果失败，使用原始MPNN前向传播
                    log_probs = self.model.forward_train(
                        X=bb_pos,
                        S=seq if self.args.train_aa else codons,
                        taxon_id=batch['taxon_id'],
                        mask=mask,
                        chain_M=torch.ones([B, L], device=self.device, dtype=torch.long),
                        residue_idx=batch['pmpnn_res_idx'],
                        chain_encoding_all=batch['pmpnn_chain_encoding'],
                    )
            else:
                # 模型不支持返回隐藏状态，使用原始前向传播
                log_probs = self.model.forward_train(
                    X=bb_pos,
                    S=seq if self.args.train_aa else codons,
                    taxon_id=batch['taxon_id'],
                    mask=mask,
                    chain_M=torch.ones([B, L], device=self.device, dtype=torch.long),
                    residue_idx=batch['pmpnn_res_idx'],
                    chain_encoding_all=batch['pmpnn_chain_encoding'],
                )
        else:
            # 原始MPNN前向传播
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
        loss = torch.nn.functional.cross_entropy(log_probs.view(-1, self.K), train_target.view(-1),
                                                 reduction='none')  # (B * L)
        loss = loss.view(B, L)
        loss = loss * mask.float()  # (B, L)
        loss = loss.sum() / mask.sum()

        # 额外损失项
        if self.use_transformer and self._get_current_fusion_weight() > 0:
            # 密码子一致性损失
            consistency_weight = getattr(self.args, 'codon_consistency_weight', 0.1)
            if consistency_weight > 0 and not self.args.train_aa:
                consistency_loss = self._compute_codon_consistency_loss(log_probs, train_target, mask)
                loss = loss + consistency_weight * consistency_loss
                self.log('codon_consistency_loss', consistency_loss)

        # 日志记录
        self.log('loss', loss)
        self.log('forward_dur', time.time() - start)
        self.log('dur', time.time() - self.last_log_time)
        self.last_log_time = time.time()
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.general_step(batch, stage='val')
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()

        mask = batch['mask']  # (B, L)
        atom37 = batch['atom37']  # (B, L, 37, 3)
        seq = batch['seq']  # (B, L, 21)
        codons = batch['codons']  # (B, L, 65)
        B, L, _, _ = atom37.shape

        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... 
        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)  # (B, L, 4, 3)

        pred_dict = self.model.sample(
            X=bb_pos,
            randn=torch.randn(B, L, device=self.device),
            S_true=seq if self.args.train_aa else codons,
            taxon_id=batch['taxon_id'],
            chain_mask=torch.ones([B, L], device=self.device, dtype=torch.long),
            chain_encoding_all=batch['pmpnn_chain_encoding'],
            residue_idx=batch['pmpnn_res_idx'],
            mask=mask,
            temperature=self.args.sampling_temp,
            omit_AAs_np=np.zeros(self.K).astype(np.float32),
            bias_AAs_np=np.zeros(self.K),
            chain_M_pos=torch.ones([B, L], device=self.device, dtype=torch.long),
            bias_by_res=torch.zeros([B, L, self.K], dtype=torch.float32, device=self.device)
        )
        probs = pred_dict['probs']  # (B, L, self.K)

        if self.args.train_aa:
            pred_res = probs.argmax(-1)
            pred_codons = torch.stack([torch.tensor(
                [codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) for i in seq], device=self.device) for seq in
                                    pred_res], dim=0)
        else:
            pred_codons = probs.argmax(-1)
            pred_res = torch.stack([torch.tensor(
                [restype_order_with_x.get(codon_to_res[codon_types[i.long().item()]], unk_restype_index) for i in seq], device=self.device) for seq
                in pred_codons], dim=0)

        # 保存预测序列
        self.save_predicted_sequences(pred_codons, pred_res, batch, 'val', batch_idx)
        
        codon_recovery = (pred_codons == codons).float() * mask
        codon_recovery = codon_recovery.sum() / mask.sum()
        res_recovery = (pred_res == batch['seq']).float() * mask
        res_recovery = res_recovery.sum() / mask.sum()

        self.val_dict['pred_codons'].append(pred_codons)
        self.val_dict['pred_res'].append(pred_res)
        self.val_dict['codons'].append(codons)
        self.val_dict['seq'].append(batch['seq'])
        self.val_dict['mask'].append(mask)
        self.log('codon_recovery', codon_recovery)
        self.log('res_recovery', res_recovery)

        if batch_idx < self.args.num_foldability_batches:
            atom37s = [atom37_e[mask_e.bool()].cpu().numpy() for atom37_e, mask_e in zip(atom37, mask)]
            pred_seqs = [seq_e[mask_e.bool()] for seq_e, mask_e in zip(pred_res, mask)]
            fold_results = run_foldability(atom37s, pred_seqs, device=self.device)
            self.log('tm_score', np.array(fold_results['tm_score']).mean())
            self.log('rmsd', np.array(fold_results['rmsd']).mean())
        else:
            self.log('tm_score', np.nan)
            self.log('rmsd', np.nan)
        return out

    def on_validation_epoch_end(self):
        max_len = max([seq.shape[1] for seq in self.val_dict['seq']])
        for k in self.val_dict:
            for i in range(len(self.val_dict[k])):
                L = self.val_dict[k][i].shape[1]
                if L < max_len:
                    self.val_dict[k][i] = torch.cat([self.val_dict[k][i], torch.zeros(self.args.batch_size, max_len - L,
                                                                                      *self.val_dict[k][i].shape[2:],
                                                                                      device=self.device)],
                                                    dim=1)
            self.val_dict[k] = torch.cat(self.val_dict[k], dim=0)

        codons = self.val_dict['codons']
        pred_codons = self.val_dict['pred_codons']
        pred_res = self.val_dict['pred_res']
        mask = self.val_dict['mask']
        B, L = codons.shape
        seq = self.val_dict['seq']

        prob_c_given_a = torch.zeros(self.K, len(restype_order_with_x), device=self.device)
        for i in range(self.K):
            for j in range(len(restype_order_with_x)):
                num_aa = ((seq == j) * mask).sum().float()
                if num_aa != 0:
                    prob_c_given_a[i, j] = ((codons == i) * (seq == j) * mask).sum().float() / num_aa

        codons_from_res = torch.stack([torch.tensor([codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) for i in s], device=self.device) for s in pred_res])
        codons_from_oracle_res = torch.stack([torch.tensor([codon_order.get(res_to_codon[restypes_with_x[i.long().item()]], unk_codon_index) for i in s], device=self.device) for s in seq])

        naive_codon_recovery = ((codons_from_res == codons).float() * mask).sum() / mask.sum()
        oracle_codon_recovery = ((codons_from_oracle_res == codons).float() * mask).sum() / mask.sum()

        per_aa_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_naive_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_aa_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        per_aa_oracle_codon_recovery = torch.zeros(len(restype_order_with_x), device=self.device)
        for i in range(len(restype_order_with_x)):
            id_mask = (seq == i) * mask
            if id_mask.sum() > 0:  # 避免除零错误
                per_aa_codon_recovery[i] = (codons == pred_codons)[id_mask.bool()].float().sum() / id_mask.sum()
                per_aa_aa_recovery[i] = (seq == pred_res)[id_mask.bool()].float().sum() / id_mask.sum()
                per_aa_naive_codon_recovery[i] = (codons_from_res == codons)[id_mask.bool()].float().sum() / id_mask.sum()
                per_aa_oracle_codon_recovery[i] = (codons_from_oracle_res == codons)[id_mask.bool()].float().sum() / id_mask.sum()

        x = np.arange(len(restypes_with_x))
        width = 0.15
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width / 2 - width, per_aa_codon_recovery.cpu(), width, label='Codon Recovery')
        rects2 = ax.bar(x - width / 2, per_aa_naive_codon_recovery.cpu(), width, label='Naive Codon Recovery')
        rects3 = ax.bar(x + width / 2, per_aa_oracle_codon_recovery.cpu(), width, label='Oracle Codon Recovery')
        rects4 = ax.bar(x + width / 2 + width, per_aa_aa_recovery.cpu(), width, label='Amino Acid Recovery')

        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Recovery Rate')
        ax.set_title('Recovery Rates Per Amino Acid')
        ax.set_xticks(x)
        ax.set_xticklabels(restypes_with_x)
        ax.legend()
        ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        # 确保目录存在
        model_dir = os.environ.get("MODEL_DIR", ".")
        os.makedirs(model_dir, exist_ok=True)
        
        plt.savefig(f'{model_dir}/recovery_rates_per_aa_epoch{self.current_epoch}_iter{self.iter_step}.png')
        if self.args.wandb:
            wandb.log({'recovery_rates_per_aa': wandb.Image(
                f'{model_dir}/recovery_rates_per_aa_epoch{self.current_epoch}_iter{self.iter_step}.png')})
        plt.close()  # 关闭图形以释放内存
        
        # 清空验证字典
        for k in self.val_dict:
            self.val_dict[k] = []
        
        # 保存预测序列
        if hasattr(self, 'predicted_sequences') and 'val' in self.predicted_sequences and self.predicted_sequences['val']:
            try:
                csv_file = self.save_sequences_to_csv('val')
                dna_fasta, protein_fasta = self.save_sequences_to_fasta('val')
                json_file = self.save_sequences_to_json('val')
                
                # 计算统计信息
                stats = self.calculate_sequence_statistics('val')
                print(f"Validation sequence statistics: {stats}")
                
                # 记录到wandb
                if self.args.wandb:
                    wandb.log({
                        'val_sequence_count': stats.get('total_sequences', 0),
                        'val_avg_sequence_length': stats.get('avg_length', 0),
                        'val_avg_gc_content': stats.get('avg_gc_content', 0),
                    })
                
                # 清空当前epoch的序列数据
                self.predicted_sequences['val'] = []
                
            except Exception as e:
                print(f"Error saving sequences: {e}")
        
        # 打印日志
        self.print_log(prefix='val', save=True, extra_logs={
            'codon_from_res_recovery': naive_codon_recovery.item(),
            'codon_from_oracle_res_recovery': oracle_codon_recovery.item()
        })

    def test_step(self, batch, batch_idx):
        self.iter_step += 1
        self.stage = 'test'

        mask = batch['mask']  # (B, L)
        atom37 = batch['atom37']  # (B, L, 37, 3)
        B, L, _, _ = atom37.shape
        seq = batch['seq']  # (B, L, 21)
        wildtype_codons = batch['wildtype_codons']  # (B, L, 65)
        mut_codons = batch['mut_codons']  # (B, L, 65)
        mut_position = torch.ceil((batch['mut_position'] + 1) / 3) - 1 
        mask_seq = torch.nn.functional.one_hot(mut_position.long(), num_classes=L)  # (B, L) 

        # take N, CA, C, and O. The order in atom37 is N, CA, C, CB, O ... 
        bb_pos = torch.cat([atom37[:, :, :3, :], atom37[:, :, 4:5, :]], dim=2)  # (B, L, 4, 3)

        chain_M = torch.ones([B, L], device=self.device, dtype=torch.long)

        decoding_order = torch.arange(L).repeat(B, 1)  # (B, L)
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
        )  # (B, L, 65)
        
        mask_logits = mask_seq.unsqueeze(-1).repeat(1, 1, self.K)  # (B, L, 65)
        mut_log_probs = log_probs[mask_logits == 1].view(B, self.K)  # (B, 65)
        self.log('mut_log_prob', mut_log_probs.detach().cpu().numpy())
        self.log('output_seq', torch.argmax(log_probs, dim=-1).tolist())

        # 保存测试序列
        pred_codons = torch.argmax(log_probs, dim=-1)
        if self.args.train_aa:
            pred_res = pred_codons
        else:
            pred_res = torch.stack([torch.tensor(
                [restype_order_with_x.get(codon_to_res[codon_types[i.long().item()]], unk_restype_index) 
                 for i in seq], device=self.device) for seq in pred_codons], dim=0)
        
        self.save_predicted_sequences(pred_codons, pred_res, batch, 'test', batch_idx)

        train_target = batch['seq'] if self.args.train_aa else batch['wildtype_codons']
        loss = torch.nn.functional.cross_entropy(log_probs.view(-1, self.K), train_target.view(-1),
                                                 reduction='none')  # (B * L)
        loss = loss.view(B, L)
        loss = loss * mask.float()  # (B, L)
        loss = loss.sum() / mask.sum()

        return loss

    def on_test_epoch_end(self):
        # 保存原有的测试结果
        if 'test_mut_log_prob' in self._log and self._log['test_mut_log_prob']:
            np.save('test_mut_log_prob', np.concatenate(self._log['test_mut_log_prob'], axis=0))
        
        if 'test_output_seq' in self._log and self._log['test_output_seq']:
            with open("test_output_seq.csv", "w") as f:
                wr = csv.writer(f)
                wr.writerows(self._log['test_output_seq'])
        
        # 保存预测序列
        if hasattr(self, 'predicted_sequences') and 'test' in self.predicted_sequences and self.predicted_sequences['test']:
            try:
                self.save_sequences_to_csv('test')
                self.save_sequences_to_fasta('test')
                self.save_sequences_to_json('test')
                
                stats = self.calculate_sequence_statistics('test')
                print(f"Test sequence statistics: {stats}")
                
            except Exception as e:
                print(f"Error saving test sequences: {e}")
        
        self.print_log(prefix='test', save=True)