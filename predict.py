"""ThermoFlow 最优预测脚本
支持：① 全序列恢复  ② 指定位点预测
作者：基于原始训练代码重构
"""

import os
import sys
import csv
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from openfold.np.residue_constants import (
    restypes_with_x,
    restype_order_with_x,
    unk_restype_index,
)
from codon.datasets import AFDBDataset, seq_collate
from codon.flow_wrapper import PMPNNWrapper
from codon.utils.codon_const import codon_order, codon_types, codon_to_res, unk_codon_index

# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 配置类（集中管理所有参数，避免散落各处）
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PredictConfig:
    # ── 路径 ──────────────────────────────────────────────
    checkpoint_path: str = "/model/fianl.ckpt"
    data_csv: str = "./your_data.csv"
    afdb_dir: str = "./"
    output_dir: str = "./predict_outputs"

    # ── 预测模式 ──────────────────────────────────────────
    # "full"   → 全序列恢复（所有位点都重新预测）
    # "masked" → 只预测 data_csv 中 ID 列指定的位点，其余保留真实值
    mode: str = "masked"          # "full" | "masked"

    # ── 结构掩码策略 ──────────────────────────────────────
    # "sidechain"      → 目标位点侧链清零（最常用）
    # "all"            → 目标位点全原子清零
    # "backbone_noise" → 侧链清零 + 主链加高斯噪声
    mask_mode: str = "sidechain"

    # ── 模型超参（需与训练时一致）────────────────────────
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 48
    dropout: float = 0.1
    max_seq_len: int = 750
    taxon_condition: bool = True
    num_taxon_ids: int = 1000
    train_aa: bool = False          # False=密码子模式，True=氨基酸模式
    use_esm2_feedback: bool = False
    use_transformer: bool = False
    high_plddt: bool = False
    overfit: bool = False

    # ── 采样 ─────────────────────────────────────────────
    sampling_temp: float = 0.1      # 越低预测越保守
    batch_size: int = 32
    num_workers: int = 0

    # ── 输出控制 ──────────────────────────────────────────
    save_full_logits: bool = True   # True=保存所有密码子/AA概率（文件较大）
    save_fasta: bool = True
    resume: bool = True             # 断点续跑
    max_samples: int = int(1e15)    # 最多处理样本数
    fasta_bucket_size: int = 1000   # 每个子目录最多文件数
    fasta_num_buckets: int = 300    # 预建子目录数量
    io_threads: int = 8             # 异步写 FASTA 的线程数


# ─────────────────────────────────────────────────────────────────────────────
# CSV 输出字段定义
# ─────────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    # 基础信息
    "global_idx", "af_id", "length", "mode", "mask_mode",
    "taxon_id", "target_positions", "num_target_positions",
    # 序列
    "predicted_dna", "predicted_protein", "true_dna", "true_protein",
    # 全局准确率
    "codon_match_all", "protein_match_all",
    # 目标位点准确率（mode=masked 时有值）
    "codon_match_target", "protein_match_target",
    # GC 含量
    "pred_gc", "true_gc",
    # 位点级详情
    "target_prediction_details",
    # 每个位点完整 logits（JSON 格式）
    "full_logits_json",
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def gc_content(dna: str) -> float:
    if not dna:
        return 0.0
    return (dna.count("G") + dna.count("C")) / len(dna)


def nucleotide_freq(dna: str) -> Dict[str, float]:
    total = len(dna)
    if total == 0:
        return {n: 0.0 for n in "ATGC"}
    c = Counter(dna)
    return {n: c.get(n, 0) / total for n in "ATGC"}


def _codon_idx_to_str(idx: int) -> str:
    return codon_types[idx] if idx < len(codon_types) else "NNN"


def _res_idx_to_str(idx: int) -> str:
    return restypes_with_x[idx] if idx < len(restypes_with_x) else "X"


def _codon_to_aa(codon: str) -> str:
    return codon_to_res.get(codon, "X")


def pred_codons_to_res(pred_codons: torch.Tensor, device) -> torch.Tensor:
    """
    [B, L] codon indices → [B, L] residue indices
    向量化实现，避免逐元素 Python 循环（原始代码瓶颈）
    """
    B, L = pred_codons.shape
    # 预构建 codon→AA 查找表（只做一次）
    lookup = torch.full((len(codon_types) + 1,), unk_restype_index,
                        dtype=torch.long, device=device)
    for i, codon in enumerate(codon_types):
        aa = codon_to_res.get(codon, "X")
        lookup[i] = restype_order_with_x.get(aa, unk_restype_index)

    flat = pred_codons.clamp(0, len(codon_types)).reshape(-1)
    return lookup[flat].reshape(B, L)


# ─────────────────────────────────────────────────────────────────────────────
# 结构掩码
# ─────────────────────────────────────────────────────────────────────────────
def apply_structure_mask(
    atom37: torch.Tensor,       # [B, L, 37, 3]
    position_mask: torch.Tensor,  # [B, L]  1=掩码位点
    mask_mode: str = "sidechain",
) -> torch.Tensor:
    masked = atom37.clone()
    pm = position_mask.unsqueeze(-1).unsqueeze(-1).float()  # [B,L,1,1]

    if mask_mode == "all":
        masked = masked * (1.0 - pm)

    elif mask_mode in ("sidechain", "backbone_noise"):
        sc = torch.zeros(atom37.shape[2], device=atom37.device)
        sc[4:] = 1.0
        sc = sc.view(1, 1, -1, 1)
        masked = masked * (1.0 - pm * sc)

        if mask_mode == "backbone_noise":
            noise = torch.randn_like(atom37[:, :, :4, :]) * 0.1
            masked[:, :, :4, :] += noise * pm[:, :, :4, :]

    else:
        raise ValueError(f"未知 mask_mode: {mask_mode}")

    return masked


# ─────────────────────────────────────────────────────────────────────────────
# FASTA 异步写盘
# ─────────────────────────────────────────────────────────────────────────────
def _fasta_path(fasta_dir: str, global_idx: int, af_id: str,
                bucket_size: int = 1000) -> str:
    bucket = global_idx // bucket_size
    bucket_dir = os.path.join(fasta_dir, f"{bucket:04d}")
    return os.path.join(bucket_dir, f"{af_id}.fasta")


def _write_fasta(path: str, af_id: str, pred_protein: str, true_protein: str):
    with open(path, "w") as f:
        f.write(f">{af_id}|pred\n{pred_protein}\n")
        f.write(f">{af_id}|true\n{true_protein}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 构建 position_mask（核心逻辑统一）
# ─────────────────────────────────────────────────────────────────────────────
def build_position_mask(
    mask: torch.Tensor,          # [B, L]
    target_map: Dict[int, List[int]],
    sample_counter: int,
    mode: str,
    device,
) -> torch.Tensor:
    """
    mode="full"   → 所有有效位点都设为 1（全序列预测）
    mode="masked" → 只将 target_map 中指定的位点设为 1；
                    若某样本无指定位点，则退化为全序列预测
    """
    B, L = mask.shape
    position_mask = torch.zeros([B, L], device=device, dtype=torch.long)

    for b in range(B):
        global_idx = sample_counter + b
        valid_len = int(mask[b].sum().item())

        if mode == "full":
            position_mask[b] = mask[b]
        else:  # masked
            target_pos = target_map.get(global_idx, [])
            if not target_pos:
                # 没有指定位点 → 全序列
                position_mask[b] = mask[b]
            else:
                for pos in target_pos:
                    if pos < valid_len:
                        position_mask[b, pos] = 1

    return position_mask


# ─────────────────────────────────────────────────────────────────────────────
# 提取单样本的位点级 logits 详情
# ─────────────────────────────────────────────────────────────────────────────
def extract_position_details(
    logits_tensor: torch.Tensor,  # [L, K]
    pred_codons: torch.Tensor,    # [L]
    pred_res: torch.Tensor,       # [L]
    true_codons: torch.Tensor,    # [L]
    true_res: torch.Tensor,       # [L]
    target_pos: List[int],
    valid_len: int,
    train_aa: bool,
    top_k: int = 5,
) -> Tuple[List[str], List[Dict], Dict[int, np.ndarray]]:
    """
    返回:
      target_details  : 人类可读的字符串列表
      full_logits_list: 包含完整 logits 的 dict 列表（用于 JSON）
      sample_logits   : {pos: ndarray} 供后续 .pt 文件保存
    """
    target_details = []
    full_logits_list = []
    sample_logits = {}
    device = logits_tensor.device

    for pos in target_pos:
        if pos >= valid_len:
            continue

        pc_idx = pred_codons[pos].item()
        tc_idx = true_codons[pos].item()
        pr_idx = pred_res[pos].item()
        tr_idx = true_res[pos].item()

        pc_str = _codon_idx_to_str(pc_idx)
        tc_str = _codon_idx_to_str(tc_idx)
        pred_aa = _res_idx_to_str(pr_idx)
        true_aa = _res_idx_to_str(tr_idx)

        pos_logits = logits_tensor[pos]  # [K]
        pos_probs = F.softmax(pos_logits, dim=-1)
        sample_logits[pos] = pos_logits.cpu().numpy()

        # Top-K
        topk_v, topk_i = torch.topk(pos_probs, min(top_k, pos_logits.shape[-1]))
        topk_i_np = topk_i.cpu().numpy()
        topk_v_np = topk_v.cpu().numpy()

        if train_aa:
            topk_tok = [_res_idx_to_str(i) for i in topk_i_np]
        else:
            topk_tok = [_codon_idx_to_str(i) for i in topk_i_np]

        topk_str = ",".join(f"{t}({v:.4f})" for t, v in zip(topk_tok, topk_v_np))
        target_details.append(
            f"pos{pos}:{true_aa}({tc_str})->{pred_aa}({pc_str}) [Top{top_k}:{topk_str}]"
        )

        # 完整 logits（所有词汇的概率）
        logits_arr = pos_logits.cpu().numpy()
        probs_arr = pos_probs.cpu().numpy()
        if train_aa:
            vocab = [_res_idx_to_str(i) for i in range(len(logits_arr))]
        else:
            vocab = [_codon_idx_to_str(i) for i in range(len(logits_arr))]

        full_logits_list.append({
            "position": pos,
            "true_aa": true_aa,
            "pred_aa": pred_aa,
            "true_codon": tc_str,
            "pred_codon": pc_str,
            "true_prob": float(probs_arr[tc_idx]),
            "pred_prob": float(probs_arr[pc_idx]),
            "entropy": float(-np.sum(probs_arr * np.log(probs_arr + 1e-10))),
            "top_k": [{"token": t, "prob": float(v)}
                      for t, v in zip(topk_tok, topk_v_np)],
            "all_probs": {v: float(p) for v, p in zip(vocab, probs_arr)},
        })

    return target_details, full_logits_list, sample_logits


# ─────────────────────────────────────────────────────────────────────────────
# 主预测函数
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(cfg: PredictConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.environ.setdefault("MODEL_DIR", cfg.output_dir)

    tag = f"{cfg.mode}_{cfg.mask_mode}"
    csv_path    = os.path.join(cfg.output_dir, f"predictions_{tag}.csv")
    logits_path = os.path.join(cfg.output_dir, f"logits_{tag}.pt")
    fasta_dir   = os.path.join(cfg.output_dir, "fasta")

    # ── 预建 FASTA 分桶目录 ────────────────────────────────
    if cfg.save_fasta:
        log.info("预建 fasta 分桶目录...")
        for i in range(cfg.fasta_num_buckets):
            os.makedirs(os.path.join(fasta_dir, f"{i:04d}"), exist_ok=True)

    # ── 断点续跑 ───────────────────────────────────────────
    completed = set()
    if cfg.resume and os.path.exists(csv_path):
        try:
            completed = set(
                pd.read_csv(csv_path, usecols=["global_idx"])["global_idx"].tolist()
            )
            log.info(f"[RESUME] 已完成 {len(completed)} 条，跳过")
        except Exception as e:
            log.warning(f"读取断点失败，重新开始: {e}")

    csv_is_new = not os.path.exists(csv_path) or not completed
    csv_f = open(csv_path, "a", buffering=1, newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if csv_is_new:
        writer.writeheader()
        csv_f.flush()

    executor = ThreadPoolExecutor(max_workers=cfg.io_threads)

    # ── 加载模型 ───────────────────────────────────────────
    log.info("加载模型...")

    # 构建与训练脚本兼容的 args 对象
    class _Args:
        pass

    args = _Args()
    for k, v in cfg.__dict__.items():
        setattr(args, k, v)
    # 补充训练时要求的其它属性
    args.pretrained_ckpt = cfg.checkpoint_path
    args.num_foldability_batches = 2
    args.accumulate_grad = 1
    args.grad_clip = 1.0
    args.wandb = False
    args.validate = False
    args.val_check_interval = 1.0
    args.ckpt_freq = 1
    args.val_epoch_freq = 1
    args.epochs = 1
    args.lr = 1e-5
    args.freeze_encoder = True
    args.print_freq = 50

    model = PMPNNWrapper.load_from_checkpoint(
        cfg.checkpoint_path, args=args, strict=False
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(f"设备: {device}")

    # 预构建 codon→AA 索引表（放在 GPU 上，避免重复构建）
    codon_lookup = torch.full(
        (len(codon_types) + 1,), unk_restype_index, dtype=torch.long, device=device
    )
    for i, c in enumerate(codon_types):
        aa = codon_to_res.get(c, "X")
        codon_lookup[i] = restype_order_with_x.get(aa, unk_restype_index)

    # ── 加载数据 ───────────────────────────────────────────
    log.info("加载数据集...")
    dataset = AFDBDataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=seq_collate,
        shuffle=False,
    )
    log.info(f"数据集大小: {len(dataset)}")

    # ── 加载目标位点 CSV ───────────────────────────────────
    log.info("加载目标位点信息...")
    df_meta = pd.read_csv(cfg.data_csv)
    target_map: Dict[int, List[int]] = {}
    for idx, row in df_meta.iterrows():
        raw = str(row.get("ID", "")).strip()
        if raw and raw.lower() not in ("nan", "none", ""):
            try:
                target_map[idx] = [int(x.strip()) for x in raw.split(",")]
            except ValueError:
                log.warning(f"行 {idx} 的 ID 格式错误: {raw}")
                target_map[idx] = []
        else:
            target_map[idx] = []

    n_with_target = sum(1 for v in target_map.values() if v)
    log.info(f"含目标位点的样本: {n_with_target} / {len(target_map)}")

    # ── 主循环 ─────────────────────────────────────────────
    all_target_logits: Dict[int, Dict[int, np.ndarray]] = {}
    sample_counter = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting")):

            if sample_counter >= cfg.max_samples:
                break
            if batch is None:
                sample_counter += cfg.batch_size
                continue

            B = batch["mask"].shape[0]

            # 整批已完成 → 快速跳过
            if cfg.resume and all((sample_counter + b) in completed for b in range(B)):
                sample_counter += B
                continue

            # 数据迁移到 GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            mask   = batch["mask"]
            atom37 = batch["atom37"]
            seq    = batch["seq"]     # [B, L] residue indices
            codons = batch["codons"]  # [B, L] codon indices

            # ── 构建 position_mask ─────────────────────────
            position_mask = build_position_mask(
                mask, target_map, sample_counter, cfg.mode, device
            )

            # ── 结构掩码 ───────────────────────────────────
            masked_atom37 = apply_structure_mask(atom37, position_mask, cfg.mask_mode)
            # 取 N, CA, C, CB（4原子主链+CB）
            bb_pos = torch.cat(
                [masked_atom37[:, :, :3, :], masked_atom37[:, :, 4:5, :]], dim=2
            )

            # ── 模型推理 ───────────────────────────────────
            pred_dict = model.model.sample(
                X=bb_pos,
                mask=mask,
                chain_M=position_mask,
                residue_idx=batch["pmpnn_res_idx"],
                chain_encoding_all=batch["pmpnn_chain_encoding"],
                taxon_id=batch["taxon_id"],
                temperature=cfg.sampling_temp,
            )

            # logits / probs shape: [B, L, K]
            probs         = pred_dict["probs"]
            logits_tensor = pred_dict.get("logits", probs)

            # ── 预测密码子 & 氨基酸 ────────────────────────
            if cfg.train_aa:
                pred_res    = probs.argmax(-1)
                # AA → 伪密码子（取该 AA 最常用密码子，仅用于对比）
                pred_codons_batch = torch.stack([
                    torch.tensor(
                        [codon_order.get(
                            model.residues_to_protein_string([i])[0]
                            if i < len(restypes_with_x) else "NNN",
                            unk_codon_index,
                         ) for i in row],
                        device=device,
                    )
                    for row in pred_res
                ])
            else:
                pred_codons_batch = probs.argmax(-1)
                # ★ 向量化 codon→AA 转换（修复原始逐元素循环瓶颈）
                flat = pred_codons_batch.clamp(0, len(codon_types)).reshape(-1)
                pred_res = codon_lookup[flat].reshape(B, -1)

            # ── 最终序列：掩码位点用预测值，其余保留真实值 ─
            final_codons = torch.where(position_mask.bool(), pred_codons_batch, codons)
            final_res    = torch.where(position_mask.bool(), pred_res, seq)

            # ── 逐样本处理 ────────────────────────────────
            for b in range(B):
                global_idx = sample_counter + b
                if global_idx >= cfg.max_samples:
                    break
                if global_idx in completed:
                    continue

                valid_len = int(mask[b].sum().item())
                if valid_len == 0:
                    continue

                vfc = final_codons[b, :valid_len]
                vfr = final_res[b, :valid_len]
                vtc = codons[b, :valid_len]
                vtr = seq[b, :valid_len]
                ppc = pred_codons_batch[b, :valid_len]
                ppr = pred_res[b, :valid_len]

                pred_dna     = model.codons_to_dna_string(vfc)
                pred_protein = model.residues_to_protein_string(vfr)
                true_dna     = model.codons_to_dna_string(vtc)
                true_protein = model.residues_to_protein_string(vtr)

                # ── 位点级详情 ─────────────────────────────
                target_pos = target_map.get(global_idx, [])
                target_details, full_logits_list, sample_logits = [], [], {}
                codon_match_target = protein_match_target = None

                if target_pos:
                    target_details, full_logits_list, sample_logits = \
                        extract_position_details(
                            logits_tensor[b], ppc, ppr, vtc, vtr,
                            target_pos, valid_len, cfg.train_aa,
                        )

                    if sample_logits:
                        all_target_logits[global_idx] = sample_logits

                    valid_target = [p for p in target_pos if p < valid_len]
                    if valid_target:
                        t_mask = torch.zeros(valid_len, dtype=torch.bool, device=device)
                        for p in valid_target:
                            t_mask[p] = True
                        codon_match_target   = (ppc[t_mask] == vtc[t_mask]).float().mean().item()
                        protein_match_target = (ppr[t_mask] == vtr[t_mask]).float().mean().item()

                # ── af_id & taxon_id 安全提取 ──────────────
                af_ids    = batch.get("af_id")
                taxon_ids = batch.get("taxon_id")
                af_id_str    = af_ids[b] if af_ids is not None else f"sample{global_idx}"
                current_taxon = taxon_ids[b].item() if taxon_ids is not None else None

                row = {
                    "global_idx":              global_idx,
                    "af_id":                   af_id_str,
                    "length":                  valid_len,
                    "mode":                    cfg.mode,
                    "mask_mode":               cfg.mask_mode,
                    "taxon_id":                current_taxon,
                    "target_positions":        ",".join(map(str, target_pos)),
                    "num_target_positions":    len(target_pos),
                    "predicted_dna":           pred_dna,
                    "predicted_protein":       pred_protein,
                    "true_dna":                true_dna,
                    "true_protein":            true_protein,
                    "codon_match_all":         (vfc == vtc).float().mean().item(),
                    "protein_match_all":       (vfr == vtr).float().mean().item(),
                    "codon_match_target":      codon_match_target,
                    "protein_match_target":    protein_match_target,
                    "pred_gc":                 gc_content(pred_dna),
                    "true_gc":                 gc_content(true_dna),
                    "target_prediction_details": "; ".join(target_details),
                    "full_logits_json":        json.dumps(full_logits_list, ensure_ascii=False)
                                               if full_logits_list else "",
                }

                writer.writerow(row)
                csv_f.flush()

                # ── 异步写 FASTA ───────────────────────────
                if cfg.save_fasta:
                    fasta_path = _fasta_path(
                        fasta_dir, global_idx, af_id_str, cfg.fasta_bucket_size
                    )
                    executor.submit(_write_fasta, fasta_path, af_id_str,
                                    pred_protein, true_protein)

            sample_counter += B

    # ── 收尾 ───────────────────────────────────────────────
    log.info("等待 FASTA 写盘完成...")
    executor.shutdown(wait=True)
    csv_f.close()

    torch.save(all_target_logits, logits_path)
    log.info(f"Logits  → {logits_path}")
    log.info(f"CSV     → {csv_path}")
    if cfg.save_fasta:
        log.info(f"FASTA   → {fasta_dir}/0000/ ~ {fasta_dir}/{cfg.fasta_num_buckets-1:04d}/")

    # ── 打印统计摘要 ──────────────────────────────────────
    _print_summary(csv_path)
    return csv_path, logits_path


# ─────────────────────────────────────────────────────────────────────────────
# 统计摘要
# ─────────────────────────────────────────────────────────────────────────────
def _print_summary(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        n = len(df)
        log.info("=" * 60)
        log.info(f"预测样本总数          : {n}")
        log.info(f"平均序列长度           : {df['length'].mean():.1f}")
        log.info(f"全序列密码子准确率     : {df['codon_match_all'].mean():.4f}")
        log.info(f"全序列蛋白质准确率     : {df['protein_match_all'].mean():.4f}")
        has_target = df["codon_match_target"].notna()
        if has_target.sum() > 0:
            log.info(f"目标位点密码子准确率   : {df.loc[has_target,'codon_match_target'].mean():.4f}")
            log.info(f"目标位点蛋白质准确率   : {df.loc[has_target,'protein_match_target'].mean():.4f}")
        log.info(f"平均预测 GC 含量       : {df['pred_gc'].mean():.4f}")
        log.info(f"平均真实 GC 含量       : {df['true_gc'].mean():.4f}")
        log.info("=" * 60)
    except Exception as e:
        log.warning(f"统计摘要生成失败: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── 示例 1：全序列恢复 ─────────────────────────────────
    cfg_full = PredictConfig(
        checkpoint_path="/data/xr/CodonMPNN/workdir2/default/epoch=33-step=397698.ckpt",
        data_csv="/data/xr/CodonMPNN/data/reference/your_data.csv",
        afdb_dir="/data/xr/CodonMPNN/shiyan/shiyan/",
        output_dir="./predict_outputs/full_recovery",
        mode="full",           # ← 全序列恢复
        mask_mode="sidechain",
        sampling_temp=0.1,
        batch_size=32,
        save_full_logits=True,
        resume=True,
    )

    # ── 示例 2：指定位点预测 ───────────────────────────────
    cfg_masked = PredictConfig(
        checkpoint_path="/data/xr/CodonMPNN/workdir2/default/epoch=33-step=397698.ckpt",
        data_csv="/data/xr/CodonMPNN/data/reference/P02144.csv",  # 含 ID 列
        afdb_dir="/data/xr/CodonMPNN/shiyan/shiyan/",
        output_dir="./predict_outputs/masked_design",
        mode="masked",         # ← 指定位点预测
        mask_mode="sidechain",
        sampling_temp=0.1,
        batch_size=32,
        save_full_logits=True,
        resume=True,
    )

    # 选择运行哪个
    run_prediction(cfg_masked)
