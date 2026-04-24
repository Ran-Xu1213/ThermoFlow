import os
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from openfold.np.residue_constants import restypes_with_x, restype_order_with_x, unk_restype_index

from codon.utils.parsing import parse_train_args
from codon.datasets import AFDBDataset, seq_collate
from codon.flow_wrapper import PMPNNWrapper
from codon.utils.codon_const import codon_order, codon_types, codon_to_res, unk_codon_index

# ── 输出目录 ───────────────────────────────────────────────────────────────────
output_dir = "./finetune_outputs/Codonflow_esm2/"
os.makedirs(output_dir, exist_ok=True)
os.environ["MODEL_DIR"] = output_dir

CHECKPOINT_PATH = "/data/xr/CodonMPNN/workdir2/default/epoch=33-step=397698.ckpt"
OUTPUT_BASE = "./finetune_outputs/Codonflow_esm2/"


class FinetuneArgs:
    pretrained_ckpt = CHECKPOINT_PATH
    afdb_dir = "/data/xr/CodonMPNN/shiyan/shiyan/"
    workdir = "./data/reference/workdir_finetune"
    run_name = "masked_design"
    epochs = 100
    batch_size = 256
    lr = 1e-5
    freeze_encoder = True
    max_seq_len = 750
    hidden_dim = 128
    num_encoder_layers = 3
    num_decoder_layers = 3
    num_neighbors = 48
    dropout = 0.1
    train_aa = False
    taxon_condition = True
    num_taxon_ids = 1000
    sampling_temp = 0.1
    num_workers = 0
    grad_clip = 1.0
    print_freq = 50
    val_check_interval = 1.0
    num_foldability_batches = 2
    high_plddt = False
    overfit = False
    validate = False
    accumulate_grad = 1
    wandb = False
    ckpt_freq = 1
    val_epoch_freq = 1
    use_transformer = False
    use_esm2_feedback = False


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def calculate_gc_content(dna):
    if not dna:
        return 0.0
    return (dna.count("G") + dna.count("C")) / len(dna)


def calculate_nucleotide_distribution(dna):
    total = len(dna)
    if total == 0:
        return {"A": 0, "T": 0, "G": 0, "C": 0}
    c = Counter(dna)
    return {n: c.get(n, 0) / total for n in "ATGC"}


def apply_structure_mask(atom37, position_mask, mask_mode="sidechain"):
    """
    atom37:        [B, L, 37, 3]
    position_mask: [B, L]  (1=掩码位点)
    mask_mode:     'sidechain' | 'all' | 'backbone_noise'
    """
    masked = atom37.clone()
    pm = position_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]

    if mask_mode == "all":
        masked = masked * (1 - pm)

    elif mask_mode in ("sidechain", "backbone_noise"):
        sc_mask = torch.zeros(atom37.shape[2], device=atom37.device)
        sc_mask[4:] = 1                          # CB 及以后为侧链
        sc_mask = sc_mask.view(1, 1, -1, 1)
        masked = masked * (1 - pm * sc_mask)     # 侧链清零

        if mask_mode == "backbone_noise":
            noise = torch.randn_like(atom37[:, :, :4, :]) * 0.1
            masked[:, :, :4, :] += noise * pm[:, :, :4, :]

    return masked


# ── FASTA 异步写盘 ─────────────────────────────────────────────────────────────

def _write_fasta(fasta_file, af_id_str, pred_protein, true_protein):
    """在线程池中执行，不阻塞主进程"""
    with open(fasta_file, "w") as ff:
        ff.write(f">{af_id_str}|pred\n{pred_protein}\n")
        ff.write(f">{af_id_str}|true\n{true_protein}\n")


def _get_fasta_path(fasta_dir, global_idx, af_id_str):
    """分桶：每1000个样本一个子目录，避免单目录文件过多"""
    bucket = global_idx // 1000
    bucket_dir = os.path.join(fasta_dir, f"{bucket:04d}")
    return os.path.join(bucket_dir, f"{af_id_str}.fasta")


# ── CSV 字段 ───────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "global_idx", "af_id", "length", "mask_mode",
    "target_positions", "num_target_positions", "taxon_id",
    "predicted_dna", "predicted_protein", "true_dna", "true_protein",
    "codon_match_all", "protein_match_all",
    "codon_match_target", "protein_match_target",
    "pred_gc", "true_gc",
    "target_prediction_details", "full_logits_json",
]

MAX_SAMPLES = 1000000000000000
# ── 主预测函数 ─────────────────────────────────────────────────────────────────

def predict_with_masking(mask_mode="sidechain"):
    csv_path    = os.path.join(OUTPUT_BASE, f"codonmpnn_{mask_mode}.csv")
    logits_path = os.path.join(OUTPUT_BASE, f"codonmpnn_logits_{mask_mode}.pt")
    fasta_dir   = os.path.join(OUTPUT_BASE, "fasta")

    # ── 预建目录结构（300个桶，覆盖30万样本）─────────────────
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    print("预建 fasta 分桶目录...")
    for i in range(300):
        os.makedirs(os.path.join(fasta_dir, f"{i:04d}"), exist_ok=True)
    print("目录建立完成")

    # ── 断点续跑：读取已完成 global_idx ───────────────────────
    completed_indices = set()
    if os.path.exists(csv_path):
        try:
            completed_indices = set(
                pd.read_csv(csv_path, usecols=["global_idx"])["global_idx"].tolist()
            )
            print(f"[RESUME] 已跳过 {len(completed_indices)} 条已完成样本")
        except Exception:
            pass

    csv_is_new = not os.path.exists(csv_path)
    csv_f      = open(csv_path, "a", buffering=1, newline="")
    csv_writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if csv_is_new:
        csv_writer.writeheader()
        csv_f.flush()

    # ── 线程池：异步写 FASTA，8线程与GPU推理并行 ──────────────
    executor = ThreadPoolExecutor(max_workers=8)

    # ── 加载模型 ───────────────────────────────────────────────
    print("Loading model...")
    args = parse_train_args()
    model = PMPNNWrapper.load_from_checkpoint(CHECKPOINT_PATH, args=args, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}")

    # ── 加载数据 ───────────────────────────────────────────────
    dataset = AFDBDataset(args)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0,
        collate_fn=seq_collate, shuffle=False,
    )

    # ── 加载目标位点 ───────────────────────────────────────────
    df_meta = pd.read_csv(args.data_csv)
    target_positions_map = {}
    for idx, row in df_meta.iterrows():
        raw = str(row.get("ID", "")).strip()
        if raw and raw.lower() not in ("nan", "none", ""):
            try:
                target_positions_map[idx] = [int(x.strip()) for x in raw.split(",")]
            except ValueError:
                target_positions_map[idx] = []
        else:
            target_positions_map[idx] = []

    all_target_logits = {}
    sample_counter = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Designing")):

            if sample_counter >= MAX_SAMPLES:
                break
            # 整批文件全缺失
            
            if batch is None:
                sample_counter += args.batch_size
                continue

            B = batch["mask"].shape[0]

            # 整批已完成，快速跳过
            if all((sample_counter + b) in completed_indices for b in range(B)):
                sample_counter += B
                continue

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            mask   = batch["mask"]
            atom37 = batch["atom37"]
            seq    = batch["seq"]
            codons = batch["codons"]

            # ── 构建 position_mask ─────────────────────────────
            position_mask = torch.zeros([B, atom37.shape[1]], device=device, dtype=torch.long)
            for b in range(B):
                global_idx = sample_counter + b
                if global_idx >= MAX_SAMPLES:        
                    break
                target_pos = target_positions_map.get(global_idx, [])
                valid_len  = mask[b].sum().item()
                if not target_pos:
                    position_mask[b] = mask[b]
                else:
                    for pos in target_pos:
                        if pos < valid_len:
                            position_mask[b, pos] = 1

            masked_atom37 = apply_structure_mask(atom37, position_mask, mask_mode)
            bb_pos = torch.cat([masked_atom37[:, :, :3, :], masked_atom37[:, :, 4:5, :]], dim=2)

            pred_dict = model.model.sample(
                X=bb_pos,
                mask=mask,
                chain_M=position_mask,
                residue_idx=batch["pmpnn_res_idx"],
                chain_encoding_all=batch["pmpnn_chain_encoding"],
                taxon_id=batch["taxon_id"],
                temperature=args.sampling_temp,
            )

            probs         = pred_dict["probs"]
            logits_tensor = pred_dict.get("logits", probs)

            if args.train_aa:
                pred_res    = probs.argmax(-1)
                pred_codons = torch.stack([
                    torch.tensor([
                        codon_order.get(
                            model.residues_to_protein_string([i])[0]
                            if i < len(restypes_with_x) else "NNN",
                            unk_codon_index,
                        ) for i in seq_e
                    ], device=device)
                    for seq_e in pred_res
                ])
            else:
                pred_codons = probs.argmax(-1)
                pred_res = torch.stack([
                    torch.tensor([
                        restype_order_with_x.get(
                            codon_to_res[codon_types[i.item()]]
                            if i.item() < len(codon_types) else "X",
                            unk_restype_index,
                        ) for i in seq_e
                    ], device=device)
                    for seq_e in pred_codons
                ])

            final_codons = torch.where(position_mask.bool(), pred_codons, codons)
            final_res    = torch.where(position_mask.bool(), pred_res, seq)

            # ── 逐样本写盘 ─────────────────────────────────────
            for b in range(B):
                global_idx = sample_counter + b
                if global_idx in completed_indices:
                    continue

                valid_len = mask[b].sum().item()
                if valid_len == 0:
                    continue

                vfc = final_codons[b, :valid_len]
                vfr = final_res[b, :valid_len]
                vtc = codons[b, :valid_len]
                vtr = seq[b, :valid_len]
                ppc = pred_codons[b, :valid_len]
                ppr = pred_res[b, :valid_len]

                pred_dna     = model.codons_to_dna_string(vfc)
                pred_protein = model.residues_to_protein_string(vfr)
                true_dna     = model.codons_to_dna_string(vtc)
                true_protein = model.residues_to_protein_string(vtr)

                target_pos           = target_positions_map.get(global_idx, [])
                target_codon_match   = None
                target_protein_match = None
                target_details       = []
                full_logits_list     = []
                sample_logits        = {}

                if target_pos:
                    t_mask = torch.zeros(valid_len, dtype=torch.bool, device=device)
                    for pos in target_pos:
                        if pos >= valid_len:
                            continue
                        t_mask[pos] = True

                        pc_idx  = ppc[pos].item()
                        tc_idx  = vtc[pos].item()
                        pr_idx  = ppr[pos].item()
                        tr_idx  = vtr[pos].item()
                        pc_str  = codon_types[pc_idx] if pc_idx < len(codon_types) else "NNN"
                        tc_str  = codon_types[tc_idx] if tc_idx < len(codon_types) else "NNN"
                        pred_aa = restypes_with_x[pr_idx] if pr_idx < len(restypes_with_x) else "X"
                        true_aa = restypes_with_x[tr_idx] if tr_idx < len(restypes_with_x) else "X"

                        pos_logits = logits_tensor[b, pos]
                        sample_logits[pos] = pos_logits.cpu().numpy()

                        topk_v, topk_i = torch.topk(pos_logits, min(5, pos_logits.shape[-1]))
                        if args.train_aa:
                            topk_tok = [restypes_with_x[i] if i < len(restypes_with_x) else "X"
                                        for i in topk_i.cpu().numpy()]
                        else:
                            topk_tok = [codon_types[i] if i < len(codon_types) else "NNN"
                                        for i in topk_i.cpu().numpy()]

                        topk_str = ",".join(f"{t}({v:.3f})" for t, v in zip(topk_tok, topk_v.cpu().numpy()))
                        target_details.append(
                            f"{pos}:{true_aa}({tc_str})->{pred_aa}({pc_str}) [Top5:{topk_str}]"
                        )

                        logits_arr = pos_logits.cpu().numpy()
                        if args.train_aa:
                            all_logits = {restypes_with_x[i] if i < len(restypes_with_x) else "X": float(v)
                                          for i, v in enumerate(logits_arr)}
                        else:
                            all_logits = {codon_types[i] if i < len(codon_types) else "NNN": float(v)
                                          for i, v in enumerate(logits_arr)}

                        full_logits_list.append({
                            "position": pos,
                            "true_aa":  true_aa,
                            "pred_aa":  pred_aa,
                            "all_logits": all_logits,
                        })

                    if sample_logits:
                        all_target_logits[global_idx] = sample_logits

                    if t_mask.sum() > 0:
                        target_codon_match   = (ppc[t_mask] == vtc[t_mask]).float().mean().item()
                        target_protein_match = (ppr[t_mask] == vtr[t_mask]).float().mean().item()

                #af_id_str = (batch.get("af_id") or [None])[b] or f"sample{global_idx}"
                #af_ids = batch.get("af_id")
                #af_id_str = af_ids[b] if af_ids is not None else f"sample{global_idx}"
                af_ids = batch.get("af_id")
                # 检查是否存在，存在则取第 b 个，不存在则使用默认命名
                af_id_str = af_ids[b] if af_ids is not None else f"sample{global_idx}"

                taxon_data = batch.get("taxon_id")
                # 关键修复：不要使用 or，直接判断是否存在。使用 .item() 确保是纯数字而非 Tensor
                current_taxon = taxon_data[b].item() if taxon_data is not None else None
                
                
                row = {
                    "global_idx":                global_idx,
                    "af_id":                     af_id_str,
                    "length":                    valid_len,
                    "mask_mode":                 mask_mode,
                    "target_positions":          ",".join(map(str, target_pos)),
                    "num_target_positions":      len(target_pos),
                    #"taxon_id":                  (batch.get("taxon_id") or [None])[b],
                    "taxon_id": current_taxon,
                    "predicted_dna":             pred_dna,
                    "predicted_protein":         pred_protein,
                    "true_dna":                  true_dna,
                    "true_protein":              true_protein,
                    "codon_match_all":           (vfc == vtc).float().mean().item(),
                    "protein_match_all":         (vfr == vtr).float().mean().item(),
                    "codon_match_target":        target_codon_match,
                    "protein_match_target":      target_protein_match,
                    "pred_gc":                   calculate_gc_content(pred_dna),
                    "true_gc":                   calculate_gc_content(true_dna),
                    "target_prediction_details": "; ".join(target_details),
                    "full_logits_json":          json.dumps(full_logits_list) if full_logits_list else "",
                }

                # ── 异步写 FASTA（线程池，不阻塞GPU）─────────────
                fasta_file = _get_fasta_path(fasta_dir, global_idx, af_id_str)
                executor.submit(_write_fasta, fasta_file, af_id_str, pred_protein, true_protein)

                # ── 同步写 CSV（每条 flush，保证断点续跑安全）────
                csv_writer.writerow(row)
                csv_f.flush()

            sample_counter += B

    # ── 收尾：等待所有 FASTA 写完 ──────────────────────────────
    print("等待所有 FASTA 文件写盘完成...")
    executor.shutdown(wait=True)

    csv_f.close()

    torch.save(all_target_logits, logits_path)
    print(f"Logits saved : {logits_path}")
    print(f"CSV          : {csv_path}")
    print(f"FASTA dir    : {fasta_dir}")
    print(f"  结构: {fasta_dir}/0000/ ~ {fasta_dir}/0299/  (每桶1000个文件)")


if __name__ == "__main__":
    predict_with_masking(mask_mode="sidechain")