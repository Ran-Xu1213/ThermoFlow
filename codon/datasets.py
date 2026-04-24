import os
import glob
import sqlite3
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openfold.np.residue_constants import (
    restype_order_with_x,
    restypes_with_x,
    aatype_to_str_sequence,
)
from torch.utils.data import default_collate
from tqdm import tqdm

from codon.utils.codon_const import (
    unk_codon,
    codon_order,
    codon_to_res,
    unk_codon_index,
)
from codon.utils.data_utils import parse_mmcif, parse_pdb
from codon.utils.pmpnn import get_weird_pmpnn_stuff


def _resolve_pdb_path(original_path, afdb_dir):
    basename = os.path.basename(original_path)
    name_no_ext = os.path.splitext(basename)[0]

    candidates = []
    if afdb_dir:
        candidates.append(os.path.join(afdb_dir, basename))
        candidates.append(os.path.join(afdb_dir, f"{name_no_ext}-F1-model.pdb"))
        candidates.extend(glob.glob(os.path.join(afdb_dir, f"{name_no_ext}*.pdb")))
    candidates.append(original_path)

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"找不到PDB文件，尝试过: {candidates}")


class AFDBDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        df = pd.read_csv(args.data_csv)
        df = df[(df["dna_sequence"].str.len() / 3) < args.max_seq_len]
        if args.high_plddt:
            df_high = pd.read_csv(
                "/data/xr/CodonMPNN/shiyan/sample_single_point_grouped.csv"
            )
            df = df[
                df["Entry"].isin(df_high["entryId"].apply(lambda x: x.split("-")[1]))
            ]
        self.df = df
        self.args = args
        self.afdb_dir = getattr(args, "afdb_dir", None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.args.overfit:
            idx = 0
        df_row = self.df.iloc[idx]
        af_id = df_row["Entry"]
        taxon_id = df_row[f"{self.args.num_taxon_ids}_grouping"]
        dna_seq = df_row["dna_sequence"]

        fpath = df_row["pdb_path"]
        base_name = os.path.splitext(fpath.split("/")[-1])[0]
        cif_path = os.path.join(self.args.afdb_dir, base_name + ".pdb")

        # ── 文件缺失直接跳过，不崩溃 ──────────────────────────
        try:
            cif_path = _resolve_pdb_path(cif_path, self.afdb_dir)
        except FileNotFoundError as e:
            print(f"[SKIP] {af_id}: {e}")
            return None

        if len(dna_seq) % 3 != 0:
            print(f"Skipping {af_id}: len(dna_seq)={len(dna_seq)} % 3 != 0")
            return None

        prots = parse_pdb(cif_path)
        prot = prots[0]

        codon_seq = [dna_seq[i: i + 3] for i in range(0, len(dna_seq), 3)]
        protein_length = len(prot["seq"])

        if len(codon_seq) == protein_length + 1:
            codon_seq = codon_seq[:-1]
        elif len(codon_seq) != protein_length:
            min_length = min(len(codon_seq), protein_length)
            codon_seq = codon_seq[:min_length]
            if len(prot["seq"]) > min_length:
                prot["seq"] = prot["seq"][:min_length]
                prot["atom37"] = prot["atom37"][:min_length]
                prot["atom_mask"] = prot["atom_mask"][:min_length]

        codons = np.array([codon_order.get(a, unk_codon_index) for a in codon_seq], dtype=np.int32)

        if len(codons) != len(prot["seq"]):
            print(f"Skipping {af_id}: len(codons)={len(codons)} != len(prot)={len(prot['seq'])}")
            return None

        seq_str = aatype_to_str_sequence(prot["seq"])
        codon_str = "".join([codon_to_res.get(c, codon_to_res[unk_codon]) for c in codon_seq])

        if seq_str != codon_str:
            mask = np.array(list(seq_str)) == np.array(list(codon_str))
            prot["seq"] = prot["seq"][mask]
            prot["atom37"] = prot["atom37"][mask]
            prot["atom_mask"] = prot["atom_mask"][mask]
            codons = codons[mask]
            if mask.sum() < 0.8 * len(codon_str):
                print(f"Skipping {af_id}: <80% codon/prot match")
                return None

        bb_mask = prot["atom_mask"][:, :3].all(-1)
        prot["atom37"] = torch.from_numpy(prot["atom37"][bb_mask]).float()
        prot["atom_mask"] = torch.from_numpy(prot["atom_mask"][bb_mask]).long()
        prot["seq"] = torch.from_numpy(prot["seq"][bb_mask]).long()
        codons = torch.from_numpy(codons[bb_mask]).long()

        residue_idx, chain_encoding = get_weird_pmpnn_stuff(
            chain_idx=torch.zeros(len(codons), dtype=torch.long)
        )
        prot.update({
            "codons": codons,
            "taxon_id": taxon_id,
            "af_id": af_id,
            "pmpnn_res_idx": residue_idx,
            "pmpnn_chain_encoding": chain_encoding,
        })
        return prot


class Shen2022Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        df = pd.read_csv(args.data_csv)
        df = df[(df["wildtype_seq"].str.len() / 3) < args.max_seq_len]
        df = df[(df["mut_seq"].str.len() / 3) < args.max_seq_len]
        self.df = df
        self.args = args

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.args.overfit:
            idx = 0
        df_row = self.df.iloc[idx]
        taxon_id = df_row["taxon_id"]
        wildtype_seq = df_row["wildtype_seq"]
        mut_seq = df_row["mut_seq"]
        gene = df_row["gene"]
        mut_position = df_row["position"]
        cif_path = os.path.join(
            f"/data/scratch/diaoc/codon/data/shen2022_structs/{gene}/pdb/best.cif"
        )

        if len(wildtype_seq) % 3 != 0 or len(mut_seq) % 3 != 0:
            print("Skipping: dna_seq length % 3 != 0")
            return self.__getitem__(np.random.randint(len(self.df) - 1))

        prots = parse_mmcif(cif_path)
        prot = prots[0]

        def _to_codons(seq):
            codon_seq = [seq[i: i + 3] for i in range(0, len(seq), 3)][:-1]
            return codon_seq, np.array([codon_order.get(a, unk_codon_index) for a in codon_seq], dtype=np.int32)

        wt_codon_seq, wt_codons = _to_codons(wildtype_seq)
        mut_codon_seq, mut_codons = _to_codons(mut_seq)

        for name, codons in [("wildtype", wt_codons), ("mut", mut_codons)]:
            if len(codons) != len(prot["seq"]):
                print(f"Skipping: len({name}_codons)={len(codons)} != len(prot)={len(prot['seq'])}")
                return self.__getitem__(np.random.randint(len(self.df) - 1))

        seq_str = aatype_to_str_sequence(prot["seq"])
        wt_codon_str = "".join([codon_to_res.get(c, codon_to_res[unk_codon]) for c in wt_codon_seq])

        if seq_str != wt_codon_str:
            mask = np.array(list(seq_str)) == np.array(list(wt_codon_str))
            prot["seq"] = prot["seq"][mask]
            prot["atom37"] = prot["atom37"][mask]
            prot["atom_mask"] = prot["atom_mask"][mask]
            wt_codons = wt_codons[mask]
            mut_codons = mut_codons[mask]
            if mask.sum() < 0.8 * len(wt_codon_str):
                print("Skipping: <80% codon/prot match")
                return self.__getitem__(np.random.randint(len(self.df) - 1))

        bb_mask = prot["atom_mask"][:, :3].all(-1)
        prot["atom37"] = torch.from_numpy(prot["atom37"][bb_mask]).float()
        prot["atom_mask"] = torch.from_numpy(prot["atom_mask"][bb_mask]).long()
        prot["seq"] = torch.from_numpy(prot["seq"][bb_mask]).long()
        wt_codons = torch.from_numpy(wt_codons[bb_mask]).long()
        mut_codons = torch.from_numpy(mut_codons[bb_mask]).long()

        residue_idx, chain_encoding = get_weird_pmpnn_stuff(
            chain_idx=torch.zeros(len(wt_codons), dtype=torch.long)
        )
        prot.update({
            "wildtype_codons": wt_codons,
            "mut_codons": mut_codons,
            "mut_position": mut_position,
            "taxon_id": taxon_id,
            "pmpnn_res_idx": residue_idx,
            "pmpnn_chain_encoding": chain_encoding,
        })
        return prot


def seq_collate(batch):
    # ── 过滤文件缺失的样本 ─────────────────────────────────────
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    import pandas as pd

    seq_len_keys = [
        "atom37", "seq", "atom_mask", "codons",
        "pmpnn_res_idx", "pmpnn_chain_encoding",
    ]
    STRING_FIELDS = {
        "af_id", "gene", "species_name", "protein_name", "chain_id",
        "Entry", "wp_id", "uniprot_entry_name", "organism_uniprot",
        "gene_names", "mapping_status", "pdb_path",
        "seqid_geneid_chromosomeid", "5utr_sequence", "dna_sequence", "Sequence",
    }

    max_L = max(len(item[seq_len_keys[0]]) for item in batch)

    seq_len_batch = {}
    mask = torch.zeros((len(batch), max_L), dtype=torch.int16)
    for key in seq_len_keys:
        elem_tensor = []
        for i, item in enumerate(batch):
            elem = item[key]
            L = len(elem)
            mask[i, :L] = 1
            elem = torch.cat(
                [elem, torch.zeros(max_L - L, *elem.shape[1:], dtype=elem.dtype)], dim=0
            )
            elem_tensor.append(elem)
        seq_len_batch[key] = torch.stack(elem_tensor, dim=0)
    seq_len_batch["mask"] = mask

    result = {}
    for key in set(batch[0].keys()) - set(seq_len_keys):
        values = [item[key] for item in batch]
        if key in STRING_FIELDS:
            result[key] = values
            continue

        cleaned = []
        for v in values:
            if isinstance(v, str):
                v = v.strip().strip('"').strip("'")
                if v.lower() in ("nan", "none", "null", "", "n/a"):
                    cleaned.append(0)
                else:
                    try:
                        cleaned.append(int(float(v)))
                    except (ValueError, TypeError):
                        cleaned.append(0)
            elif v is None or (isinstance(v, float) and pd.isna(v)):
                cleaned.append(0)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                cleaned.append(int(v) if isinstance(v, (int, np.integer)) else v)
            else:
                try:
                    cleaned.append(int(v))
                except Exception:
                    cleaned.append(0)
        try:
            result[key] = torch.tensor(cleaned, dtype=torch.long)
        except Exception:
            result[key] = torch.zeros(len(batch), dtype=torch.long)

    result.update(seq_len_batch)
    return result


def multi_seq_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    seq_len_keys = [
        "atom37", "seq", "atom_mask", "wildtype_codons",
        "mut_codons", "pmpnn_res_idx", "pmpnn_chain_encoding",
    ]
    max_L = max(len(item[seq_len_keys[0]]) for item in batch)

    seq_len_batch = {}
    mask = torch.zeros((len(batch), max_L), dtype=torch.int16)
    for key in seq_len_keys:
        elem_tensor = []
        for i, item in enumerate(batch):
            elem = item[key]
            L = len(elem)
            mask[i, :L] = 1
            elem = torch.cat(
                [elem, torch.zeros(max_L - L, *elem.shape[1:], dtype=elem.dtype)], dim=0
            )
            elem_tensor.append(elem)
        seq_len_batch[key] = torch.stack(elem_tensor, dim=0)
    seq_len_batch["mask"] = mask

    for item in batch:
        for key in seq_len_keys:
            del item[key]
    batch = default_collate(batch)
    batch.update(seq_len_batch)
    return batch


class CodonSqliteDataset(torch.utils.data.Dataset):
    def __init__(self, db_path: Union[str, Path], ids=[]):
        self.db_path = db_path
        self.ids = ids
        self.conn = sqlite3.connect(db_path, isolation_level="DEFERRED")
        self.cursor = self.conn.cursor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        (emblcds_id, Entry) = self.ids[idx]
        self.cursor.execute(
            "SELECT nt, aa, tax_id FROM dataset WHERE emblcds = ? AND uniprot = ?",
            (emblcds_id, Entry),
        )
        nt, aa, tax_id = self.cursor.fetchone()
        return nt, aa, tax_id