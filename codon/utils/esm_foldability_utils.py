import os
import torch
import esm
import numpy as np
from openfold.np.residue_constants import restypes
from tmtools import tm_align
from tqdm import tqdm

def rigid_transform_3D(A, B, verbose=False):
    """Transforms A to look like B using Kabsch algorithm"""
    assert A.shape == B.shape
    A = A.T
    B = B.T
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    
    centroid_A = np.mean(A, axis=1).reshape(-1, 1)
    centroid_B = np.mean(B, axis=1).reshape(-1, 1)
    
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        reflection_detected = True
    
    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t
    return optimal_A.T, R, t, reflection_detected


def get_aligned_rmsd(pos_1, pos_2):
    """Calculate RMSD after optimal alignment"""
    aligned_pos_1 = rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1))


def get_tm_score(pos_1, pos_2, seq_1, seq_2):
    """Calculate TM-score between two structures"""
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def convert_seq_to_string(pred_seq):
    """将预测序列转换为字符串"""
    if isinstance(pred_seq, str):
        return pred_seq
    
    # 如果是 torch Tensor
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.cpu().numpy()
    
    # 如果是 numpy array 索引
    if isinstance(pred_seq, np.ndarray):
        if pred_seq.dtype in [np.int32, np.int64, np.int16, np.int8]:
            # 氨基酸索引转字母
            seq_str = ''.join([restypes[int(idx)] for idx in pred_seq if 0 <= int(idx) < len(restypes)])
            return seq_str
    
    return str(pred_seq)


def run_foldability(atom37s, pred_seqs, device):
    """
    使用 ESMFold 评估预测序列的可折叠性
    
    Args:
        atom37s: List of ground truth CA atom coordinates [N, L, 3]
        pred_seqs: List of predicted amino acid sequences
        device: torch device
    
    Returns:
        dict: {'tm_score': [...], 'rmsd': [...], 'plddt': [...]}
    """
    # 初始化结果字典
    results = {
        'tm_score': [],
        'rmsd': [],
        'plddt': []
    }
    
    # 输入验证
    if pred_seqs is None or len(pred_seqs) == 0:
        print("⚠️  No sequences provided for foldability evaluation")
        return results
    
    # 设置缓存目录
    torch_cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    os.makedirs(torch_cache, exist_ok=True)
    
    old_torch_home = os.environ.get('TORCH_HOME', None)
    os.environ['TORCH_HOME'] = os.path.expanduser("~/.cache/torch")
    
    esmf_model = None
    
    try:
        print(f"Loading ESMFold model for {len(pred_seqs)} sequences...")
        esmf_model = esm.pretrained.esmfold_v1().eval().to(device)
        print("✓ ESMFold loaded successfully")
        
        # 使用 tqdm 显示进度
        with torch.no_grad():
            for i, pred_seq in enumerate(tqdm(pred_seqs, desc="Evaluating foldability")):
                try:
                    # 转换为字符串序列
                    seq_str = convert_seq_to_string(pred_seq)
                    
                    # 验证序列
                    if not seq_str or len(seq_str) == 0:
                        print(f" Empty sequence {i} after conversion")
                        results['tm_score'].append(0.0)
                        results['rmsd'].append(0.0)
                        results['plddt'].append(0.0)
                        continue
                    
                    # ESMFold 预测结构 - 使用转换后的字符串
                    output = esmf_model.infer(seq_str)
                    
                    # 提取 pLDDT (置信度分数)
                    plddt = output['plddt'].mean().item() if 'plddt' in output else 0.0
                    
                    # 提取预测的 CA 原子坐标
                    pred_coords = output['positions'][-1, 0, :, 1, :].cpu().numpy()  # [L, 3]
                    
                    # 如果有真实结构，计算 TM-score 和 RMSD
                    if atom37s is not None and i < len(atom37s):
                        true_coords = atom37s[i]
                        
                        # 如果是 Tensor，转换为 numpy
                        if isinstance(true_coords, torch.Tensor):
                            true_coords = true_coords.cpu().numpy()
                        
                        # 确保坐标长度匹配
                        min_len = min(len(true_coords), len(pred_coords))
                        if min_len < 5:
                            results['tm_score'].append(0.0)
                            results['rmsd'].append(0.0)
                            results['plddt'].append(plddt)
                            continue
                        
                        true_coords = true_coords[:min_len]
                        pred_coords = pred_coords[:min_len]
                        
                        # 计算 TM-score
                        try:
                            tm_score_1, tm_score_2 = get_tm_score(
                                pred_coords, true_coords, 
                                seq_str[:min_len], seq_str[:min_len]
                            )
                            tm_score = (tm_score_1 + tm_score_2) / 2.0
                        except Exception as e:
                            tm_score = 0.0
                        
                        # 计算 RMSD
                        try:
                            rmsd = get_aligned_rmsd(pred_coords, true_coords)
                        except Exception as e:
                            rmsd = 0.0
                    else:
                        tm_score = 0.0
                        rmsd = 0.0
                    
                    results['tm_score'].append(tm_score)
                    results['rmsd'].append(rmsd)
                    results['plddt'].append(plddt)
                    
                except Exception as e:
                    print(f"  ❌ Error processing sequence {i}: {str(e)[:100]}")
                    
                    # 添加默认值
                    results['tm_score'].append(0.0)
                    results['rmsd'].append(0.0)
                    results['plddt'].append(0.0)
        
        # 打印统计信息
        valid_results = [r for r in results['plddt'] if r > 0]
        if valid_results:
            print(f"\n✓ Foldability evaluation completed:")
            print(f"  - Total sequences: {len(pred_seqs)}")
            print(f"  - Successfully evaluated: {len(valid_results)}")
            print(f"  - Mean pLDDT: {np.mean(valid_results):.2f}")
            
            valid_tm = [r for r in results['tm_score'] if r > 0]
            if valid_tm:
                print(f"  - Mean TM-score: {np.mean(valid_tm):.4f}")
            
            valid_rmsd = [r for r in results['rmsd'] if r > 0]
            if valid_rmsd:
                print(f"  - Mean RMSD: {np.mean(valid_rmsd):.2f} Å")
        
    except Exception as e:
        print(f"❌ ESMFold evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回占位符结果
        if len(results['plddt']) < len(pred_seqs):
            missing = len(pred_seqs) - len(results['plddt'])
            results['tm_score'].extend([0.0] * missing)
            results['rmsd'].extend([0.0] * missing)
            results['plddt'].extend([0.0] * missing)
    
    finally:
        # 清理模型释放显存
        if esmf_model is not None:
            del esmf_model
            torch.cuda.empty_cache()
        
        # 恢复环境变量
        if old_torch_home is not None:
            os.environ['TORCH_HOME'] = old_torch_home
        elif 'TORCH_HOME' in os.environ:
            del os.environ['TORCH_HOME']
    
    return results