from argparse import ArgumentParser
import subprocess, os

def parse_train_args():
    parser = ArgumentParser()
    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--validate", action='store_true', default=False)
    parser.add_argument("--num_workers", type=int, default=4)

    ## Epoch settings
    group = parser.add_argument_group("Epoch settings")
    group.add_argument("--epochs", type=int, default=10)
    group.add_argument("--overfit", action='store_true')
    group.add_argument("--train_batches", type=int, default=None)
    group.add_argument("--val_batches", type=int, default=None)
    group.add_argument("--batch_size", type=int, default=16)
    group.add_argument("--val_check_interval", type=int, default=1.0)
    group.add_argument("--val_epoch_freq", type=int, default=1)
    group.add_argument("--num_foldability_batches", type=int, default=3)
    group.add_argument("--train_aa", action='store_true')

    ## Logging args
    group = parser.add_argument_group("Logging settings")
    group.add_argument("--print_freq", type=int, default=100)
    group.add_argument("--ckpt_freq", type=int, default=1)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--run_name", type=str, default="default")
    group.add_argument("--workdir", type=str, default="workdir")

    ## Optimization settings
    group = parser.add_argument_group("Optimization settings")
    group.add_argument("--accumulate_grad", type=int, default=1)
    group.add_argument("--grad_clip", type=float, default=1.)
    group.add_argument("--lr", type=float, default=1e-3)

    ## Training data
    group = parser.add_argument_group("Training data settings")
    group.add_argument('--afdb_dir', type=str, default="./data/pdb_structures_from_tsv/")
    group.add_argument('--data_csv', type=str, default="./data/cluster_ini/processed_proteins_uppercase.csv")
    group.add_argument('--max_seq_len', type=int, default=750)
    group.add_argument('--num_taxon_ids', type=int, default=100)
    group.add_argument("--high_plddt", action="store_true")


    ## Model settings
    group = parser.add_argument_group("Model settings")
    group.add_argument('--hidden_dim', type=int, default=128)
    group.add_argument("--taxon_condition", action="store_true")
    group.add_argument('--num_encoder_layers', type=int, default=3)
    group.add_argument('--num_decoder_layers', type=int, default=3)
    group.add_argument('--num_neighbors', type=int, default=48)
    group.add_argument('--dropout', type=float, default=0.1)
    group.add_argument('--backbone_noise', type=float, default=0.02)

    # ========== 新增：Transformer相关参数 ==========
    group = parser.add_argument_group("Transformer settings")
    group.add_argument('--use_transformer', action='store_true', default=None,
                      help='启用Transformer增强')
    group.add_argument('--transformer_heads', type=int, default=8,
                      help='Transformer注意力头数')
    group.add_argument('--transformer_layers', type=int, default=2,
                      help='Transformer层数')
    group.add_argument('--transformer_fusion_weight', type=float, default=0.3,
                      help='Transformer特征融合权重')
    group.add_argument('--use_positional_encoding', action='store_true', default=True,
                      help='是否使用位置编码')
    group.add_argument('--codon_consistency_weight', type=float, default=0.1,
                      help='密码子一致性损失权重')
    group.add_argument('--transformer_start_epoch', type=int, default=10,
                      help='开始使用Transformer的epoch')

    parser.add_argument('--use_codon_preference', action='store_true',
                       help='Enable codon preference constraints')
    parser.add_argument('--codon_preference_weight', type=float, default=0.1,
                       help='Weight for codon preference bias')
    parser.add_argument('--preference_temperature', type=float, default=1.0,
                       help='Temperature for codon preference distribution')
                       
    ## Inference settings
    group = parser.add_argument_group("Inference settings")
    group.add_argument('--sampling_temp', type=float, default=0.1)

    # ========== 【新增】ESM2参数 ==========
    parser.add_argument('--use_esm2_feedback', action='store_true')
    parser.add_argument('--esm2_weight', type=float, default=0.1)
    parser.add_argument('--esm2_model', type=str, default='esm2_t30_150M_UR50D',
                       choices=['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 
                               'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D'])
    parser.add_argument('--esm2_start_epoch', type=int, default=1)
    parser.add_argument('--esm2_check_interval', type=int, default=4)
    parser.add_argument('--esm2_baseline_momentum', type=float, default=0.9)
  
    # ======================================
    
    args = parser.parse_args()
    os.environ["MODEL_DIR"] = os.path.join(args.workdir, args.run_name)
    os.makedirs(os.environ["MODEL_DIR"], exist_ok=True)
    os.environ["WANDB_LOGGING"] = str(int(args.wandb))
    os.environ["TORCH_HOME"] = "/data/rsg/nlp/hstark/torch_cache" if os.getcwd() != '/Users/hstark/projects/codon' else "/Users/hstark/projects/torch_cache"

    if args.wandb:
        if subprocess.check_output(["git", "status", "-s"]):
            print("There were uncommited changes. Commit before running")
            exit()
    args.commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )
    return args

