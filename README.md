# ThermoFlow

**ThermoFlow** is a deep learning framework for **structure-based codon optimization and protein sequence design**. By integrating geometric deep learning (ProteinMPNN) with large-scale protein language models (ESM2), ThermoFlow enables the generation of DNA sequences that are both structurally informed and host-adapted.

## 🚀 Core Architecture: Dual-Model Support

ThermoFlow provides two specialized pre-trained models to meet different research and engineering requirements:

1.  **Thermo-Predictive Model**: 
    * **Focus**: Trained on comprehensive thermophilic protein datasets.
    * **Application**: Specifically optimized for increasing protein stability in high-temperature environments and designing sequences adapted to thermophilic microbial hosts.
2.  **Fine-tuned Model**: 
    * **Focus**: Deeply fine-tuned on specific species or functional datasets.
    * **Application**: Offers higher prediction accuracy for general high-expression codon optimization and precise sequence reconstruction.

---

## 🎨 Dual Design Modes

ThermoFlow supports flexible workflows ranging from local optimization to complete *de novo* design:

### 1. Site-Specific Design (Targeted Masking)
* **Use Case**: Modification of enzyme active sites or preservation of specific epitopes.
* **Functionality**: Users can specify one or more residue positions for masking. The model keeps the rest of the structure fixed and samples the optimal codons specifically for the target sites.

### 2. Global Sequence Design (Full Reconstruction)
* **Use Case**: Whole-gene codon redesign or *de novo* protein-DNA pair design.
* **Functionality**: Performs a full-sequence redesign based on the protein backbone. It generates entirely new DNA/protein sequence pairs while maintaining the original structural topology.

---

## 🌟 Key Features

-   **Structure-Aware Design**: Utilizes 3D atomic coordinates (Atom37) to guide codon selection rather than relying solely on 1D sequences.
-   **ESM2 Feedback Integration**: Leverages ESM2 embeddings to refine the semantic accuracy of generated sequences.
-   **Taxon-Specific Sampling**: Supports conditioned generation based on Taxon IDs to respect host-specific Codon Usage Bias (CUB).
-   **Flexible Masking Protocols**: Supports multiple modes including `sidechain` (masking sidechain atoms), `backbone_noise` (adding Gaussian noise to coordinates), and `all` (complete sequence masking).

---

## 🛠 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Ran-Xu1213/ThermoFlow.git
    cd ThermoFlow
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements_extracted.txt
    ```
    *Note: Ensure your CUDA version is compatible with `torch` and `openfold`.*

---

## 📖 Usage Guide

### Running Inference
You can switch between models or adjust design modes by modifying the `FinetuneArgs` class within the script:
```bash
python predict_flow_codonflow_esm.py
```

### Configuration Parameters
| Parameter | Options | Description |
| :--- | :--- | :--- |
| `pretrained_ckpt` | `thermo_path` / `ft_path` | Switch between Thermophilic or Fine-tuned models |
| `mask_mode` | `sidechain` / `all` | Switch between Site-Specific or Global design |
| `sampling_temp` | $0.1 \sim 1.0$ | Controls sampling diversity and randomness |
| `taxon_id` | e.g., `4932` (Yeast) | Specify the target host organism |

---

## 📂 Project Structure

-   `codon/`: Core model logic, flow wrappers, and codon-to-residue mappings.
-   `openfold/`: Structural bioinformatics utilities and residue constants.
-   `predict_flow_codonflow_esm.py`: Main execution script with ESM2 feedback integration.
-   `finetune_outputs/`: Storage for generated CSV statistics, log files, and FASTA buckets.

## ✉️ Contact
Author: **Ran-Xu1213** Project Link: [https://github.com/Ran-Xu1213/ThermoFlow](https://github.com/Ran-Xu1213/ThermoFlow)

---
*ThermoFlow is an ongoing research project dedicated to the intersection of thermostable protein engineering and synthetic biology.*
