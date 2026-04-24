# ThermoFlow

**ThermoFlow** is a deep learning framework designed for **structure-based codon optimization and protein sequence design**. By integrating geometric deep learning (ProteinMPNN) with large-scale protein language models (ESM2), ThermoFlow enables high-fidelity DNA sequence generation that is both structurally informed and host-adapted.

## 🌟 Key Features

- **Structure-Aware Codon Design**: Utilizes 3D atomic coordinates (Atom37) to guide codon selection.
- **ESM2 Feedback Integration**: Leverages protein language model embeddings to refine sequence generation.
- **Taxon-Specific Sampling**: Supports conditioned generation based on Taxon IDs to respect host-specific codon usage bias.
- **Flexible Masking Modes**: 
  - `sidechain`: Masks sidechain atoms while preserving backbone geometry.
  - `backbone_noise`: Adds Gaussian noise to backbone coordinates for robust design.
  - `all`: Full structure masking for de novo design tasks.
- **High-Efficiency Pipeline**: Features asynchronous FASTA writing and bucketed directory structures to handle datasets exceeding $10^5$ samples.

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Ran-Xu1213/ThermoFlow.git](https://github.com/Ran-Xu1213/ThermoFlow.git)
   cd ThermoFlow
