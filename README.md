# 🧬 GeneWeaver — GPU-Accelerated CRISPR Alignment Engine

<p align="center">
  <b>High-Performance Genomic Sequence Alignment Powered by CUDA, Numba & Dask</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/CUDA-GPU%20Accelerated-76B900?logo=nvidia" />
  <img src="https://img.shields.io/badge/Numba-JIT%20Compilation-orange" />
  <img src="https://img.shields.io/badge/Dask-Distributed%20Computing-purple" />
  <img src="https://img.shields.io/badge/BioPython-Genomics-green" />
  <img src="https://img.shields.io/badge/Interface-Textual-black" />
</p>

---

## 🚀 Overview

**GeneWeaver** is a GPU-accelerated CRISPR DNA sequence alignment engine designed to identify potential unintended genetic mutations, commonly known as *off-target effects*, in large genomic datasets.

Searching through the human genome, which contains approximately 3.2 billion base pairs, requires substantial computational resources. Traditional CPU-based sequence matching can become computationally expensive and time-consuming when processing massive genomic datasets.

GeneWeaver addresses this challenge by leveraging **NVIDIA CUDA, Numba JIT compilation, and distributed computing with Dask** to parallelize DNA sequence comparisons across available GPU resources.

The system provides a terminal-based monitoring dashboard that displays alignment progress, GPU resource utilization, and potential off-target locations.

**Our Goal:** Make large-scale genomic sequence analysis faster, more accessible, and computationally efficient through Python-based GPU acceleration.

## 🎯 Problem Statement

CRISPR-Cas9 is a powerful gene-editing technology, but it may sometimes modify DNA sequences at unintended genomic locations.

Identifying these unintended modifications requires searching massive DNA sequences against a target guide RNA sequence while considering mismatches and relevant PAM (Protospacer Adjacent Motif) regions.

Conventional sequential CPU-based approaches can become inefficient for large-scale genomic analysis.

GeneWeaver aims to solve this problem through:

- GPU-based parallel DNA sequence comparison.
- Efficient genomic data preprocessing.
- Distributed execution across multiple GPUs or machines.
- Real-time computational monitoring.
- Identification and reporting of potential CRISPR off-target sites.

## ✨ Key Features

- **GPU-Accelerated Alignment:** Executes parallel DNA sequence comparisons using custom CUDA kernels.
- **JIT Compilation:** Uses Numba to compile Python functions into GPU-executable kernels.
- **Large FASTA File Processing:** Reads and preprocesses genomic sequences using BioPython.
- **Distributed Computing:** Uses Dask to divide large genomic datasets into manageable computational chunks.
- **Off-Target Detection:** Identifies candidate genomic locations based on sequence mismatches and PAM requirements.
- **Live Terminal Dashboard:** Displays alignment progress and available GPU resource statistics.
- **Scalable Architecture:** Designed to support multiple GPUs and distributed computing environments.
- **Detailed Reports:** Generates genomic coordinates, mismatch information, and candidate off-target results.

## 🏗️ System Architecture

GeneWeaver follows a modular processing pipeline:

```text
        GENOMIC RESEARCHER
                |
                v
       Upload FASTA Dataset
                |
                v
       +------------------+
       | Genomic Parser   |
       |    BioPython     |
       +------------------+
                |
                v
       Sequence Cleaning
       & Preprocessing
                |
                v
       +------------------+
       | Dask Scheduler   |
       | Data Partitioning|
       +------------------+
                |
                v
       +----------------------+
       | CUDA Kernel Engine   |
       |   Numba JIT          |
       +----------------------+
                |
                v
       Parallel DNA Alignment
                |
                v
       PAM & Mismatch Filtering
                |
                v
       Off-Target Identification
                |
                v
       +------------------+
       | Results Engine   |
       +------------------+
                |
                v
       +------------------+
       | Textual TUI      |
       | Live Monitoring  |
       +------------------+
                |
                v
       Genomic Alignment Report
```

## 🧩 Core Modules

### 1. Genomic Parser — Python & BioPython

Responsible for loading and preprocessing genomic datasets.

**Responsibilities:**
- Import genomic sequences from `.fasta` files.
- Parse DNA sequences into computationally manageable chunks.
- Remove invalid characters and normalize nucleotide sequences.
- Prepare genomic data for GPU processing.

### 2. CUDA Kernel Engine — Numba

The computational core of GeneWeaver.

**Responsibilities:**
- Develop custom CUDA kernels using Python and Numba.
- Execute parallel nucleotide comparisons across GPU threads.
- Evaluate mismatches between target guide sequences and genomic regions.
- Reduce CPU processing overhead through GPU acceleration.

### 3. Distributed Scheduler — Dask

Manages computational workloads across available processing resources.

**Responsibilities:**
- Divide large genomic datasets into smaller chunks.
- Schedule independent alignment tasks.
- Coordinate multiple GPUs or distributed workers.
- Aggregate alignment results from completed tasks.

### 4. Terminal Dashboard — Textual

Provides an interactive terminal-based interface for monitoring genomic analysis.

**Dashboard information:**
- Current alignment progress.
- GPU memory consumption.
- GPU utilization, where supported.
- Processed genomic chunks.
- Number of candidate off-target matches.
- Execution time and processing statistics.

### 5. Results & Reporting Engine

Processes and organizes the detected candidate off-target locations.

**Output information:**
- Chromosome or contig identifier.
- Genomic start and end positions.
- Target DNA sequence.
- Candidate matching sequence.
- Number and positions of mismatches.
- PAM sequence information.
- Alignment score.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development language |
| BioPython | FASTA parsing and genomic preprocessing |
| Numba | JIT compilation and CUDA programming |
| NVIDIA CUDA | GPU parallel processing |
| Dask | Distributed task scheduling |
| Textual | Interactive terminal dashboard |
| NumPy | Numerical and sequence data operations |
| Git & GitHub | Version control and collaboration |

## 📂 Project Structure

```text
GeneWeaver/
│
├── data/
│   ├── input/
│   │   └── sample.fasta
│   └── output/
│       └── alignment_results.csv
│
├── src/
│   ├── parser/
│   │   └── genomic_parser.py
│   │
│   ├── engine/
│   │   ├── cuda_kernels.py
│   │   └── alignment.py
│   │
│   ├── scheduler/
│   │   └── distributed_scheduler.py
│   │
│   ├── dashboard/
│   │   └── terminal_ui.py
│   │
│   └── results/
│       └── report_generator.py
│
├── tests/
│   ├── test_parser.py
│   └── test_alignment.py
│
├── requirements.txt
├── config.py
├── main.py
├── LICENSE
└── README.md
```

## ⚙️ Installation & Setup

### Prerequisites

Before running GeneWeaver, ensure that you have:

- Python 3.10 or later.
- An NVIDIA GPU with CUDA support.
- Compatible NVIDIA GPU drivers.
- A compatible CUDA Toolkit and Numba CUDA environment.
- Git installed on your system.

### Step 1: Clone the Repository

```bash
git clone https://github.com/ProgrammerVrushabh/GeneWeaver.git

cd GeneWeaver
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install biopython numpy numba dask distributed textual
```

### Step 4: Verify GPU Availability

```python
from numba import cuda

print("CUDA Available:", cuda.is_available())

if cuda.is_available():
    print("GPU:", cuda.get_current_device().name)
```

### Step 5: Run GeneWeaver

```bash
python main.py --input data/input/sample.fasta
```

*Note: The command-line interface and module paths represent the planned project structure and should be updated to match the implemented version.*

## 💻 Example Usage

A typical GeneWeaver analysis follows these steps:

**Input:**
- Reference genome: `human_genome.fasta`
- CRISPR guide RNA: User-provided guide sequence.
- PAM requirement: For example, NGG for SpCas9.
- Maximum allowed mismatches: User-configurable threshold.

**Execution:**

```bash
python main.py \
    --input human_genome.fasta \
    --guide GAGTCCGAGCAGAAGAAGAA \
    --pam NGG \
    --mismatches 3
```

**Illustrative Output:**

```text
==================================================
              GENEWEAVER ENGINE
       GPU-ACCELERATED CRISPR ALIGNMENT
==================================================

[INFO] Loading genomic FASTA dataset...
[INFO] Preprocessing DNA sequences...
[INFO] Initializing CUDA environment...
[INFO] GPU detected successfully.
[INFO] Distributing genomic chunks using Dask...

[PROCESS] Genomic alignment in progress...

[GPU] Device utilization: Monitoring
[GPU] Memory usage: Monitoring
[SCAN] Alignment progress: 100%

[RESULT] Candidate off-target sites identified.

==================================================
              ALIGNMENT COMPLETED
==================================================

Total genomic regions scanned : N
Candidate off-target sites    : N
Maximum mismatches            : 3
Execution time                : Calculated at runtime

[INFO] Report generated successfully.
==================================================
```

*The output above is illustrative, not a measured execution result.*

## 📊 Performance & Optimization

GeneWeaver is designed to improve computational efficiency through:

1. **Parallel Processing:** Distributes independent sequence comparisons across GPU threads.
2. **Memory Optimization:** Processes large genomic datasets in manageable chunks.
3. **JIT Compilation:** Reduces Python execution overhead by compiling computational kernels.
4. **Distributed Execution:** Allows workloads to be distributed across multiple workers.
5. **Efficient Scheduling:** Uses Dask to coordinate data processing and result aggregation.

Performance should be benchmarked against a CPU baseline using identical datasets, mismatch thresholds, and alignment criteria.

## 🔬 Scientific Applications

GeneWeaver is designed to support computational genomics research, particularly:

- CRISPR-Cas9 off-target candidate identification.
- Large-scale DNA sequence alignment.
- Genomic data processing.
- GPU-accelerated bioinformatics.
- High-performance scientific computing.
- Computational analysis of gene-editing specificity.

**Research Disclaimer:** GeneWeaver identifies computationally predicted off-target candidates. Its results are not proof of actual mutations or biological effects and require appropriate experimental validation.

## 🚧 Future Enhancements

- [ ] Implement optimized multi-GPU CUDA execution.
- [ ] Support additional CRISPR systems and PAM configurations.
- [ ] Implement mismatch-position weighting and improved alignment scoring.
- [ ] Add reverse-complement strand analysis.
- [ ] Integrate more advanced off-target prediction algorithms.
- [ ] Add GPU memory optimization for whole-genome processing.
- [ ] Implement detailed CSV and JSON reports.
- [ ] Add performance benchmarking against CPU implementations.
- [ ] Introduce a web-based genomic analysis interface.
- [ ] Add cloud-based distributed execution support.

## 👨‍💻 Author

**Vrushabh Nipane**

B.Tech (Hons.) — Computer Science & Engineering (Data Science)

Chhattisgarh Swami Vivekananda Technical University (CSVTU)

- GitHub: [@ProgrammerVrushabh](https://github.com/ProgrammerVrushabh)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/vrushabh-l-a42a581a8/)

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test your implementation.
5. Submit a Pull Request.

## 📜 License

This project is intended to be distributed under the GNU License. See the `LICENSE` file for details.

---

<p align="center">
  <b>GeneWeaver — Accelerating Genomics Through Parallel Computing.</b>
  <br>
  Built with Python, CUDA, and a passion for computational innovation. 🧬
</p>
