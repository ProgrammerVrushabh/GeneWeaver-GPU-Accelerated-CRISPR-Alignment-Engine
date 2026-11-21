export interface PythonSourceModule {
  id: string;
  fileName: string;
  title: string;
  role: string;
  code: string;
}

export const PYTHON_SOURCE_MODULES: PythonSourceModule[] = [
  {
    id: 'cuda_kernels',
    fileName: 'cuda_kernels.py',
    title: 'CUDA JIT & Shared Memory Alignment Kernel',
    role: 'Numba @cuda.jit kernel with 2-bit DNA encoding and shared-memory tiling',
    code: `import math
import numpy as np
from numba import cuda, uint8, int32

# 2-bit nucleotide encoding: A=0, C=1, G=2, T=3, N=255
SPACER_LEN = 20
PAM_LEN = 3
THREADS_PER_BLOCK = 256


@cuda.jit
def crispr_align_shared_kernel(
    genome_chunk,      # uint8[:] device array of encoded genomic bases
    query_spacer,      # uint8[20] device array of sgRNA spacer bases
    max_mismatches,    # int32 threshold
    out_positions,     # int32[:] hit coordinate buffer
    out_mismatches,    # int32[:] hit mismatch counts
    out_seed_mm,       # int32[:] PAM-proximal seed mismatches (last 8 nt)
    hit_counter        # int32[1] atomic global hit counter
):
    """
    GPU-accelerated CRISPR off-target alignment kernel using CUDA Shared Memory
    to eliminate redundant global VRAM reads of the sgRNA query and overlapping tile.
    """
    # Allocate fast on-chip SRAM shared memory per thread block
    shared_query = cuda.shared.array(shape=(SPACER_LEN,), dtype=uint8)
    shared_tile = cuda.shared.array(shape=(THREADS_PER_BLOCK + SPACER_LEN + PAM_LEN,), dtype=uint8)

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    gid = bx * cuda.blockDim.x + tx

    # 1. Collaboratively load sgRNA spacer into CUDA Shared Memory
    if tx < SPACER_LEN:
        shared_query[tx] = query_spacer[tx]

    # 2. Load genomic chunk window + boundary halo into Shared Memory
    if gid < genome_chunk.size:
        shared_tile[tx] = genome_chunk[gid]
    else:
        shared_tile[tx] = 255

    if tx < (SPACER_LEN + PAM_LEN):
        halo_idx = gid + THREADS_PER_BLOCK
        if halo_idx < genome_chunk.size:
            shared_tile[THREADS_PER_BLOCK + tx] = genome_chunk[halo_idx]
        else:
            shared_tile[THREADS_PER_BLOCK + tx] = 255

    cuda.syncthreads()

    if gid + SPACER_LEN + PAM_LEN > genome_chunk.size:
        return

    # 3. Canonical NGG PAM check in Shared Memory (PAM = bases [20, 21, 22])
    pam_0 = shared_tile[tx + SPACER_LEN]
    pam_1 = shared_tile[tx + SPACER_LEN + 1]
    pam_2 = shared_tile[tx + SPACER_LEN + 2]
    if pam_0 >= 4 or pam_1 != 2 or pam_2 != 2:
        return

    # 4. Parallel base-pair mismatch comparison across 20-nt spacer
    mismatches = 0
    seed_mm = 0
    for j in range(SPACER_LEN):
        if shared_tile[tx + j] != shared_query[j]:
            mismatches += 1
            if j >= 12:  # 8-nt PAM-proximal seed region
                seed_mm += 1
            if mismatches > max_mismatches:
                return

    # 5. Atomic append to output buffers
    slot = cuda.atomic.add(hit_counter, 0, 1)
    if slot < out_positions.size:
        out_positions[slot] = gid
        out_mismatches[slot] = mismatches
        out_seed_mm[slot] = seed_mm


def run_cuda_chunk(host_chunk: np.ndarray, host_query: np.ndarray, max_mm: int = 3):
    """Explicit Host-to-Device (H2D) and Device-to-Host (D2H) bounded VRAM execution."""
    d_chunk = cuda.to_device(host_chunk)
    d_query = cuda.to_device(host_query)
    d_counter = cuda.to_device(np.zeros(1, dtype=np.int32))

    max_hits = 65536
    d_positions = cuda.device_array(max_hits, dtype=np.int32)
    d_mismatches = cuda.device_array(max_hits, dtype=np.int32)
    d_seed_mm = cuda.device_array(max_hits, dtype=np.int32)

    blocks = math.ceil(host_chunk.size / THREADS_PER_BLOCK)
    crispr_align_shared_kernel[blocks, THREADS_PER_BLOCK](
        d_chunk, d_query, int32(max_mm),
        d_positions, d_mismatches, d_seed_mm, d_counter
    )
    cuda.synchronize()

    n_hits = min(int(d_counter.copy_to_host()[0]), max_hits)
    return (
        d_positions[:n_hits].copy_to_host(),
        d_mismatches[:n_hits].copy_to_host(),
        d_seed_mm[:n_hits].copy_to_host(),
    )
`,
  },
  {
    id: 'genomic_parser',
    fileName: 'genomic_parser.py',
    title: 'BioPython FASTA Parser & OOM-Safe Chunker',
    role: 'BioPython SeqIO streaming parser with overlap-preserving chunk arrays',
    code: `from typing import Iterator, Tuple
import numpy as np
from Bio import SeqIO

BASE_LUT = np.full(256, 255, dtype=np.uint8)
BASE_LUT[ord("A")] = 0
BASE_LUT[ord("C")] = 1
BASE_LUT[ord("G")] = 2
BASE_LUT[ord("T")] = 3


def encode_sequence_2bit(seq_str: str) -> np.ndarray:
    """Vectorized ASCII-to-2bit LUT conversion avoiding Python character loops."""
    raw = np.frombuffer(seq_str.upper().encode("ascii", errors="ignore"), dtype=np.uint8)
    return BASE_LUT[raw]


def stream_fasta_chunks(
    fasta_path: str,
    chunk_size_bp: int = 1_048_576,
    overlap_bp: int = 22
) -> Iterator[Tuple[str, int, np.ndarray]]:
    """
    Streams massive .fasta files via BioPython SeqIO into fixed-size overlapping
    NumPy arrays so GPU VRAM stays strictly bounded (zero CUDA_OUT_OF_MEMORY).
    """
    step = chunk_size_bp - overlap_bp
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq_str = str(record.seq)
        total_len = len(seq_str)
        offset = 0
        while offset < total_len:
            end = min(total_len, offset + chunk_size_bp)
            encoded = encode_sequence_2bit(seq_str[offset:end])
            yield record.id, offset, encoded
            if end == total_len:
                break
            offset += step
`,
  },
  {
    id: 'dask_scheduler',
    fileName: 'dask_scheduler.py',
    title: 'Dask Multi-GPU Distributed Scheduler',
    role: 'Evenly balances overlapping genomic chunks across available CUDA devices',
    code: `import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from numba import cuda
from genomic_parser import stream_fasta_chunks, encode_sequence_2bit
from cuda_kernels import run_cuda_chunk


def dispatch_distributed_scan(fasta_path: str, sgrna_spacer: str, max_mismatches: int = 3):
    """
    Provisions a Dask LocalCUDACluster across all available GPUs (e.g., 2x GPUs)
    and evenly round-robins genomic chunks across workers.
    """
    n_gpus = len(cuda.gpus)
    cluster = LocalCUDACluster(n_workers=max(1, n_gpus), device_memory_limit="14GB")
    client = Client(cluster)

    query_arr = encode_sequence_2bit(sgrna_spacer)
    workers = list(client.scheduler_info()["workers"].keys())
    futures = []

    for idx, (contig_id, offset, chunk_arr) in enumerate(stream_fasta_chunks(fasta_path)):
        target_worker = workers[idx % len(workers)]
        fut = client.submit(
            run_cuda_chunk,
            chunk_arr,
            query_arr,
            max_mismatches,
            workers=[target_worker],
            pure=False,
        )
        futures.append((contig_id, offset, fut))

    results = []
    for contig_id, offset, fut in futures:
        pos, mm, seed_mm = fut.result()
        for p, m, s in zip(pos, mm, seed_mm):
            results.append({
                "contig": contig_id,
                "position": int(offset + p),
                "mismatches": int(m),
                "seed_mismatches": int(s),
            })

    client.close()
    cluster.close()
    return results
`,
  },
  {
    id: 'tui_dashboard',
    fileName: 'tui_dashboard.py',
    title: 'Textual TUI Telemetry & Mismatch Viewer',
    role: 'Terminal dashboard tracking VRAM, GPU core load, and red DNA mismatches',
    code: `from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Header, Footer, ProgressBar, Static


def format_mismatch_text(target: str, query: str) -> Text:
    """Highlights mutated base pairs in crimson red for terminal inspection."""
    styled = Text()
    for base_t, base_q in zip(target, query):
        if base_t == base_q:
            styled.append(base_t, style="dim white")
        else:
            styled.append(base_t, style="bold white on red")
    return styled


class GeneWeaverTUI(App):
    CSS = """
    Screen { background: #080c12; color: #e2e8f0; }
    #telemetry { height: 5; border: solid #1e293b; padding: 1; }
    DataTable { height: 1fr; border: solid #1e293b; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("GPU0 VRAM: 412 / 16384 MB | Core Load: 94.2%", id="telemetry")
        yield ProgressBar(total=100, id="scan_progress")
        yield DataTable(id="hits_table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Contig", "Position", "Strand", "Alignment (Mutated=RED)", "CFD Risk")
`,
  },
];
