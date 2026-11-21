export type PamPattern = 'NGG' | 'NAG' | 'NG' | 'TTTV';

export type HardwareTopology = '1x-gpu' | '2x-gpu-dask' | 'cpu-baseline';

export type SeverityTier = 'On-Target' | 'Critical' | 'High' | 'Moderate' | 'Low';

export interface FastaContig {
  id: string;
  header: string;
  sequence: string;
  lengthBp: number;
  gcPercent: number;
  baseCounts: {
    A: number;
    C: number;
    G: number;
    T: number;
    N: number;
  };
}

export interface ParsedFastaResult {
  fileName: string;
  contigs: FastaContig[];
  totalBp: number;
  overallGcPercent: number;
  parseTimeMs: number;
}

export interface SequenceChunk {
  chunkIndex: number;
  contigId: string;
  startOffsetBp: number;
  endOffsetBp: number;
  lengthBp: number;
  sequence: string;
  assignedGpu: 0 | 1;
}

export interface AlignmentHit {
  id: string;
  contigId: string;
  chunkIndex: number;
  assignedGpu: 0 | 1;
  strand: '+' | '-';
  startPos: number;
  endPos: number;
  matchedSpacer: string;
  pamSequence: string;
  flankLeft: string;
  flankRight: string;
  mismatchCount: number;
  mismatchIndices: number[];
  seedMismatchCount: number;
  distalMismatchCount: number;
  pamProximityScore: number;
  cfdRiskScore: number;
  severity: SeverityTier;
  mutationType: string;
}

export interface ChunkTelemetry {
  chunkIndex: number;
  contigId: string;
  assignedGpu: 0 | 1;
  lengthBp: number;
  h2dTransferMs: number;
  kernelComputeMs: number;
  d2hTransferMs: number;
  vramAllocatedMb: number;
  gpuCoreUtilPercent: number;
  hitsInChunk: number;
}

export interface VramBudgetEstimate {
  chunkSizeBp: number;
  perChunkVramMb: number;
  peakVramPerGpuMb: number;
  gpuVramCapacityMb: number;
  utilizationPercent: number;
  sharedMemPerBlockKb: number;
  unChunkedFullGenomeGb: number;
  oomRisk: 'Safe' | 'Elevated' | 'OOM Risk';
}

export interface ScanSummary {
  scanId: string;
  fileName: string;
  sgRna: string;
  pamPattern: PamPattern;
  maxMismatches: number;
  topology: HardwareTopology;
  sharedMemoryEnabled: boolean;
  chunkSizeBp: number;
  totalBpScanned: number;
  totalChunks: number;
  gpu0Chunks: number;
  gpu1Chunks: number;
  hits: AlignmentHit[];
  chunkTelemetry: ChunkTelemetry[];
  vramBudget: VramBudgetEstimate;
  cpuBaselineMs: number;
  gpuKernelMs: number;
  effectiveThroughputMbpSec: number;
  speedupFactor: number;
  sharedMemoryLatencyReductionPercent: number;
  completedAt: string;
}

export interface BenchmarkRunResult {
  sampleSizeBp: number;
  cpuTimeMs: number;
  gpuGlobalTimeMs: number;
  gpuSharedMemTimeMs: number;
  cpuThroughputMbpSec: number;
  gpuSharedThroughputMbpSec: number;
  speedupGlobalVsCpu: number;
  speedupSharedVsCpu: number;
  hitsVerifiedEqual: boolean;
  hitsFound: number;
  projectedHumanGenomeCpuHours: number;
  projectedHumanGenomeGpuSeconds: number;
}

export interface ScanConfig {
  sgRna: string;
  pamPattern: PamPattern;
  maxMismatches: number;
  chunkSizeBp: number;
  topology: HardwareTopology;
  sharedMemoryEnabled: boolean;
}
