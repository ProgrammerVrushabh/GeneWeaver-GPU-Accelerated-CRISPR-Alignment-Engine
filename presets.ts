import type { PamPattern } from './types.ts';

export interface GenomePreset {
  id: string;
  label: string;
  locus: string;
  description: string;
  recommendedSgRna: string;
  recommendedPam: PamPattern;
  targetBp: number;
}

export const SGRNA_PRESETS: Array<{
  id: string;
  label: string;
  gene: string;
  sequence: string;
  pam: PamPattern;
}> = [
  {
    id: 'tp53-ex4',
    label: 'TP53 Exon 4 (Tumor Suppressor)',
    gene: 'TP53',
    sequence: 'GTCAGATCCTAGCGTCGAGC',
    pam: 'NGG',
  },
  {
    id: 'hbb-sickle',
    label: 'HBB Codon 6 (Beta-Globin)',
    gene: 'HBB',
    sequence: 'GTAACGGCAGACTTCTCCTC',
    pam: 'NGG',
  },
  {
    id: 'cftr-df508',
    label: 'CFTR Exon 11 (ΔF508 Locus)',
    gene: 'CFTR',
    sequence: 'AGAAGAAAATATCATTGGTG',
    pam: 'NGG',
  },
  {
    id: 'emx1-std',
    label: 'EMX1 Benchmark Guide',
    gene: 'EMX1',
    sequence: 'GAGTCCGAGCAGAAGAAGAA',
    pam: 'NGG',
  },
];

export const GENOME_PRESETS: GenomePreset[] = [
  {
    id: 'chr17-tp53',
    label: 'Chr17 TP53 Genomic Region (120 Kbp)',
    locus: 'NC_000017.11 · 17p13.1',
    description: 'Contains canonical on-target TP53 exon 4 site plus seeded distal and seed-region SNV off-targets across both strands.',
    recommendedSgRna: 'GTCAGATCCTAGCGTCGAGC',
    recommendedPam: 'NGG',
    targetBp: 120_000,
  },
  {
    id: 'chr11-hbb',
    label: 'Chr11 HBB Beta-Globin Cluster (180 Kbp)',
    locus: 'NC_000011.10 · 11p15.4',
    description: 'Includes HBB, HBD delta-globin paralog, and pseudogene HBBP1 off-target homology sites.',
    recommendedSgRna: 'GTAACGGCAGACTTCTCCTC',
    recommendedPam: 'NGG',
    targetBp: 180_000,
  },
  {
    id: 'grch38-stress',
    label: 'GRCh38 Multi-Contig Stress Array (640 Kbp)',
    locus: 'GRCh38.p14 · Multi-Chr',
    description: '4-contig human genomic array for multi-GPU Dask load-balancing and memory-ceiling verification.',
    recommendedSgRna: 'GAGTCCGAGCAGAAGAAGAA',
    recommendedPam: 'NGG',
    targetBp: 640_000,
  },
];

/**
 * Deterministic PRNG (Mulberry32) so preset FASTA genomes are reproducible across runs.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mutateSpacer(spacer: string, indices: number[], rng: () => number): string {
  const bases = ['A', 'C', 'G', 'T'];
  const chars = spacer.split('');
  for (const idx of indices) {
    const orig = chars[idx];
    const alts = bases.filter((b) => b !== orig);
    chars[idx] = alts[Math.floor(rng() * alts.length)];
  }
  return chars.join('');
}

function revComp(seq: string): string {
  const map: Record<string, string> = { A: 'T', T: 'A', C: 'G', G: 'C' };
  return seq
    .split('')
    .reverse()
    .map((c) => map[c] ?? 'N')
    .join('');
}

/**
 * Generates a realistic FASTA string with planted biological on-target & off-target sites
 * (both + and - strands, seed and distal mismatches, canonical NGG and NAG PAMs).
 */
export function buildPresetFasta(presetId: string): { fileName: string; content: string } {
  const preset = GENOME_PRESETS.find((p) => p.id === presetId) ?? GENOME_PRESETS[0];
  const rng = mulberry32(presetId === 'chr17-tp53' ? 1704 : presetId === 'chr11-hbb' ? 1106 : 3814);
  const bases = ['A', 'C', 'G', 'T'];

  const makeRandomSeq = (len: number): string[] => {
    const arr = new Array<string>(len);
    for (let i = 0; i < len; i++) {
      // Slight GC bias (~42%) realistic for human genomic DNA
      const r = rng();
      arr[i] = r < 0.29 ? 'A' : r < 0.58 ? 'T' : r < 0.79 ? 'C' : 'G';
    }
    return arr;
  };

  const injectSite = (
    arr: string[],
    pos: number,
    spacer: string,
    pam: string,
    strand: '+' | '-'
  ) => {
    const full = strand === '+' ? spacer + pam : revComp(spacer + pam);
    for (let i = 0; i < full.length && pos + i < arr.length; i++) {
      arr[pos + i] = full.charAt(i);
    }
  };

  const sg = preset.recommendedSgRna;

  if (preset.id === 'grch38-stress') {
    const contigs = [
      { id: 'NC_000002.12_CHR2_EMX1', len: 160_000 },
      { id: 'NC_000007.14_CHR7_CFTR', len: 160_000 },
      { id: 'NC_000011.10_CHR11_HBB', len: 160_000 },
      { id: 'NC_000017.11_CHR17_TP53', len: 160_000 },
    ];
    const parts: string[] = [];
    contigs.forEach((c, cIdx) => {
      const seqArr = makeRandomSeq(c.len);
      if (cIdx === 0) {
        injectSite(seqArr, 14_280, sg, 'TGG', '+'); // On-target
        injectSite(seqArr, 58_910, mutateSpacer(sg, [1], rng), 'AGG', '-'); // Critical distal
        injectSite(seqArr, 112_400, mutateSpacer(sg, [2, 5], rng), 'CGG', '+');
      } else if (cIdx === 1) {
        injectSite(seqArr, 24_150, mutateSpacer(sg, [0, 3], rng), 'GGG', '+');
        injectSite(seqArr, 93_400, mutateSpacer(sg, [14, 17], rng), 'TGG', '-');
      } else if (cIdx === 2) {
        injectSite(seqArr, 31_720, mutateSpacer(sg, [4], rng), 'AGG', '+');
        injectSite(seqArr, 128_880, mutateSpacer(sg, [1, 13, 18], rng), 'CGG', '-');
      } else {
        injectSite(seqArr, 44_090, mutateSpacer(sg, [2], rng), 'TGG', '-');
        injectSite(seqArr, 141_200, mutateSpacer(sg, [6, 15], rng), 'AGG', '+');
      }
      parts.push(`>${c.id} Homo sapiens GRCh38.p14 primary assembly\n${seqArr.join('')}`);
    });
    return { fileName: 'grch38_multi_contig_640kb.fasta', content: parts.join('\n') };
  }

  const seqArr = makeRandomSeq(preset.targetBp);
  // Plant biologically representative on-target and off-target sites across both strands
  injectSite(seqArr, 8_420, sg, 'AGG', '+'); // Exact On-Target
  injectSite(seqArr, 23_190, mutateSpacer(sg, [1], rng), 'TGG', '+'); // Critical: 1 distal MM
  injectSite(seqArr, 41_805, mutateSpacer(sg, [0, 4], rng), 'CGG', '-'); // Critical/High: 2 distal MM on - strand
  injectSite(seqArr, 62_340, mutateSpacer(sg, [15], rng), 'GGG', '+'); // High: 1 seed MM
  injectSite(seqArr, 79_115, mutateSpacer(sg, [3, 16], rng), 'AGG', '-'); // Moderate: 1 distal + 1 seed
  injectSite(seqArr, 94_550, mutateSpacer(sg, [13, 17, 19], rng), 'TGG', '+'); // Low: 3 seed MM
  injectSite(seqArr, 108_220, mutateSpacer(sg, [2, 7, 14], rng), 'CGG', '-'); // Moderate/Low

  const headerId =
    preset.id === 'chr17-tp53'
      ? 'NC_000017.11_TP53_LOCUS'
      : 'NC_000011.10_HBB_CLUSTER';

  return {
    fileName: `${preset.id}.fasta`,
    content: `>${headerId} ${preset.locus} Homo sapiens chromosome assembly\n${seqArr.join('')}`,
  };
}
