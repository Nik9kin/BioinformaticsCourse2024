import numpy as np
from Bio.Align import substitution_matrices


def hamming_dist(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


rev_codon_table = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'C': ['TGT', 'TGC'],
    'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'],
    'F': ['TTT', 'TTC'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'M': ['ATG'],
    'N': ['AAT', 'AAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
}

amino_acids = list(rev_codon_table.keys())

blosum62 = substitution_matrices.load('BLOSUM62')
blosum62_np = np.zeros((20, 20))
mean_codon_dist = np.zeros((20, 20))

for i, aa1 in enumerate(amino_acids):
    codons1 = rev_codon_table[aa1]
    for j, aa2 in enumerate(amino_acids):
        codons2 = rev_codon_table[aa2]

        blosum62_np[i, j] = blosum62[aa1, aa2]

        mean_codon_dist[i, j] = sum(
            hamming_dist(s1, s2) for s1 in codons1 for s2 in codons2
        ) / (len(codons1) * len(codons2))

print(np.corrcoef(blosum62_np.flatten(), mean_codon_dist.flatten())[0, 1])
