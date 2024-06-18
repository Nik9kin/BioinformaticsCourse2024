import numpy as np

from Bio.Seq import Seq, MutableSeq
from Bio.SeqIO import SeqRecord, write


DNA_ALPHABET = ('A', 'T', 'G', 'C')

rng = np.random.default_rng(2024)


def write_test(genome, reads, test_id):
    genome = SeqRecord(
        genome,
        id=f'genome_{test_id}',
        description=f'{len(genome)}bp'
    )
    reads = [
        SeqRecord(seq, id=str(i), description='')
        for i, seq in enumerate(reads)
    ]

    with open(f'genome_{test_id}.fasta', 'w') as f_out:
        write(genome, f_out, 'fasta')

    with open(f'reads_{test_id}.fasta', 'w') as f_out:
        write(reads, f_out, 'fasta')


# Genome length: 12bp
# Reads: all 3-mers exactly once
# No errors

genome0 = Seq(''.join(rng.choice(DNA_ALPHABET, 12)))
reads0 = [genome0[i:i + 3] for i in range(len(genome0) - 2)]
rng.shuffle(reads0)
write_test(genome0, reads0, 0)


# Genome length: 300bp
# Reads: Random position and length (20 - 40bp)
# Coverage: 10
# No errors

genome1 = Seq(''.join(rng.choice(DNA_ALPHABET, 300)))
reads1 = []
for i in range(100):
    read_len = rng.integers(20, 40, endpoint=True)
    start = rng.integers(301 - read_len)
    reads1.append(genome1[start:start + read_len])

write_test(genome1, reads1, 1)


# Genome length: 1000bp
# Reads: Random position and length (40 - 60bp)
# Coverage: 20
# Errors: 0.1%

genome2 = Seq(''.join(rng.choice(DNA_ALPHABET, 1000)))
reads2 = []

for i in range(400):
    read_len = rng.integers(40, 60, endpoint=True)
    start = rng.integers(1001 - read_len)
    read = MutableSeq(genome2[start:start + read_len])

    error_poses = rng.binomial(1, 0.001, size=read_len)
    for j in range(read_len):
        if error_poses[j]:
            read[j] = rng.choice(DNA_ALPHABET)

    reads2.append(read)

write_test(genome2, reads2, 2)


# Genome length: 100_000bp
# Reads: Random position and length (150 - 250bp)
# Coverage: 40
# Errors: 0.1%

genome3 = Seq(''.join(rng.choice(DNA_ALPHABET, 100_000)))
reads3 = []

for i in range(20_000):
    read_len = rng.integers(150, 250, endpoint=True)
    start = rng.integers(100_001 - read_len)
    read = MutableSeq(genome3[start:start + read_len])

    error_poses = rng.binomial(1, 0.001, size=read_len)
    for j in range(read_len):
        if error_poses[j]:
            read[j] = rng.choice(DNA_ALPHABET)

    reads3.append(read)

write_test(genome3, reads3, 3)
