from pathlib import Path
from tqdm import trange

import numpy as np
import pandas as pd


_error_nucleotides = {
    'A': ['T', 'G', 'C'],
    'T': ['A', 'G', 'C'],
    'G': ['A', 'T', 'C'],
    'C': ['A', 'T', 'G'],
}


_ERROR_RATE_TABLE = Path("error_rate_table.csv")


def _calc_real_error_rate(n_mistakes, n_signals, n_gens=100_000):
    if 2 * n_mistakes < n_signals:
        # in fact, our simulation algorithm will never produce errors in such a case
        return 1 / n_gens

    if n_mistakes > 3 * (n_signals - n_mistakes):
        return 1.0

    rng = np.random.default_rng()
    n_read_errors = 0
    for _ in range(n_gens):
        error_signals = rng.choice(3, size=n_mistakes)
        _, counts = np.unique(error_signals, return_counts=True)
        if counts.max() > n_signals - n_mistakes:
            n_read_errors += 1

    if n_read_errors:
        return n_read_errors / n_gens
    else:
        return 1 / n_gens


def _create_error_rate_table(n_signals=100):
    data = []
    for n_mistakes in trange(n_signals + 1):
        err = _calc_real_error_rate(n_mistakes, n_signals)
        q_err = int(-10 * np.log10(err))
        data.append((err, q_err, chr(q_err + 33)))
    df = pd.DataFrame(data, columns=["error_rate", "Q", "chr"])
    return df


def run(sequence: str, n_reads: int, out_filename="reads", rng=None):
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    if not _ERROR_RATE_TABLE.exists():
        _create_error_rate_table().to_csv(_ERROR_RATE_TABLE, index=False)

    ert = pd.read_csv(_ERROR_RATE_TABLE)

    reads = []
    ground_truth = []

    for i in trange(n_reads):
        read_length = int(rng.normal(250, 30))
        start = rng.integers(0, len(sequence) - read_length)
        read_sequence = sequence[start: start + read_length]

        sequenced_read = []
        quality_scores = []
        for nucleotide in read_sequence:
            n_mistakes = rng.integers(75, endpoint=True)
            if n_mistakes <= 50:
                sequenced_read.append(nucleotide)
            else:
                pre_read = rng.choice(_error_nucleotides[nucleotide], size=n_mistakes)
                pre_read = np.hstack((pre_read, [nucleotide] * (100 - n_mistakes)))
                values, counts = np.unique(pre_read, return_counts=True)
                sequenced_read.append(values[np.argmax(counts)])
            quality_scores.append(ert.loc[n_mistakes, "Q"])

        reads.append((i, ''.join(sequenced_read), quality_scores))
        ground_truth.append((i, start, read_length))

    with open(out_filename + ".fastq", 'w') as f:
        for i, sequenced_read, quality_scores in reads:
            f.writelines([
                f"@{i}\n",
                sequenced_read,
                "\n+\n",
                ''.join(map(lambda x: chr(x + 33), quality_scores)),
                '\n',
            ])

    gt_df = pd.DataFrame(ground_truth, columns=["id", "start", "read_length"])
    gt_df.to_csv(out_filename + "_ground_truth.txt", index=False)

    return ground_truth, reads


if __name__ == '__main__':
    rng = np.random.default_rng(2024)
    genome = ''.join(rng.choice(list("ATGC"), size=50000))
    with open("genome.fasta", 'w') as f_out:
        f_out.write(">genome\n")
        f_out.write(genome)
        f_out.write("\n")

    ground_truth, reads = run(genome, n_reads=50000, out_filename="reads", rng=rng)
