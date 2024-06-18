import itertools as it
import networkx as nx
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from Bio.SeqIO import parse
import numpy as np
from tqdm import tqdm


class Assembler():
    def __init__(self, k):
        """
        Genome assembler based on De Bruijn graph

        :param k: k-mer length. Recommended k = 4n + 1
        """
        self.G = nx.MultiDiGraph()
        self.k = k

    def build_graph(self, reads):
        self.G = nx.MultiDiGraph()

        for read in tqdm(reads):
            for i in range(len(read) - self.k + 1):
                kmer = read[i: i + self.k]
                if not self.G.has_edge(kmer[:-1], kmer[1:]):
                    self.G.add_edge(
                        kmer[:-1], kmer[1:],
                        key=kmer,
                        coverage=0.0
                    )

                edge = self.G[kmer[:-1]][kmer[1:]][kmer]
                edge['coverage'] += 1.0

    def _compactify(self):
        nodes_list = list(self.G.nodes)
        np.random.shuffle(nodes_list)

        for node in nodes_list:
            if self.G.in_degree(node) == self.G.out_degree(node) == 1:
                pred = next(self.G.predecessors(node))
                succ = next(self.G.successors(node))
                if pred != node != succ:
                    seq1 = next(iter(self.G[pred][node]))
                    seq2 = next(iter(self.G[node][succ]))
                    seq_conc = seq1 + seq2[self.k - 1:]

                    cov1 = self.G[pred][node][seq1]['coverage']
                    cov2 = self.G[node][succ][seq2]['coverage']
                    len1, len2 = len(seq1), len(seq2)
                    cov_new = (cov1 * len1 + cov2 * len2) / (len1 + len2)

                    self.G.add_edge(pred, succ, key=seq_conc, coverage=cov_new)
                    self.G.remove_node(node)

    def _cut_tails(self, factor=0.3):
        tails = []

        for pred, succ, seq, attrs in self.G.edges(keys=True, data=True):
            cov = attrs['coverage']
            if (self.G.degree(pred) == 1) != (self.G.degree(succ) == 1):
                tails.append((len(seq) * cov, pred, succ, seq))

        if not tails:
            return

        tails = sorted(tails)
        max_len_cov = tails[-1][0]

        for len_cov, pred, succ, seq in tails:
            if len_cov < factor * max_len_cov:
                self.G.remove_edge(pred, succ, seq)

        self.G.remove_nodes_from(list(nx.isolates(self.G)))

    def _burst_bubbles(self, factor=0.3):
        bubble_pairs = set()

        for prev, succ in self.G.edges(keys=False):
            if len(self.G[prev][succ]) > 1:
                bubble_pairs.add((prev, succ))

        for prev, succ in bubble_pairs:
            bubble_edges = []
            for seq, attrs in self.G[prev][succ].items():
                cov = attrs['coverage']
                if len(seq) == 2 * self.k - 1:
                    bubble_edges.append((cov, seq))

            if not bubble_edges:
                continue

            bubble_edges = sorted(bubble_edges)
            max_cov = bubble_edges[-1][0]

            for cov, seq in bubble_edges:
                if cov < factor * max_cov:
                    self.G.remove_edge(prev, succ, seq)

    def _drop_low_covered_edges(self, factor=0.1):
        edges_list = []

        for prev, succ, seq, attrs in self.G.edges(keys=True, data=True):
            cov = attrs['coverage']
            edges_list.append((cov, prev, succ, seq))

        if not edges_list:
            return

        edges_list = sorted(edges_list)
        max_cov = edges_list[-1][0]

        for cov, prev, succ, seq in edges_list:
            if cov < factor * max_cov:
                self.G.remove_edge(prev, succ, seq)

        self.G.remove_nodes_from(list(nx.isolates(self.G)))

    def run(self):
        self._compactify()
        self._burst_bubbles()
        self._cut_tails()
        self._drop_low_covered_edges()
        self._compactify()

    def get_contigs(self):
        contigs = []

        for *_, seq, attrs in self.G.edges(keys=True, data=True):
            cov = attrs['coverage']
            contigs.append((seq, cov))

        contigs = sorted(contigs, key=lambda tup: tup[1], reverse=True)
        return contigs

    def print_graph_size(self):
        print(f'Graph size: {self.G.number_of_nodes()} nodes '
              f'and {self.G.number_of_edges()} edges')

    def plot_graph(self,
                   ax=None,
                   edge_labels='auto',
                   show_node_labels='auto',
                   layout='spring'):
        if ax is None:
            plt.figure(figsize=(12, 12))

        connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
        if layout == 'spring':
            pos = nx.spring_layout(self.G)
        elif layout == 'shell':
            pos = nx.shell_layout(self.G)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        nx.draw_networkx_nodes(self.G, pos, ax=ax)

        if show_node_labels == 'auto':
            if self.k <= 9:
                nx.draw_networkx_labels(self.G, pos, ax=ax)
        elif show_node_labels:
            nx.draw_networkx_labels(self.G, pos, ax=ax)

        nx.draw_networkx_edges(
            self.G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
        )

        if edge_labels == 'auto':
            labels = {
                tuple(edge): f"{self._short_view(seq)}\ncov={attrs['coverage']:.2f}"
                for *edge, seq, attrs in self.G.edges(keys=True, data=True)
            }
        elif edge_labels == 'coverage':
            labels = {
                tuple(edge): f"{attrs['coverage']:.2f}"
                for *edge, seq, attrs in self.G.edges(keys=True, data=True)
            }
        else:
            raise ValueError(f"Unexpected value for edge_labels: {edge_labels}")

        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            labels,
            connectionstyle=connectionstyle,
            label_pos=0.3,
            font_color="blue",
            bbox={"alpha": 0},
            ax=ax,
        )

    @staticmethod
    def _short_view(seq: Seq | str) -> str:
        if len(seq) <= 9:
            return str(seq)
        else:
            return f"<{len(seq)}bp>"


if __name__ == '__main__':
    for filename, k in [
        ('test/reads_0.fasta', 3),
        ('test/reads_1.fasta', 9),
        ('test/reads_2.fasta', 21),
        ('test/reads_3.fasta', 41),
    ]:
        assembler = Assembler(k=k)

        with open(filename, 'r') as f_in:
            reads = (record.seq for record in parse(f_in, 'fasta'))
            assembler.build_graph(reads)
            assembler.run()
            contigs = assembler.get_contigs()
            for contig in contigs:
                print(contig)

            if assembler.G.order() < 120:
                assembler.plot_graph()
                plt.show()
            else:
                print(assembler.G)
