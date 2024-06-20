import itertools as it

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm


class Assembler:
    def __init__(self, k):
        """
        Genome assembler based on De Bruijn graph

        :param k: k-mer length
        """
        self.G = nx.MultiDiGraph()
        self.k = k

    def build_graph(self, reads):
        self.G = nx.MultiDiGraph()

        for read in tqdm(reads):
            for i in range(len(read) - self.k + 1):
                kmer = str(read[i: i + self.k])
                prefix, suffix = kmer[:-1], kmer[1:]
                if not self.G.has_edge(prefix, suffix):
                    self.G.add_edge(
                        prefix, suffix,
                        key=kmer,
                        coverage=1.0
                    )
                else:
                    self.G[prefix][suffix][kmer]['coverage'] += 1.0

    def compactify(self, verbose=False):
        nodes_list = list(self.G.nodes)
        np.random.shuffle(nodes_list)

        if verbose:
            nodes_list = tqdm(nodes_list)

        for node in nodes_list:
            if self.G.in_degree(node) == self.G.out_degree(node) == 1:
                pred = next(self.G.predecessors(node))
                succ = next(self.G.successors(node))
                if pred != node != succ:
                    seq1 = next(iter(self.G[pred][node]))
                    seq2 = next(iter(self.G[node][succ]))
                    seq_new = seq1 + seq2[self.k - 1:]

                    cov1 = self.G[pred][node][seq1]['coverage']
                    cov2 = self.G[node][succ][seq2]['coverage']
                    len1, len2 = len(seq1), len(seq2)
                    cov_new = (cov1 * len1 + cov2 * len2) / (len1 + len2)

                    self.G.add_edge(pred, succ, key=seq_new, coverage=cov_new)
                    self.G.remove_node(node)

    def _del_isolates(self):
        self.G.remove_nodes_from(list(nx.isolates(self.G)))

    def cut_tails(self, factor=0.3):
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

        self._del_isolates()

    def burst_bubbles(self, cov_thres=1.001):
        bubble_pairs = set()

        for pred, succ in self.G.edges(keys=False):
            if len(self.G[pred][succ]) > 1:
                bubble_pairs.add((pred, succ))

        for pred, succ in bubble_pairs:
            bubble_edges = []
            for seq, attrs in self.G[pred][succ].items():
                cov = attrs['coverage']
                if len(seq) == 2 * self.k - 1:
                    bubble_edges.append((cov, seq))

            if not bubble_edges:
                continue

            for cov, seq in bubble_edges:
                if cov <= cov_thres:
                    self.G.remove_edge(pred, succ, seq)

    def drop_low_covered_edges(self, cov_thres=1.0):
        low_covered_edges = []

        for pred, succ, seq, attrs in self.G.edges(keys=True, data=True):
            if attrs['coverage'] <= cov_thres:
                low_covered_edges.append((pred, succ, seq))

        self.G.remove_edges_from(low_covered_edges)
        self._del_isolates()

    def run(self,
            verbose=False):
        """
        Run standard pipeline with default params: `compactify`, `burst_bubbles`,
        `cut_tails`, `drop_low_covered_edges` and `compactify` again

        :param verbose: (default *False*)
        """
        self.compactify(verbose=verbose)
        self.burst_bubbles()
        self.cut_tails()
        self.drop_low_covered_edges()
        self.compactify(verbose=verbose)

    def get_node_degrees(self):
        return list(self.G.degree)

    def get_tails(self):
        tails = []

        for pred, succ, seq, attrs in self.G.edges(keys=True, data=True):
            cov = attrs['coverage']
            if (self.G.degree(pred) == 1) != (self.G.degree(succ) == 1):
                tails.append((len(seq), cov, seq))

        return tails

    def get_edges(self):
        return [(attrs['coverage'], seq)
                for *_, seq, attrs in self.G.edges(keys=True, data=True)]

    def get_contigs(self):
        # should be modified with using euler paths in graph
        return sorted(self.get_edges(), reverse=True)

    def print_graph_size(self):
        print(f'Graph size: {self.G.number_of_nodes()} nodes '
              f'and {self.G.number_of_edges()} edges')

    def plot_graph(self,
                   subgraph=None,
                   ax=None,
                   edge_labels='auto',
                   show_node_labels='auto',
                   font_size=10,
                   layout=nx.kamada_kawai_layout):
        if subgraph is None:
            subgraph = self.G

        if ax is None:
            plt.figure(figsize=(12, 12))

        connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.30] * 4)]
        pos = layout(subgraph)

        nx.draw_networkx_nodes(subgraph, pos, ax=ax)

        if show_node_labels == 'auto':
            if self.k <= 9:
                nx.draw_networkx_labels(subgraph, pos, ax=ax)
        elif show_node_labels:
            nx.draw_networkx_labels(subgraph, pos, ax=ax)

        nx.draw_networkx_edges(
            subgraph, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
        )

        if edge_labels == 'auto':
            labels = {
                tuple(edge): f"{self._short_view(edge[-1])}\ncov={attrs['coverage']:.2f}"
                for *edge, attrs in subgraph.edges(keys=True, data=True)
            }
        elif edge_labels == 'coverage':
            labels = {
                tuple(edge): f"{attrs['coverage']:.2f}"
                for *edge, attrs in subgraph.edges(keys=True, data=True)
            }
        else:
            raise ValueError(f"Unexpected value for edge_labels: {edge_labels}")

        nx.draw_networkx_edge_labels(
            subgraph,
            pos,
            labels,
            connectionstyle=connectionstyle,
            label_pos=0.3,
            font_color="blue",
            font_size=font_size,
            bbox={"alpha": 0},
            ax=ax,
        )

    def plot_graph_componentwise(self, *args, **kwargs):
        for nodes in nx.weakly_connected_components(self.G):
            subgraph = self.G.subgraph(nodes)
            self.plot_graph(subgraph, *args, **kwargs)
            plt.show()

    @staticmethod
    def _short_view(seq: str) -> str:
        if len(seq) <= 9:
            return str(seq)
        else:
            return f"<{len(seq)}bp>"


if __name__ == '__main__':
    from Bio.SeqIO import parse

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
            print(f'Graph before: {assembler.G}')
            assembler.run(verbose=(k > 10))
            print(f'Graph after: {assembler.G}')
            # contigs = assembler.get_contigs()
            # for contig in contigs:
            #     print(contig)

            if assembler.G.order() < 120:
                assembler.plot_graph(font_size=9)
                plt.show()
            else:
                print(assembler.G)
