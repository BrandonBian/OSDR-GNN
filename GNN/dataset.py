import sys

sys.path.append("..")
import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
from collections import Counter
from autodesk_colab_KG import load_data
from torch_geometric.data import DataLoader, Data
from sklearn.model_selection import train_test_split


def get_vocab():
    # Input: a list of NetworkX graphs (from initial step)

    graphs = load_data('../data/autodesk_colab_fullv3_202010291746.csv')

    # TODO: add material counts - done
    subfunc_counts, tier1_counts, tier2_counts, tier3_counts, material_counts = [], [], [], [], []
    sys_name_counts = []
    vocab = {
        'component_basis': set(),
        'sys_name': set(),
        'sys_type_name': set(),
        'material_name': set(),
        'subfunction_basis': set(),
        'tier_1_function': set(),
        'tier_2_function': set(),
        'tier_3_function': set(),
        'flow': set(),
    }
    for g in graphs:
        for __, attr in g.nodes(data=True):
            vocab['component_basis'].add(attr['component_basis'])
            vocab['sys_name'].add(attr['sys_name'])
            vocab['sys_type_name'].add(attr['sys_type_name'])
            vocab['material_name'].add(attr['material_name'])
            vocab['subfunction_basis'].add(attr['subfunction_basis'])
            vocab['tier_1_function'].add(attr['tier_1_function'])
            vocab['tier_2_function'].add(attr['tier_2_function'])
            vocab['tier_3_function'].add(attr['tier_3_function'])

            subfunc_counts.append(attr['subfunction_basis'])
            tier1_counts.append(attr['tier_1_function'])
            tier2_counts.append(attr['tier_2_function'])
            tier3_counts.append(attr['tier_3_function'])

            sys_name_counts.append(attr['sys_name'])

            # TODO: update material_counts - done
            material_counts.append(attr['material_name'])

        for attr in g.edges(data=True):

            if 'input_flow' in attr[-1]:
                vocab['flow'].add(attr[-1]['input_flow'])
            if 'output_flow' in attr[-1]:
                vocab['flow'].add(attr[-1]['output_flow'])

    for k, v in vocab.items():
        vocab[k] = {s: idx for idx, s in enumerate(sorted(v))}

    # print(vocab['sys_name'])

    subfunc_counts = [vocab['subfunction_basis'][l] for l in subfunc_counts]
    tier1_counts = [vocab['tier_1_function'][l] for l in tier1_counts]
    tier2_counts = [vocab['tier_2_function'][l] for l in tier2_counts]
    tier3_counts = [vocab['tier_3_function'][l] for l in tier3_counts]

    # Other counts (for plotting distribution histograms)

    sys_name_counts = [vocab['sys_name'][l] for l in sys_name_counts]

    # print(Counter(sys_name_counts))




    # TODO: update material_counts - done
    material_counts = [vocab['material_name'][l] for l in material_counts]

    subfunc_w = [0.] * len(vocab['subfunction_basis'])
    tier1_w = [0.] * len(vocab['tier_1_function'])
    tier2_w = [0.] * len(vocab['tier_2_function'])
    tier3_w = [0.] * len(vocab['tier_3_function'])

    # TODO: update material_w weights - done
    material_w = [0.] * len(vocab['material_name'])

    # k = index of instance, v = number of appearance of this instance

    for k, v in Counter(subfunc_counts).items():  # 15636/number of instances of a certain attribute (v)
        subfunc_w[k] = len(subfunc_counts) / v
    for k, v in Counter(tier1_counts).items():
        tier1_w[k] = len(tier1_counts) / v
    for k, v in Counter(tier2_counts).items():
        tier2_w[k] = len(tier2_counts) / v
    for k, v in Counter(tier3_counts).items():
        tier3_w[k] = len(tier3_counts) / v




    # TODO: calculate the weights of the material - done
    for k, v in Counter(material_counts).items():
        material_w[k] = len(material_counts) / v



    # Dictionary of weights
    # TODO: add the weights of material to the weights dictionary - done
    weights = {0: torch.tensor(subfunc_w), 1: torch.tensor(tier1_w),
               2: torch.tensor(tier2_w), 3: torch.tensor(tier3_w), 4: torch.tensor(material_w)}

    # graphs: original list of graphs
    # vocab: dictionary of sets of unique attributes
    # weights: dictionary of (15636/number of instances of a certain attribute)
    return graphs, vocab, weights


def degree_encoding(max_degree=100, dimension=16):
    deg_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dimension) for j in range(dimension)]
        if pos != 0 else np.zeros(dimension) for pos in range(max_degree + 1)])
    deg_enc[1:, 0::2] = np.sin(deg_enc[1:, 0::2])
    deg_enc[1:, 1::2] = np.cos(deg_enc[1:, 1::2])
    return deg_enc


def draw_class(): # TODO: change to draw other classes distribution as well - done
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter

    graphs = load_data('../data/autodesk_colab_fullv3_202010291746.csv')
    tier1_counts, tier2_counts, tier3_counts = [], [], []
    sys_name_counts, flow_counts, component_basis = [], [], []

    for g in graphs:
        for __, attr in g.nodes(data=True): # Node features
            # tier1_counts.append(attr['tier_1_function'])
            # tier2_counts.append(attr['tier_2_function'])
            # tier3_counts.append(attr['tier_3_function'])

            sys_name_counts.append(attr['sys_name'])
            component_basis.append(attr['component_basis'])

        for attr in g.edges(data=True): # Edge features
            if 'input_flow' in attr[-1]:
                flow_counts.append(attr[-1]['input_flow'])
            if 'output_flow' in attr[-1]:
                flow_counts.append(attr[-1]['output_flow'])


    f, l = [], []
    for k, v in Counter(sys_name_counts).most_common():
        if k:
            f.append(v / len(sys_name_counts))
            l.append(k)
    d = pd.DataFrame({'Frequency': f, 'Class': l})
    plt.figure(figsize=(12, 9))
    sn.barplot(data=d, y='Class', x='Frequency', color='cyan')
    plt.xlabel('Frequency', size='xx-large')
    plt.ylabel('System Name', size='xx-large')
    plt.xticks(size='xx-large')
    plt.yticks(size='xx-large')
    plt.tight_layout()
    # plt.savefig(fname='logs/tier1_freq.pdf', format='pdf')
    plt.show()

    f, l = [], []
    for k, v in Counter(flow_counts).most_common():
        if k:
            f.append(v / len(flow_counts))
            l.append(k)
    d = pd.DataFrame({'Frequency': f, 'Class': l})
    plt.figure(figsize=(13, 10))
    sn.barplot(data=d, y='Class', x='Frequency', color='cyan')
    plt.xlabel('Frequency', size='xx-large')
    plt.ylabel('Flow', size='xx-large')
    plt.xticks(size='xx-large')
    plt.yticks(size='xx-large')
    plt.tight_layout()
    # plt.savefig(fname='logs/tier2_freq.pdf', format='pdf')
    plt.show()

    f, l = [], []
    for k, v in Counter(component_basis).most_common():
        if k:
            f.append(v / len(component_basis))
            l.append(k)
    d = pd.DataFrame({'Frequency': f, 'Class': l})
    plt.figure(figsize=(13, 10))
    sn.barplot(data=d, y='Class', x='Frequency', color='cyan')
    plt.xlabel('Frequency', size='xx-large')
    plt.ylabel('Component Basis', size='xx-large')
    plt.xticks(size='xx-large')
    plt.yticks(size='xx-large')
    plt.tight_layout()
    # plt.savefig(fname='logs/tier2_freq.pdf', format='pdf')
    plt.show()

    # f, l = [], []
    # for k, v in Counter(tier3_counts).most_common():
    #     if k:
    #         f.append(v / len(tier3_counts))
    #         l.append(k)
    # d = pd.DataFrame({'Frequency': f, 'Class': l})
    # plt.figure(figsize=(13, 10))
    # sn.barplot(data=d, y='Class', x='Frequency', color='cyan')
    # plt.xlabel('Frequency', size='xx-large')
    # plt.ylabel('Class', size='xx-large')
    # plt.xticks(size='xx-large')
    # plt.yticks(size='xx-large')
    # plt.tight_layout()
    # plt.savefig(fname='logs/tier3_freq.pdf', format='pdf')
    # plt.show()


def preprocess(node_feature, edge_feature):
    print("Start preprocessing...")
    graphs = []
    data, vocab, weights = get_vocab()  # Get the graphs, the vocab, and the weights

    max_degree = max([max(dict(nx.degree(graph)).values()) for graph in data])

    for graph in data:
        if graph.number_of_nodes() < 3 or graph.number_of_edges() < 2:  # graph too small, pass
            continue
        nodes, edges = [], []

        # relabel index/uid into corresponding strings/names (e.g. 58 -> battery tray)
        mappings = {n: idx for idx, n in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mappings)

        # For initial node attributes, we concatenate one-hot encoding of component basis, system name,
        # system type, and material features resulting in a 316- dimensional multi-hot initial node feature.

        # one-hot encoding of the vocabs, concatenated as a node attribute
        if node_feature == 'none':
            deg_enc = degree_encoding(max_degree)
            nodes = torch.tensor([deg_enc[graph.degree[n[0]]] for n in graph.nodes(data=True)])
        elif node_feature == 'component':
            nodes = F.one_hot(
                torch.tensor([vocab['component_basis'][n[-1]['component_basis']] for n in graph.nodes(data=True)]),
                len(vocab['component_basis']))
        elif node_feature == 'name':
            nodes = F.one_hot(
                torch.tensor([vocab['sys_name'][n[-1]['sys_name']] for n in graph.nodes(data=True)]),
                len(vocab['sys_name']))
        elif node_feature == 'type':
            nodes = F.one_hot(
                torch.tensor([vocab['sys_type_name'][n[-1]['sys_type_name']] for n in graph.nodes(data=True)]),
                len(vocab['sys_type_name']))
        elif node_feature == 'material':
            nodes = F.one_hot(
                torch.tensor([vocab['material_name'][n[-1]['material_name']] for n in graph.nodes(data=True)]),
                len(vocab['material_name']))

        # TODO: add the choice for considering only tier functions (for ablation study) - done

        elif node_feature == 'tier_function':
            nodes = torch.cat((
                F.one_hot(
                    torch.tensor(
                        [vocab['subfunction_basis'][n[-1]['subfunction_basis']] for n in graph.nodes(data=True)]),
                    len(vocab['subfunction_basis'])),

                F.one_hot(
                    torch.tensor(
                        [vocab['tier_1_function'][n[-1]['tier_1_function']] for n in graph.nodes(data=True)]),
                    len(vocab['tier_1_function'])),

                F.one_hot(
                    torch.tensor(
                        [vocab['tier_2_function'][n[-1]['tier_2_function']] for n in graph.nodes(data=True)]),
                    len(vocab['tier_2_function'])),

                F.one_hot(
                    torch.tensor(
                        [vocab['tier_3_function'][n[-1]['tier_3_function']] for n in graph.nodes(data=True)]),
                    len(vocab['tier_3_function'])),

            ), -1)

        elif node_feature == 'all':
            nodes = torch.cat((
                F.one_hot(
                    torch.tensor([vocab['component_basis'][n[-1]['component_basis']] for n in graph.nodes(data=True)]),
                    len(vocab['component_basis'])),
                F.one_hot(
                    torch.tensor([vocab['sys_name'][n[-1]['sys_name']] for n in graph.nodes(data=True)]),
                    len(vocab['sys_name'])),
                F.one_hot(
                    torch.tensor([vocab['sys_type_name'][n[-1]['sys_type_name']] for n in graph.nodes(data=True)]),
                    len(vocab['sys_type_name'])),

                # TODO: remove one-hot encoding of materials - done
                # F.one_hot(
                #     torch.tensor([vocab['material_name'][n[-1]['material_name']] for n in graph.nodes(data=True)]),
                #     len(vocab['material_name'])),

                # TODO: add the one-hot encoding of function basis and tier functions - done
                F.one_hot(
                    torch.tensor(
                        [vocab['subfunction_basis'][n[-1]['subfunction_basis']] for n in graph.nodes(data=True)]),
                    len(vocab['subfunction_basis'])),

                F.one_hot(
                    torch.tensor(
                        [vocab['tier_1_function'][n[-1]['tier_1_function']] for n in graph.nodes(data=True)]),
                    len(vocab['tier_1_function'])),

                F.one_hot(
                    torch.tensor(
                        [vocab['tier_2_function'][n[-1]['tier_2_function']] for n in graph.nodes(data=True)]),
                    len(vocab['tier_2_function'])),

                F.one_hot(
                    torch.tensor(
                        [vocab['tier_3_function'][n[-1]['tier_3_function']] for n in graph.nodes(data=True)]),
                    len(vocab['tier_3_function'])),

            ), -1)

        y = torch.tensor([vocab[f'subfunction_basis'][n[-1][f'subfunction_basis']] for n in graph.nodes(data=True)])
        y1 = torch.tensor([vocab[f'tier_1_function'][n[-1][f'tier_1_function']] for n in graph.nodes(data=True)])
        y2 = torch.tensor([vocab[f'tier_2_function'][n[-1][f'tier_2_function']] for n in graph.nodes(data=True)])
        y3 = torch.tensor([vocab[f'tier_3_function'][n[-1][f'tier_3_function']] for n in graph.nodes(data=True)])

        # TODO: create a tensor to store the material_name ids (0-16) of all nodes of a graph - done
        material = torch.tensor([vocab[f'material_name'][n[-1][f'material_name']] for n in graph.nodes(data=True)])

        if edge_feature == 'all':
            for edge in graph.edges(data=True):
                edges.append(torch.zeros(2 * len(vocab['flow']) + 1))  # list of tensors
                if len(edge[-1]) == 0:  # no flow
                    edges[-1][0] = 1.
                else:
                    if 'input_flow' in edge[-1]:
                        # what is this doing? Multi-hot encoding? (+1 means including assembly relation)?
                        edges[-1][vocab['flow'][edge[-1]['input_flow']] + 1] = 1.
                    if 'output_flow' in edge[-1]:
                        edges[-1][vocab['flow'][edge[-1]['output_flow']] + len(vocab['flow']) + 1] = 1.

            edges = torch.stack(edges)  # concatenate torch tensors, dim=0 by default

            edge_index = torch.tensor([[e[0], e[1]] for e in graph.edges()]).transpose(1, 0)  # each edge 2 node indexes

        elif edge_feature == 'flow':  # not including assembly relation
            try:
                edge_index = \
                    torch.tensor([[e[0], e[1]] for e in graph.edges(data=True) if len(e[-1]) > 0]).transpose(1, 0)
            except:
                continue
            for edge in graph.edges(data=True):
                if len(edge[-1]) > 0:
                    edges.append(torch.zeros(2 * len(vocab['flow'])))
                    if 'input_flow' in edge[-1]:
                        edges[-1][vocab['flow'][edge[-1]['input_flow']]] = 1.
                    if 'output_flow' in edge[-1]:
                        edges[-1][vocab['flow'][edge[-1]['output_flow']] + len(vocab['flow'])] = 1.
            edges = torch.stack(edges)

        elif edge_feature == 'solid':
            try:
                edge_index = torch.tensor([[e[0], e[1]] for e in graph.edges(data=True)
                                           if len(e[-1]) > 0 and
                                           (('input_flow' in e[-1] and e[-1]['input_flow'] == 'solid') or
                                            ('output_flow' in e[-1] and e[-1]['output_flow'] == 'solid'))]) \
                    .transpose(1, 0)
            except:
                continue
            for edge in graph.edges(data=True):
                if len(edge[-1]) > 0:
                    e = torch.zeros(2)
                    if 'input_flow' in edge[-1] and edge[-1]['input_flow'] == 'solid':
                        e[0] = 1.
                    if 'output_flow' in edge[-1] and edge[-1]['output_flow'] == 'solid':
                        e[1] = 1.
                    if torch.sum(e) > 0:
                        edges.append(e)
            edges = torch.stack(edges)

        elif edge_feature == 'assembly':
            try:
                edge_index = \
                    torch.tensor([[e[0], e[1]] for e in graph.edges(data=True) if len(e[-1]) == 0]).transpose(1, 0)
            except:
                continue
            for edge in graph.edges(data=True):
                if len(edge[-1]) == 0:
                    edges.append(torch.ones(5))
            edges = torch.stack(edges)

        elif edge_feature == 'none':
            edges = torch.ones((nx.number_of_edges(graph), 5))
            edge_index = torch.tensor([[e[0], e[1]] for e in graph.edges()]).transpose(1, 0)

        # TODO: add material into the data structure - done
        graphs.append(Data(x=nodes, edge_index=edge_index, e=edges, y=y, y1=y1, y2=y2, y3=y3, material=material))

        # print(Data(x=nodes, edge_index=edge_index, e=edges, y=y, y1=y1, y2=y2, y3=y3, material = material))

    # Return: weights not changed, list of new graph Data structure, weights not changed
    return graphs, vocab, weights


class DataSet(object):
    def __init__(self, batch_size, node_feature, edge_feature):
        self.graphs, self.vocab, self.weight = preprocess(node_feature, edge_feature)
        self.batch_size = batch_size
        self.node_dim = self.graphs[0].x.shape[-1]
        self.edge_dim = self.graphs[0].e.shape[-1]
        self.num_class_l1 = len(self.vocab['tier_1_function'])
        self.num_class_l2 = len(self.vocab['tier_2_function'])
        self.num_class_l3 = len(self.vocab['tier_3_function'])
        self.num_materials = len(self.vocab['material_name'])
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.shuffle()

    def shuffle(self):  # shuffle and split the list of graphs

        # train, test = train_test_split(self.graphs, test_size=.3, shuffle=True)
        # train, val = train_test_split(self.graphs, test_size=.1,
        #                               shuffle=True)  # TODO: change train_test_split to split on train set

        train, test_val = train_test_split(self.graphs, test_size=0.4, shuffle=True) # Train = 0.6
        val, test = train_test_split(test_val, test_size=0.75, shuffle=True)
        # Val = 0.4 * 0.25 = 0.1
        # Test = 0.4 * 0.75 = 0.3

        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)


if __name__ == '__main__':
    # preprocess("all", "all")
    draw_class()
