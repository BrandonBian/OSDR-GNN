import sys

sys.path.append("..")
import os
import time
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from GNN.dataset import DataSet
from itertools import product
from gnn import HierarchicalGNN, MLP, Linear, CustomizedGNN
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

material = True
ITERATION = 50


class HierarchicalClassifier(object):
    def __init__(self, args):
        self.verbose = args.verbose
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_layers = args.num_layers
        self.num_class_l1 = args.num_class_l1
        self.num_class_l2 = args.num_class_l2
        self.num_class_l3 = args.num_class_l3
        self.patience = args.patience
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.hid_dim = args.hid_dim
        self.lr = args.lr

        if args.network == 'mlp':
            self.model = MLP(self.node_dim, self.hid_dim, self.num_class_l1,
                             self.num_class_l2, self.num_class_l3).to(self.device)
        elif args.network == 'linear':
            self.model = Linear(self.node_dim, self.hid_dim, self.num_class_l1,
                                self.num_class_l2, self.num_class_l3).to(self.device)
        else:  # Default is here
            self.model = HierarchicalGNN(self.node_dim, self.edge_dim, self.hid_dim,
                                         self.num_class_l1, self.num_class_l2, self.num_class_l3,
                                         self.num_layers, args.network).to(self.device)

    def load(self):
        if os.path.exists('checkpoint.pkl'):
            self.model.load_state_dict(torch.load('checkpoint.pkl'))
        else:
            raise Exception('Checkpoint not found ...')

    def train(self, train_loader, val_loader, weights):
        best_loss, best_state, patience_count = 1e9, self.model.state_dict(), 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # Adam Optimizer

        # scheduler: change the attributes of your neural network such as weights and learning rate in order to reduce the losses
        # adjust learning rate: Set the learning rate of each parameter group using a cosine annealing schedule

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        weights_l1 = weights[1].to(self.device)  # tier 1 function weights
        weights_l2 = weights[2].to(self.device)  # tier 2 function weights
        weights_l3 = weights[3].to(self.device)  # tier 3 function weights

        for epoch in range(self.num_epochs):
            self.model.train()  # set to train mode
            epoch_loss = 0.
            start = time.time()
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                # gnn.forward() parameters:
                # x = batch.x.float() - node attributes
                # edge_index = batch.edge_index - 2 nodes to each edge
                # e = batch.e.float() - edge attributes
                # y1 = F.one_hot(batch.y1, self.num_class_l1) - tier 1 function ground truth
                # y2 = F.one_hot(batch.y2, self.num_class_l2) - tier 2 function ground truth

                # compute tier predictions

                if args.network in ('mlp', 'linear'):
                    logits_l1, logits_l2, logits_l3 = self.model(
                        batch.x.float(), F.one_hot(batch.y1, self.num_class_l1), F.one_hot(batch.y2, self.num_class_l2))
                else:  # by default
                    logits_l1, logits_l2, logits_l3 = self.model(
                        batch.x.float(), batch.edge_index, batch.e.float(),
                        F.one_hot(batch.y1, self.num_class_l1),
                        F.one_hot(batch.y2, self.num_class_l2))

                # compute joint loss across all tiers (with ground truth)

                is_labeled = batch.y1 > 0  # It actually has a ground truth
                loss1 = nn.CrossEntropyLoss(weight=weights_l1)(logits_l1[is_labeled], batch.y1[is_labeled])
                is_labeled = batch.y2 > 0
                loss2 = nn.CrossEntropyLoss(weight=weights_l2)(logits_l2[is_labeled], batch.y2[is_labeled])
                is_labeled = batch.y3 > 0
                loss3 = nn.CrossEntropyLoss(weight=weights_l3)(logits_l3[is_labeled], batch.y3[is_labeled])
                loss = loss1 + loss2 + loss3
                epoch_loss += loss.item()
                loss.backward()  # back propagation (compute gradients and update parameters)
                optimizer.step()  # optimizer takes one step

            scheduler.step()  # scheduler takes one step
            end = time.time()
            val_loss, _, _, _, _, _, _ = self.predict(val_loader)  # evaluate and obtain loss on validation set

            if self.verbose:
                print(f'Epoch: {epoch + 1:03d}/{self.num_epochs}, Time: {end - start:.2f}s, '
                      f'Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss: .4f}')

            if best_loss > val_loss:  # if this state better than previous best, store it
                best_loss = val_loss
                best_state = self.model.state_dict()
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == self.patience:
                if self.verbose:
                    print('Early stopping ...')
                break

        self.model.load_state_dict(best_state)
        print("Saving the model...")
        torch.save(best_state, 'checkpoint.pkl')

    @torch.no_grad()
    def predict(self, data_loader):  # data_loader is the testing/validation set
        self.model.eval()  # set to evaluation

        loss = 0.
        yp_l1, yp_l2, yp_l3 = [], [], []
        yt_l1, yt_l2, yt_l3 = [], [], []

        for batch in data_loader:
            batch = batch.to(self.device)
            # x = node attributes; edge_index = 2 nodes to each edge; e = edge attributes
            if args.network in ('mlp', 'linear'):
                logits_l1, logits_l2, logits_l3 = self.model.predict(batch.x.float())
            else:
                logits_l1, logits_l2, logits_l3 = self.model.predict(batch.x.float(), batch.edge_index, batch.e.float())

            is_labeled = batch.y1 > 0
            loss1 = nn.CrossEntropyLoss()(logits_l1[is_labeled],
                                          batch.y1[is_labeled])  # compare predicted with ground truth

            is_labeled = batch.y2 > 0
            loss2 = nn.CrossEntropyLoss()(logits_l2[is_labeled], batch.y2[is_labeled])

            is_labeled = batch.y3 > 0
            loss3 = nn.CrossEntropyLoss()(logits_l3[is_labeled], batch.y3[is_labeled])

            loss += (loss1 + loss2 + loss3).item()

            yp_l1.append(torch.argmax(logits_l1, dim=-1))
            yp_l2.append(torch.argmax(logits_l2, dim=-1))
            yp_l3.append(torch.argmax(logits_l3, dim=-1))
            yt_l1.append(batch.y1)
            yt_l2.append(batch.y2)
            yt_l3.append(batch.y3)

        loss /= len(data_loader)
        yp_l1 = torch.cat(yp_l1, -1)
        yp_l2 = torch.cat(yp_l2, -1)
        yp_l3 = torch.cat(yp_l3, -1)
        yt_l1 = torch.cat(yt_l1, -1)
        yt_l2 = torch.cat(yt_l2, -1)
        yt_l3 = torch.cat(yt_l3, -1)

        return loss, yp_l1, yp_l2, yp_l3, yt_l1, yt_l2, yt_l3


class CustomizedClassifier(object):
    def __init__(self, args):
        self.verbose = args.verbose
        self.device = torch.device(args.device)
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.num_layers = args.num_layers
        self.num_class_l1 = args.num_class_l1
        self.num_class_l2 = args.num_class_l2
        self.num_class_l3 = args.num_class_l3

        # TODO: add num_materials - done
        self.num_materials = args.num_materials

        self.patience = args.patience
        self.node_dim = args.node_dim
        self.edge_dim = args.edge_dim
        self.hid_dim = args.hid_dim
        self.lr = args.lr

        if args.network == 'mlp':
            self.model = MLP(self.node_dim, self.hid_dim, self.num_class_l1,
                             self.num_class_l2, self.num_class_l3).to(self.device)
        elif args.network == 'linear':
            self.model = Linear(self.node_dim, self.hid_dim, self.num_class_l1,
                                self.num_class_l2, self.num_class_l3).to(self.device)
        else:  # Default is here

            # TODO: change to customized GNN model - done

            self.model = CustomizedGNN(self.node_dim, self.edge_dim, self.hid_dim,
                                       self.num_materials,
                                       self.num_layers, args.network).to(self.device)

    def load(self):
        if os.path.exists('checkpoint.pkl'):
            self.model.load_state_dict(torch.load('checkpoint.pkl'))
        else:
            raise Exception('Checkpoint not found ...')

    def train(self, train_loader, val_loader, weights):
        best_loss, best_state, patience_count = 1e9, self.model.state_dict(), 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)  # Adam Optimizer

        # scheduler: change the attributes of your neural network such as weights and learning rate in order to reduce the losses
        # adjust learning rate: Set the learning rate of each parameter group using a cosine annealing schedule

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)

        # TODO: Extract the weights of material from the weights dictionary - done
        weights_material = weights[4].to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()  # set to train mode
            epoch_loss = 0.
            start = time.time()
            for batch in train_loader:

                batch = batch.to(self.device)
                optimizer.zero_grad()

                if args.network in ('mlp', 'linear'):
                    logits_l1, logits_l2, logits_l3 = self.model(
                        batch.x.float(), F.one_hot(batch.y1, self.num_class_l1), F.one_hot(batch.y2, self.num_class_l2))
                else:  # by default

                    # TODO: changed to with no ground truth for training - done
                    material_predictions = self.model(
                        batch.x.float(), batch.edge_index, batch.e.float())

                # compute loss (with ground truth)

                # TODO: calculate loss for material only - done

                is_labeled = batch.material > 0  # It actually has a ground truth
                loss = nn.CrossEntropyLoss(weight=weights_material)(material_predictions[is_labeled],
                                                                    batch.material[is_labeled])

                epoch_loss += loss.item()
                loss.backward()  # back propagation (compute gradients and update parameters)
                optimizer.step()  # optimizer takes one step

            scheduler.step()  # scheduler takes one step
            end = time.time()
            val_loss, _, _ = self.predict(val_loader, weights_material)  # evaluate and obtain loss on validation set

            if self.verbose:
                print(f'Epoch: {epoch + 1:03d}/{self.num_epochs}, Time: {end - start:.2f}s, '
                      f'Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss: .4f}')

            if best_loss > val_loss:  # if this state better than previous best, store it
                best_loss = val_loss
                best_state = self.model.state_dict()
                patience_count = 0
            else:
                patience_count += 1

            if patience_count == self.patience:
                if self.verbose:
                    print('Early stopping ...')
                break

        self.model.load_state_dict(best_state)
        print("Saving the model...")
        torch.save(best_state, 'checkpoint.pkl')

    @torch.no_grad()
    def predict(self, data_loader, weights_material):  # data_loader is the testing/validation set
        self.model.eval()  # set to evaluation

        loss = 0.

        # TODO: only need prediction and ground truths of the material - done
        material_p, material_t = [], []

        for batch in data_loader:
            batch = batch.to(self.device)
            # x = node attributes; edge_index = 2 nodes to each edge; e = edge attributes
            if args.network in ('mlp', 'linear'):  # ignore this
                logits_l1, logits_l2, logits_l3 = self.model.predict(batch.x.float())
            else:
                material_predictions = self.model.predict(batch.x.float(), batch.edge_index, batch.e.float())

            is_labeled = batch.material > 0
            loss = nn.CrossEntropyLoss()(material_predictions[is_labeled],  # TODO: need weights or not?
                                         batch.material[is_labeled])  # compare predicted with ground truth

            # TODO: predictions and ground truths for materials only - done
            material_p.append(torch.argmax(material_predictions, dim=-1))
            material_t.append(batch.material)

        loss /= len(data_loader)

        material_p = torch.cat(material_p, -1)
        material_t = torch.cat(material_t, -1)

        return loss, material_p, material_t


def cross_validate(args):
    dataset = DataSet(args.batch_size, args.node_feature, args.edge_feature)
    args.node_dim = dataset.node_dim
    args.edge_dim = dataset.edge_dim
    args.num_class_l1 = dataset.num_class_l1
    args.num_class_l2 = dataset.num_class_l2
    args.num_class_l3 = dataset.num_class_l3

    # # TODO: add num_materials
    # args.num_materials = dataset.num_materials

    # print("node_dim: ", dataset.node_dim)  # 316
    # print("edge_dim: ", dataset.edge_dim)  # 75
    # print("num_class_l1: ", dataset.num_class_l1)  # 9
    # print("num_class_l2: ", dataset.num_class_l2)  # 22
    # print("num_class_l3: ", dataset.num_class_l3)  # 23

    result = {  # Store the results to be put into the test log
        'tier1': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        },
        'tier2': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        },
        'tier3': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        },
    }

    print("Start Training...")

    for __ in tqdm(range(ITERATION), unit_scale=True, desc='Running experiments...'):

        # Iterating multiple times:
        # This allows us to investigate the model's performance without bias towards train/test splits.

        classifier = HierarchicalClassifier(args)
        classifier.train(dataset.train_loader, dataset.val_loader, dataset.weight)  # Train on train dataset

        print("Start Testing...")
        loss, yp_l1, yp_l2, yp_l3, yt_l1, yt_l2, yt_l3 = classifier.predict(dataset.test_loader)  # Test on test dataset
        yp_l1 = yp_l1[yt_l1 > 0].cpu().numpy()
        yp_l2 = yp_l2[yt_l2 > 0].cpu().numpy()
        yp_l3 = yp_l3[yt_l3 > 0].cpu().numpy()
        yt_l1 = yt_l1[yt_l1 > 0].cpu().numpy()
        yt_l2 = yt_l2[yt_l2 > 0].cpu().numpy()
        yt_l3 = yt_l3[yt_l3 > 0].cpu().numpy()

        for measure in ['micro', 'macro', 'weighted']:  # store the data
            result[f'tier1']['f1'][measure]['data'].append(f1_score(yp_l1, yt_l1, average=measure, zero_division=0))
            result[f'tier2']['f1'][measure]['data'].append(f1_score(yp_l2, yt_l2, average=measure, zero_division=0))
            result[f'tier3']['f1'][measure]['data'].append(f1_score(yp_l3, yt_l3, average=measure, zero_division=0))
            result[f'tier1']['precision'][measure]['data'].append(
                precision_score(yp_l1, yt_l1, average=measure, zero_division=0))
            result[f'tier2']['precision'][measure]['data'].append(
                precision_score(yp_l2, yt_l2, average=measure, zero_division=0))
            result[f'tier3']['precision'][measure]['data'].append(
                precision_score(yp_l3, yt_l3, average=measure, zero_division=0))
            result[f'tier1']['recall'][measure]['data'].append(
                recall_score(yp_l1, yt_l1, average=measure, zero_division=0))
            result[f'tier2']['recall'][measure]['data'].append(
                recall_score(yp_l2, yt_l2, average=measure, zero_division=0))
            result[f'tier3']['recall'][measure]['data'].append(
                recall_score(yp_l3, yt_l3, average=measure, zero_division=0))

        torch.cuda.empty_cache()
        dataset.shuffle()  # shuffle the entire dataset

    for measure in ['micro', 'macro', 'weighted']:  # calculate mean and std
        result[f'tier1']['f1'][measure]['mean'] = np.mean(result[f'tier1']['f1'][measure]['data'])
        result[f'tier2']['f1'][measure]['mean'] = np.mean(result[f'tier2']['f1'][measure]['data'])
        result[f'tier3']['f1'][measure]['mean'] = np.mean(result[f'tier3']['f1'][measure]['data'])
        result[f'tier1']['f1'][measure]['std'] = np.std(result[f'tier1']['f1'][measure]['data'])
        result[f'tier2']['f1'][measure]['std'] = np.std(result[f'tier2']['f1'][measure]['data'])
        result[f'tier3']['f1'][measure]['std'] = np.std(result[f'tier3']['f1'][measure]['data'])

        result[f'tier1']['precision'][measure]['mean'] = np.mean(result[f'tier1']['precision'][measure]['data'])
        result[f'tier2']['precision'][measure]['mean'] = np.mean(result[f'tier2']['precision'][measure]['data'])
        result[f'tier3']['precision'][measure]['mean'] = np.mean(result[f'tier3']['precision'][measure]['data'])
        result[f'tier1']['precision'][measure]['std'] = np.std(result[f'tier1']['precision'][measure]['data'])
        result[f'tier2']['precision'][measure]['std'] = np.std(result[f'tier2']['precision'][measure]['data'])
        result[f'tier3']['precision'][measure]['std'] = np.std(result[f'tier3']['precision'][measure]['data'])

        result[f'tier1']['recall'][measure]['mean'] = np.mean(result[f'tier1']['recall'][measure]['data'])
        result[f'tier2']['recall'][measure]['mean'] = np.mean(result[f'tier2']['recall'][measure]['data'])
        result[f'tier3']['recall'][measure]['mean'] = np.mean(result[f'tier3']['recall'][measure]['data'])
        result[f'tier1']['recall'][measure]['std'] = np.std(result[f'tier1']['recall'][measure]['data'])
        result[f'tier2']['recall'][measure]['std'] = np.std(result[f'tier2']['recall'][measure]['data'])
        result[f'tier3']['recall'][measure]['std'] = np.std(result[f'tier3']['recall'][measure]['data'])

    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    with open(f'logs/{hash(str(args))}.json', 'w') as f:  # save the test results
        json.dump({**args.__dict__, **result}, f, indent=2)


def cross_validate_material(args):
    dataset = DataSet(args.batch_size, args.node_feature, args.edge_feature)
    args.node_dim = dataset.node_dim
    args.edge_dim = dataset.edge_dim
    args.num_class_l1 = dataset.num_class_l1
    args.num_class_l2 = dataset.num_class_l2
    args.num_class_l3 = dataset.num_class_l3

    # TODO: add num_materials argument - done
    args.num_materials = dataset.num_materials

    # TODO: modified the result log's dictionary - done
    result = {  # Store the results to be put into the test log
        'material test results': {
            'f1': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'precision': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            },
            'recall': {
                'macro': {'mean': .0, 'std': .0, 'data': []},
                'micro': {'mean': .0, 'std': .0, 'data': []},
                'weighted': {'mean': .0, 'std': .0, 'data': []},
            }
        }
    }

    print("[Customized] Start Training...")

    for __ in tqdm(range(ITERATION), unit_scale=True, desc='Running experiments...'):

        # Iterating multiple times:
        # This allows us to investigate the model's performance without bias towards train/test splits.

        # TODO: create a customized classifier and initialize an instance of it - done

        classifier = CustomizedClassifier(args)
        classifier.train(dataset.train_loader, dataset.val_loader, dataset.weight)  # Train on train dataset

        print("[Customized] Start Testing...")
        weights_material = dataset.weight[4].to('cuda:0')
        loss, material_p, material_t = classifier.predict(dataset.test_loader, weights_material)  # Test on test dataset

        material_p = material_p[material_t > 0].cpu().numpy()
        material_t = material_t[material_t > 0].cpu().numpy()

        # TODO: store the test results of the material predictions - done
        for measure in ['micro', 'macro', 'weighted']:  # store the data
            result[f'material test results']['f1'][measure]['data'].append(
                f1_score(material_p, material_t, average=measure, zero_division=0))

            result[f'material test results']['precision'][measure]['data'].append(
                precision_score(material_p, material_t, average=measure, zero_division=0))

            result[f'material test results']['recall'][measure]['data'].append(
                recall_score(material_p, material_t, average=measure, zero_division=0))

        torch.cuda.empty_cache()
        dataset.shuffle()  # shuffle the entire dataset

    for measure in ['micro', 'macro', 'weighted']:  # calculate mean and std across all iterations
        result[f'material test results']['f1'][measure]['mean'] = np.mean(
            result[f'material test results']['f1'][measure]['data'])

        result[f'material test results']['f1'][measure]['std'] = np.std(
            result[f'material test results']['f1'][measure]['data'])

        result[f'material test results']['precision'][measure]['mean'] = np.mean(
            result[f'material test results']['precision'][measure]['data'])

        result[f'material test results']['precision'][measure]['std'] = np.std(
            result[f'material test results']['precision'][measure]['data'])

        result[f'material test results']['recall'][measure]['mean'] = np.mean(
            result[f'material test results']['recall'][measure]['data'])

        result[f'material test results']['recall'][measure]['std'] = np.std(
            result[f'material test results']['recall'][measure]['data'])

    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    with open(f'logs/{hash(str(args))}.json', 'w') as f:  # save the test results
        json.dump({**args.__dict__, **result}, f, indent=2)


def draw_confusion_material(args):
    import seaborn as sn
    import matplotlib.pyplot as plt
    dataset = DataSet(args.batch_size, args.node_feature, args.edge_feature)
    args.node_dim = dataset.node_dim
    args.edge_dim = dataset.edge_dim
    args.num_class_l1 = dataset.num_class_l1
    args.num_class_l2 = dataset.num_class_l2
    args.num_class_l3 = dataset.num_class_l3

    yp_l1_all = []
    yt_l1_all = []

    # TODO: add num_materials argument - done
    args.num_materials = dataset.num_materials

    for __ in tqdm(range(ITERATION), unit_scale=True, desc='Running experiments...'):
        # TODO: create a customized classifier and initialize an instance of it - done

        classifier = CustomizedClassifier(args)
        classifier.train(dataset.train_loader, dataset.val_loader, dataset.weight)  # Train on train dataset

        loss, material_p, material_t = classifier.predict(dataset.test_loader)  # Test on test dataset

        yp_l1_all.append(material_p[material_t > 0].cpu().numpy())

        yt_l1_all.append(material_t[material_t > 0].cpu().numpy())

        torch.cuda.empty_cache()
        dataset.shuffle()

    yt_l1_all = np.concatenate(yt_l1_all)
    yp_l1_all = np.concatenate(yp_l1_all)

    cf_l1 = confusion_matrix(yt_l1_all[yp_l1_all > 0], yp_l1_all[yp_l1_all > 0], normalize='true')

    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    plt.figure(figsize=(12, 9))
    label = list(dataset.vocab['material_name'].keys())[1:]
    sn.heatmap(cf_l1, annot=False, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.xticks(size='xx-large', rotation=45)
    plt.yticks(size='xx-large', rotation=45)
    plt.tight_layout()
    plt.savefig(fname='logs/tier1.pdf', format='pdf')
    plt.show()


def draw_confusion(args):
    import seaborn as sn
    import matplotlib.pyplot as plt

    dataset = DataSet(args.batch_size, args.node_feature, args.edge_feature)
    args.node_dim = dataset.node_dim
    args.edge_dim = dataset.edge_dim
    args.num_class_l1 = dataset.num_class_l1
    args.num_class_l2 = dataset.num_class_l2
    args.num_class_l3 = dataset.num_class_l3

    yp_l1_all, yp_l2_all, yp_l3_all = [], [], []
    yt_l1_all, yt_l2_all, yt_l3_all = [], [], []

    for __ in tqdm(range(ITERATION), unit_scale=True, desc='Running experiments...'):
        classifier = HierarchicalClassifier(args)
        classifier.train(dataset.train_loader, dataset.val_loader, dataset.weight)
        loss, yp_l1, yp_l2, yp_l3, yt_l1, yt_l2, yt_l3 = classifier.predict(dataset.test_loader)
        yp_l1_all.append(yp_l1[yt_l1 > 0].cpu().numpy())
        yp_l2_all.append(yp_l2[yt_l2 > 0].cpu().numpy())
        yp_l3_all.append(yp_l3[yt_l3 > 0].cpu().numpy())
        yt_l1_all.append(yt_l1[yt_l1 > 0].cpu().numpy())
        yt_l2_all.append(yt_l2[yt_l2 > 0].cpu().numpy())
        yt_l3_all.append(yt_l3[yt_l3 > 0].cpu().numpy())
        torch.cuda.empty_cache()
        dataset.shuffle()

    yt_l1_all = np.concatenate(yt_l1_all)
    yp_l1_all = np.concatenate(yp_l1_all)
    yt_l2_all = np.concatenate(yt_l2_all)
    yp_l2_all = np.concatenate(yp_l2_all)
    yt_l3_all = np.concatenate(yt_l3_all)
    yp_l3_all = np.concatenate(yp_l3_all)

    cf_l1 = confusion_matrix(yt_l1_all[yp_l1_all > 0], yp_l1_all[yp_l1_all > 0], normalize='true')
    cf_l2 = confusion_matrix(yt_l2_all[yp_l2_all > 0], yp_l2_all[yp_l2_all > 0], normalize='true')
    cf_l3 = confusion_matrix(yt_l3_all[yp_l3_all > 0], yp_l3_all[yp_l3_all > 0], normalize='true')

    if not os.path.exists('logs/'):
        os.makedirs('logs/')

    plt.figure(figsize=(12, 9))
    label = list(dataset.vocab['tier_1_function'].keys())[1:]
    sn.heatmap(cf_l1, annot=False, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.xticks(size='xx-large', rotation=45)
    plt.yticks(size='xx-large', rotation=45)
    plt.tight_layout()
    plt.savefig(fname='logs/tier1.pdf', format='pdf')
    plt.show()

    plt.figure(figsize=(12, 9))
    label = list(dataset.vocab['tier_2_function'].keys())[1:]
    sn.heatmap(cf_l2, annot=False, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.xticks(size='xx-large')
    plt.yticks(size='xx-large')
    plt.tight_layout()
    plt.savefig(fname='logs/tier2.pdf', format='pdf')
    plt.show()

    plt.figure(figsize=(12, 9))
    label = list(dataset.vocab['tier_3_function'].keys())[1:]
    sn.heatmap(cf_l3, annot=False, fmt='.2f', cmap='Blues', xticklabels=label, yticklabels=label)
    plt.xticks(size='xx-large')
    plt.yticks(size='xx-large')
    plt.tight_layout()
    plt.savefig(fname='logs/tier3.pdf', format='pdf')
    plt.show()


def search(args):
    print("Beginning Grid Search...")
    if args.network in ('mlp', 'linear'):
        for h in [64, 128, 256]:
            args.hid_dim = h
            cross_validate_material(args)
    else:  # Default
        grid = [[64, 128, 256], [1, 2, 3]]
        for c in product(*grid):
            args.hid_dim = c[0]
            args.num_layers = c[1]
            cross_validate_material(args)


def get_parser():
    # Fine-tune results:
    # Network: SAGE; num_layers: 1; hidden_dim: 256
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='sage', choices=['gcn', 'gat', 'gin', 'sage', 'mlp', 'linear'])

    parser.add_argument('--node_feature', type=str, default='all',  # TODO: Change to perform ablation study
                        choices=['all', 'none', 'component', 'name', 'type', 'material', 'tier_function'])
    parser.add_argument('--edge_feature', type=str, default='all', choices=['all', 'none', 'flow', 'assembly'])

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda:0')  # change to cuda:0, default is cuda:3
    parser.add_argument('--node_dim', type=int)
    parser.add_argument('--edge_dim', type=int)
    parser.add_argument('--num_class_l1', type=int)
    parser.add_argument('--num_class_l2', type=int)
    parser.add_argument('--num_class_l3', type=int)
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    # TODO: separate the original tier function with customized material - done
    if material == False:
        print("Training for [Tier Functions]")
        cross_validate(args)
        # draw_confusion(args)
    else:
        print("Training for [Materials]")
        cross_validate_material(args)
        # draw_confusion_material(args)

        # args.device = 'cuda:0'
        # args.verbose = True
        # # for network in ['sage', 'gin']:
        # for network in ['gin']:
        #     args.network = network
        #     search(args)

    print("Program finished normally.")
