import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import numpy as np
import time
import random
import argparse
import dataset
import IPython
import json
from sklearn.metrics import f1_score
from collections import defaultdict

from encoders import Encoder
from aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSageConcat(nn.Module):
    def __init__(self, num_classes, enc1, enc2):
        super(SupervisedGraphSageConcat, self).__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc1.embed_dim + enc2.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        self.enc2(nodes)
        embeds = torch.cat([self.enc1.forward_result, self.enc2.forward_result], axis=0)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

class SupervisedGraphSageConcat2(nn.Module):
    def __init__(self, num_classes, enc1, enc2):
        super(SupervisedGraphSageConcat2, self).__init__()
        self.enc1 = enc1
        self.enc2 = enc2
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc1.embed_dim + enc2.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        self.enc2(nodes)
        embeds = torch.cat([self.enc1.forward_result, self.enc2.forward_result], axis=0)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def accuracy(output, labels):
    preds = output.detach().numpy().argmax(1)
    correct = (preds == labels.flatten())
    correct = correct.sum()
    return correct / len(labels)

def run_graphsage(feat_data, labels, adj_lists, train, val, test, num_classes,
                  model_class=SupervisedGraphSage):
    np.random.seed(1)
    random.seed(1)
    num_nodes = feat_data.shape[0]
    # feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    if args.cuda:
        features.cuda()

    if model_class == SupervisedGraphSageConcat2:
        raise NotImplementedError()
        # The code seems to be not working...
        linear_embed_weights = nn.Parameter(
                torch.FloatTensor(feat_data.shape[1], args.hid_units), requires_grad=True)
        init.xavier_uniform(linear_embed_weights)
        features.weight = nn.Parameter(features.weight.mm(linear_embed_weights), requires_grad=False)
    
    agg1 = MeanAggregator(features, cuda=args.cuda, gcn=args.gcn_aggregator)
    enc1 = Encoder(features, features.weight.shape[1], args.hid_units, adj_lists, agg1, gcn=args.gcn_encoder, cuda=args.cuda)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=args.cuda, gcn=args.gcn_aggregator)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, args.hid_units, adj_lists, agg2,
            base_model=enc1, gcn=args.gcn_encoder, cuda=args.cuda)
    enc1.num_samples = args.num_samples[0]
    enc2.num_samples = args.num_samples[1]


    if model_class == SupervisedGraphSageConcat:
        graphsage = model_class(num_classes, enc1, enc2)
    elif model_class == SupervisedGraphSageConcat2:
        graphsage = model_class(num_classes, enc1, enc2)
    else:
        graphsage = model_class(num_classes, enc2)
    if args.cuda:
        graphsage.cuda()
    
    optimizer = torch.optim.SGD([p for p in graphsage.parameters() if p.requires_grad], lr=args.lr)
    times = []
    record_dict = dict()
    best_val_record_dict = None

    for batch in range(args.epochs):
        batch_nodes = train[:args.batch_size]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        
        train_acc = accuracy(graphsage.forward(train), labels[train])
        val_acc = accuracy(graphsage.forward(val), labels[val])
        test_acc = accuracy(graphsage.forward(test), labels[test])
        print(batch, loss.data, train_acc, val_acc, test_acc)
        record_dict.update(dict(
            epoch=int(batch + 1), train_loss=float(loss.data), train_acc=float(train_acc),
            val_acc=float(val_acc), test_accuracy=float(test_acc), time=str(end_time-start_time), early_stopping=False
        ))

        if (best_val_record_dict is None) or (record_dict["val_acc"] >= best_val_record_dict["val_acc"]):
            best_val_record_dict = record_dict.copy()

    val_output = graphsage.forward(val) 
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))
    print(best_val_record_dict)

    if args.use_signac:
        with open(job.fn("results.json"), "w") as f:
            json.dump(best_val_record_dict, f)
        print("Results recorded to {}".format(job.fn("results.json")))
        job.data[f"correct_label"] = labels

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", "--learning_rate", default=0.7, type=float)
    parser.add_argument("--hid_units", default=128, type=int)
    parser.add_argument("--num_samples", nargs="+", default=[5, 5], type=int)
    parser.add_argument("--gcn_encoder", action="store_true")
    parser.add_argument("--gcn_aggregator", action="store_true")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.', dest="_no_cuda")

    parser.add_argument("--dataset", default="ind.cora", type=str)
    parser.add_argument("--run_id", default="", type=str)
    parser.add_argument("--use_signac", default=False, action="store_true")
    parser.add_argument("--signac_root", default=None, dest="_signac_root")
    parser.add_argument("--dataset_path", default="../../../data/cora", type=str, dest="_dataset_path")
    parser.add_argument("--gpu_limit", default=0, type=float, dest="_gpu_limit")
    parser.add_argument("--debug", action="store_true", dest="_debug")
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--model_class", choices=[
        "SupervisedGraphSage", "SupervisedGraphSageConcat", "SupervisedGraphSageConcat2"],
        default="SupervisedGraphSage")

    args = parser.parse_args()
    if args._debug:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
        ptvsd.wait_for_attach()
        breakpoint()
    
    if args.use_signac:
        import signac
        project = signac.get_project(root=args._signac_root)
        job_dict = {name: value for name, value in vars(args).items() if not name.startswith("_")}
        job = project.open_job(job_dict).init()
    args.cuda = not args._no_cuda and torch.cuda.is_available()

    cora = dataset.PlanetoidData(args.dataset, args._dataset_path, val_size=args.val_size)

    run_graphsage(cora.features.toarray(), cora.labels[:, np.newaxis], cora.dos_graph, 
        np.where(cora.train_mask)[0], np.where(cora.val_mask)[0], np.where(cora.test_mask)[0], cora.num_labels,
        model_class=eval(args.model_class))
