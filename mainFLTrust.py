
import numpy as np
import argparse
import importlib
import random
import os
import sys

from Models.alexnet import AlexNet
from Models.models import *
from Models.svhn import svhn
from Models.resnet import *
from Models.basic_nets import *
from Server.ServerFLTrust import FLTrust


import torch
torch.manual_seed(0)

def main(dataset, algorithm,batch_size, learning_rate,num_glob_iters,
         local_epochs, numusers, times, n_attackers,attacker_type,iid,select_users_ratio,datadir,beta,partition,localstep):

    # Get device status: Check GPU or CPU
    device = torch.device("cuda")

    for i in range(times):
        print("---------------Running time:------------",i)
        if dataset=="cifar10":
            if numusers==50:
                model = ResNet18().to(device)
            elif numusers==500:
                model= ResNet(8, n_classes=10).to(device)
            else:
                model = ResNet(8, n_classes=10).to(device)
        if dataset=="svhn":
            if numusers==50:
                model = ResNet18().to(device)
            elif numusers==500:
                model= ResNet(8, n_classes=10).to(device)
            else:
                model = ResNet(8, n_classes=10).to(device)
        # select algorithm
        if(algorithm == "FLTrust"):
            server = FLTrust(device, dataset, algorithm, model, batch_size, learning_rate,num_glob_iters, local_epochs,n_attackers,attacker_type, iid,int(numusers*select_users_ratio),datadir,beta,partition,numusers,localstep)
        server.train()


    # Average data 
    #average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=lamda,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["MNIST", "synthetic", "cifar10","Cifar100","svhn","FashionMNIST"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Local learning rate")
    parser.add_argument("--num_global_iters", type=int, default=1000)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--algorithm", type=str, default="FLTrust")
    parser.add_argument("--numusers", type=int, default=10, help="Number of Users per round")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--attacker_type", type=str, default="LIE", choices=["LIE","Fang","NDSS","Mimic","MinMax","MinSum","LabelFlip","DynamicLabelFlip"])
    parser.add_argument("--n_attackers", type=int, default=0, help="The number of attackers")
    parser.add_argument("--num_GoodUsers", type=int, default=0, help="The number of Good Users in the system")
    parser.add_argument("--iid", type=bool, default=True, help="IID or NonIID")
    parser.add_argument("--select_users_ratio", type=float, default=1)
    parser.add_argument("--datadir", type=str, default="dataset/cifar10_data")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--partition", type=str, default="noniid-labeldir")
    parser.add_argument("--localstep", type=int, default=5)
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Number of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Attacker Type       : {}".format(args.attacker_type))
    print("Number of attackers       : {}".format(args.n_attackers))
    print("Number of Good Users       : {}".format(args.num_GoodUsers))
    print("IID       : {}".format(args.iid))
    print("Select users ratio       : {}".format(args.select_users_ratio))
    print("Data directory       : {}".format(args.datadir))
    print("Beta of Dirichlet       : {}".format(args.beta))
    print("Partition of data       : {}".format(args.partition))
    print("Local Steps       : {}".format(args.localstep))
    print("=" * 80)

    main(
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        numusers = args.numusers,
        times = args.times,
        n_attackers=args.n_attackers,
        attacker_type=args.attacker_type,
        iid=args.iid,
        select_users_ratio=args.select_users_ratio,
        datadir=args.datadir,
        beta=args.beta,
        partition=args.partition,
        localstep=args.localstep
        )
