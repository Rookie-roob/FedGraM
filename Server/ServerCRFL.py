import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader

from User.UserAVG import UserAVG
from User.UserLIE import UserLIE
from User.UserFang_new import UserFang
from User.UserNDSS import UserNDSS
from User.UserMinMax import UserMinMax
from User.UserMinSum import UserMinSum
from User.UserLF import UserLF
from User.UserDLF import UserDLF
from Server.Server import Server
from Server.sgd import SGD
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

# Implementation for FedAvg Server

class CRFL(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, num_glob_iters,local_epochs,n_attackers,attacker_type, iid,select_users,datadir,beta,partition,numusers,localstep):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate,
                         num_glob_iters, local_epochs, num_users=numusers, times=1,select_users_num=select_users,datadir=datadir,beta=beta,partition=partition)

        # Initialize data for all  users
        #data = read_data(dataset,iid)
        total_users = self.num_users
        self.n_attackers=n_attackers
        self.benign_users=[]
        self.attackers=[]
        self.optimizer = SGD(self.model.parameters(), lr=1)
        self.criterion=nn.CrossEntropyLoss()
        self.attacker_type=attacker_type
        self.local_server_model = copy.deepcopy(list(self.model.parameters()))
        self.iid = iid
        self.local_epochs=local_epochs
        self.total_train_samples=0
        self.glob_norm=[]
        if dataset=="cifar10":
            self.globalmodel_norm_bound=28
            if self.num_users==50:
                self.globalmodel_norm_bound =107
        if dataset=="svhn":
            self.globalmodel_norm_bound=28
            if self.num_users==50:
                self.globalmodel_norm_bound =107
        if dataset=="mnist":
            if self.num_users==500:
                self.globalmodel_norm_bound=9


        if total_users-n_attackers>0:
            for i in range(0, total_users-n_attackers):
                id=i
                train=self.users_trainset[i]
                #test=self.users_testset[i]
                #id, train,test  = read_user_data(i, data, dataset)
                user = UserAVG(device, id, train, model, batch_size, learning_rate,local_epochs,localstep)
                #user.settest(test)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if n_attackers>0:
            for i in range(total_users-n_attackers,total_users):
                if self.attacker_type=="LIE":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserLIE(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type=="Fang":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserFang(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type=="NDSS":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserNDSS(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type=="MinMax":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserMinMax(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type=="MinSum":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserMinSum(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "LabelFlip":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    classnum=10
                    if self.dataset == "cifar100":
                        classnum=100
                    user = UserLF(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,classnum)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "DynamicLabelFlip":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    classnum=10
                    if self.dataset == "cifar100":
                        classnum=100
                    user = UserDLF(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,classnum)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples


    def update_globalmodel_norm_bound(self):
        if self.dataset=="cifar10" or self.dataset=="svhn":
            if self.num_users==50:
                self.globalmodel_norm_bound=self.globalmodel_norm_bound+22.5
            else:
                self.globalmodel_norm_bound=self.globalmodel_norm_bound+1.5
        if self.dataset=="mnist":
            if self.num_users==500:
                self.globalmodel_norm_bound = self.globalmodel_norm_bound + 0.004



    def get_grads_difference(self):
        grad=[]
        for param_1,param_0 in zip(self.model.parameters(),self.local_server_model):
            param=param_0.data-param_1.data
            #param=param_0.data-param_0.data
            grad=param.data.view(-1) if not len(grad) else torch.cat((grad,param.view(-1)))
        return grad

    def Evaluate(self):
        global best_acc
        self.model.eval()
        #self.model.train()
        device=self.device
        net=self.model
        criterion=self.criterion
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                if self.dataset=="Cifar100":
                    targets=targets.to("cpu").numpy()
                    for i in range(100):
                        targets[i]=self.targetdic.get(targets[i])
                    targets=torch.from_numpy(targets).to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                #loss.backward()
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print("Average Global Accurancy: ", 100.*correct/total)
            print("Average Global Trainning Loss: ",test_loss/100)


    def aggregate_grads_AVG_new(self):

        user_updates = []
        #print("collect benign users' state and turn to 1 dimension vector")
        for user in self.selected_users:
            param = user.get_params()
            global_param = self.get_params()
            param = global_param-param
            user_updates = param[None, :] if len(user_updates) == 0 else torch.cat((user_updates, param[None, :]),
                                                                               0)
        aggregated_update=torch.mean(user_updates,0)

        global_para=self.model.state_dict()
        start_idx = 0
        for key in global_para:
            global_para[key] = global_para[key]-aggregated_update[start_idx:start_idx + len(global_para[key].data.view(-1))].reshape(
                global_para[key].data.shape)
            start_idx = start_idx + len(global_para[key].data.view(-1))
        self.model.load_state_dict(global_para)

    def aggregate_grads_CRFL(self):

        user_updates = []
        #print("collect benign users' state and turn to 1 dimension vector")
        for user in self.selected_users:
            param = user.get_params()
            global_param = self.get_params()
            param = global_param-param
            user_updates = param[None, :] if len(user_updates) == 0 else torch.cat((user_updates, param[None, :]),
                                                                               0)
        aggregated_update=torch.mean(user_updates,0)

        global_para=self.model.state_dict()
        start_idx = 0
        for key in global_para:
            global_para[key] = global_para[key]-aggregated_update[start_idx:start_idx + len(global_para[key].data.view(-1))].reshape(
                global_para[key].data.shape)
            start_idx = start_idx + len(global_para[key].data.view(-1))
        self.model.load_state_dict(global_para)

        global_param = self.get_params()
        global_norm = torch.norm(global_param).cpu().numpy()
        global_param=global_param*np.min((self.globalmodel_norm_bound/global_norm,1))
        if self.dataset=="cifar10":
            if self.num_users == 50:
                perturb=torch.normal(torch.zeros_like(global_param),torch.ones_like(global_param)*0.0001)
            else:
                perturb=torch.normal(torch.zeros_like(global_param),torch.ones_like(global_param)*0.00005)
        if self.dataset=="svhn":
            if self.num_users == 50:
                perturb=torch.normal(torch.zeros_like(global_param),torch.ones_like(global_param)*0.0001)
            else:
                perturb=torch.normal(torch.zeros_like(global_param),torch.ones_like(global_param)*0.00005)

        if self.dataset=="mnist":
            if self.num_users == 500:
                perturb=torch.normal(torch.zeros_like(global_param),torch.ones_like(global_param)*0.00000012)

        global_param=global_param+perturb

        global_para=self.model.state_dict()
        start_idx = 0
        for key in global_para:
            global_para[key] = global_param[start_idx:start_idx + len(global_para[key].data.view(-1))].reshape(
                global_para[key].data.shape)
            start_idx = start_idx + len(global_para[key].data.view(-1))
        self.model.load_state_dict(global_para)



    def train(self):
        loss = []

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            self.model.to(self.device)
            self.send_parameters()
            # Evaluate model each interation
            self.evaluate()

            self.select_users()
            selected_benign_users=[]
            for user in self.selected_users:
                if user in self.benign_users:
                    selected_benign_users.append(user)
                    user.train(self.local_epochs) #* user.train_samples
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    user.set_param_new(selected_benign_users)
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    malicious_state = user.get_malicious_model_state(selected_benign_users)
                    user.load_malicious_state(malicious_state)
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    user.load_malicious_state(malicious_state)
            '''
            #self.aggregate_grads_AVG()
            self.aggregate_grads_CRFL()
            self.update_globalmodel_norm_bound()
        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_results()
        self.save_model()

    
