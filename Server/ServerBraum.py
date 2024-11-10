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

class Braum(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, num_glob_iters, local_epochs,
                 n_attackers, attacker_type, iid, select_users, datadir, beta, partition, numusers, localstep):
        super().__init__(device, dataset, algorithm, model, batch_size, learning_rate,
                         num_glob_iters, local_epochs, num_users=numusers, times=1, select_users_num=select_users,
                         datadir=datadir, beta=beta, partition=partition)

        # Initialize data for all  users
        # data = read_data(dataset,iid)
        total_users = self.num_users
        self.n_attackers = n_attackers
        self.benign_users = []
        self.attackers = []
        self.optimizer = SGD(self.model.parameters(), lr=1)
        self.criterion = nn.CrossEntropyLoss()
        self.attacker_type = attacker_type
        self.local_server_model = copy.deepcopy(list(self.model.parameters()))
        self.iid = iid
        self.local_epochs = local_epochs
        self.total_train_samples = 0
        self.glob_norm = []
        self.save_norms = [] # [attacker's biggest norm, attacker's smallest norm, benign's biggest norm, benign's smallest norm]

        if total_users - n_attackers > 0:
            for i in range(0, total_users - n_attackers):
                id = i
                train = self.users_trainset[i]
                # test=self.users_testset[i]
                # id, train,test  = read_user_data(i, data, dataset)
                user = UserAVG(device, id, train, model, batch_size, learning_rate, local_epochs, localstep)
                # user.settest(test)
                self.benign_users.append(user)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        if n_attackers > 0:
            for i in range(total_users - n_attackers, total_users):
                if self.attacker_type == "LIE":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserLIE(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                   n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "Fang":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserFang(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                    n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "NDSS":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserNDSS(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                    n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "MinMax":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserMinMax(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                      n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "MinSum":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    user = UserMinSum(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                      n_attackers)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "LabelFlip":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    classnum = 10
                    if self.dataset == "cifar100":
                        classnum = 100
                    user = UserLF(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                  classnum)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples
                if self.attacker_type == "DynamicLabelFlip":
                    id = i
                    train = self.users_trainset[i]
                    # test=self.users_testset[i]
                    # id, train,test  = read_user_data(i, data, dataset)
                    classnum = 10
                    if self.dataset == "cifar100":
                        classnum = 100
                    user = UserDLF(device, id, train, model, batch_size, learning_rate, local_epochs, localstep,
                                   classnum)
                    # user.settest(test)
                    self.attackers.append(user)
                    self.users.append(user)
                    self.total_train_samples += user.train_samples

    def get_grads_difference(self):
        grad = []
        for param_1, param_0 in zip(self.model.parameters(), self.local_server_model):
            param = param_0.data - param_1.data
            # param=param_0.data-param_0.data
            grad = param.data.view(-1) if not len(grad) else torch.cat((grad, param.view(-1)))
        return grad

    def Evaluate(self):
        global best_acc
        self.model.eval()
        # self.model.train()
        device = self.device
        net = self.model
        criterion = self.criterion
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                if self.dataset == "Cifar100":
                    targets = targets.to("cpu").numpy()
                    for i in range(100):
                        targets[i] = self.targetdic.get(targets[i])
                    targets = torch.from_numpy(targets).to(self.device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                # loss.backward()
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            print("Average Global Accurancy: ", 100. * correct / total)
            print("Average Global Trainning Loss: ", test_loss / 100)

    def aggregate_grads_AVG(self):

        global_para = self.model.state_dict()
        idx = 0
        total_data_nums = 0
        for user in self.selected_users:
            total_data_nums += user.train_samples
        for user in self.selected_users:
            net_para = user.model.cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] / len(self.selected_users)
            else:
                for key in net_para:
                    global_para[key] += net_para[key] / len(self.selected_users)
            idx = idx + 1
        self.model.load_state_dict(global_para)

    def aggregate_grads_AVG_new(self):

        user_updates = []
        # print("collect benign users' state and turn to 1 dimension vector")
        for user in self.selected_users:
            param = user.get_params()
            global_param = self.get_params()
            param = global_param - param
            user_updates = param[None, :] if len(user_updates) == 0 else torch.cat((user_updates, param[None, :]),
                                                                                   0)
        aggregated_update = torch.mean(user_updates, 0)

        global_para = self.model.state_dict()
        start_idx = 0
        for key in global_para:
            global_para[key] = global_para[key] - aggregated_update[
                                                  start_idx:start_idx + len(global_para[key].data.view(-1))].reshape(
                global_para[key].data.shape)
            start_idx = start_idx + len(global_para[key].data.view(-1))
        self.model.load_state_dict(global_para)

    def calculate_globalmodel_norm(self):
        global_param = self.get_params()
        global_norm = torch.norm(global_param).cpu().numpy()
        print(global_norm)
        self.glob_norm.append(global_norm)

    def save_globalmodel_norm(self):
        alg = self.algorithm
        alg = alg + "_" + str(self.num_users) + "_" + self.attacker_type + "_" + str(
            self.n_attackers) + "_" + self.partition + "_" + str(self.beta)
        alg_acc = alg + "_globnorm"
        np.save(os.path.join("results", alg_acc), self.glob_norm)
    

    def train(self):
        loss = []

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.model.to(self.device)
            self.send_parameters()
            # Evaluate model each interation
            self.evaluate()
            # self.calculate_globalmodel_norm()

            self.select_users()

            selected_benign_users = []
            for user in self.selected_users:
                if user in self.benign_users:
                    selected_benign_users.append(user)
                    user.train(self.local_epochs)  # * user.train_samples
                    # user.train_root(self.local_epochs)  # * user.train_samples


            '''
            for user in self.selected_users:
                if user in self.attackers:
                    user.set_param_new(selected_benign_users)
            '''


            #self.collected_benign_users = selected_benign_users
            for user in self.selected_users:
                if user in self.attackers:
                    malicious_state = user.get_malicious_model_state(selected_benign_users)
                    user.load_malicious_state(malicious_state)

                    #self.collected_malicious_users.append(user)
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    user.load_malicious_state(malicious_state)
            '''
            ret = self.cal_embedding(self.selected_users)
            self.totalret.append(ret)
            self.remove_users_by_norm(remove_num=15,cur_matrix_dict=ret, if_save_norm=True)
            # self.aggregate_grads_AVG()
            self.aggregate_grads_AVG_new()
        alg =self.algorithm
        alg = alg  +  "_" + str(self.num_users)+"_"+self.attacker_type+"_"+str(self.n_attackers)+"_"+self.partition+"_"+str(self.beta)
        write_file=alg+"_matrix.npy"
        self.write_embedding_to_file(write_file, self.totalret)
        # self.read_embedding_file(write_file)

        print("Highest accuracy")
        print(max(self.rs_glob_acc))
        self.save_results()
        # self.save_globalmodel_norm()
        self.save_model()

        norm_file_name = alg+"_savednorm.npy"
        self.save_norm_result(norm_file_name, if_save_norm=True)



    def save_norm_result(self, file_name, if_save_norm):
        if if_save_norm == True:
            np.save(file_name, self.save_norms)
        
    def remove_users_by_norm(self, remove_num, cur_matrix_dict, if_save_norm=True):
        norm_list = []
        for key, value in cur_matrix_dict.items():
            vector = value.reshape(1, -1)
            norm = np.linalg.norm(vector)
            norm_list.append((key, norm))
        sorted_norm_list = sorted(norm_list, key=lambda x: x[1], reverse=True)

        if if_save_norm == True:
            pending_save_norms = []
            for tup in sorted_norm_list:
                user_idx = tup[0]
                if user_idx >= (self.num_users - self.n_attackers):
                    pending_save_norms.append(tup[1])
                    break
            for tup in reversed(sorted_norm_list):
                user_idx = tup[0]
                if user_idx >= (self.num_users - self.n_attackers):
                    pending_save_norms.append(tup[1])
                    break
            for tup in sorted_norm_list:
                user_idx = tup[0]
                if user_idx < (self.num_users - self.n_attackers):
                    pending_save_norms.append(tup[1])
                    break
            for tup in reversed(sorted_norm_list):
                user_idx = tup[0]
                if user_idx < (self.num_users - self.n_attackers):
                    pending_save_norms.append(tup[1])
                    break
            self.save_norms.append(pending_save_norms)

        cur_num = 0
        for tup in sorted_norm_list:
            user_idx = tup[0]
            for user in self.selected_users:
                if user_idx == user.id:
                    self.selected_users.remove(user)
            cur_num = cur_num + 1
            if cur_num == remove_num:
                break



def showdistribution(target):
    targets = target
    a = np.zeros(10)
    for i in targets:
        a[i] = a[i] + 1
    print(a / np.sum(a))

