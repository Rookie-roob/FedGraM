import torch
import os
import torch.nn as nn
from User.UserAVG import UserAVG
from Server.Server import Server
from User.UserLIE import UserLIE

from Server.sgd import SGD
import torchvision
import torchvision.transforms as transforms
import numpy as np
import copy

# Implementation for FedAvg Server

class Krum(Server):
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
        self.multi_k=False
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
    def aggregate_grads_Krum(self):
        assert (self.users is not None and len(self.users) > 0)
        users_param = []
        for user in self.selected_users:
            param = user.get_params()
            users_param = param[None, :] if len(users_param) == 0 else torch.cat((users_param, param[None, :]),
                                                                                 0)

        global_param,_ = self.multi_krum(users_param, int(self.select_users_num * 0.3),multi_k=self.multi_k)
        local_param = self.model.state_dict()
        start_idx = 0
        for key in local_param:
            local_param[key] = global_param[start_idx:start_idx + len(local_param[key].data.view(-1))].reshape(
                local_param[key].data.shape)
            start_idx = start_idx + len(local_param[key].data.view(-1))
        self.model.load_state_dict(local_param)
    def multi_krum(self,all_updates, n_attackers, multi_k=False):
        candidates = []
        candidate_indices = []
        remaining_updates = all_updates
        all_indices = np.arange(len(all_updates))

        while len(remaining_updates) > 2 * n_attackers + 2:
            print(len(remaining_updates))
            torch.cuda.empty_cache()
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)

            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

            candidate_indices.append(all_indices[indices[0].cpu().numpy()])
            all_indices = np.delete(all_indices, indices[0].cpu().numpy())
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)

            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            if not multi_k:
                break

        aggregate = torch.mean(candidates, dim=0)

        return aggregate, np.array(candidate_indices)



    def train(self):
        loss = []

        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            self.send_parameters()
            # Evaluate model each interation
            self.evaluate()
            self.select_users()
            selected_benign_users = []
            for user in self.selected_users:
                if user in self.benign_users:
                    selected_benign_users.append(user)
                    user.train(self.local_epochs)  # * user.train_samples
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    user.set_param_new(selected_benign_users)
            '''
            for user in self.selected_users:
                if user in self.attackers:
                    malicious_state = user.get_malicious_model_state(selected_benign_users)
                    break
            for user in self.selected_users:
                if user in self.attackers:
                    user.load_malicious_state(malicious_state)
            self.aggregate_grads_Krum()
        print(max(self.rs_glob_acc))
        self.save_results()
        self.save_model()

    