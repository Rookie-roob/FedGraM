import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserNDSS(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs,localstep, n_attackers):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs,localstep)
        self.n_attackers=n_attackers
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]


    def get_malicious_model_state(self, selected_users):

        user_param = []
        print("collect benign users' state and turn to 1 dimension vector")
        for user in selected_users:
            param = user.get_params()
            global_param=self.get_params()
            param=global_param-param
            user_param = param[None, :] if len(user_param) == 0 else torch.cat((user_param, param[None, :]),
                                                                               0)
        print("generate malicious vector")
        malicious_param = self.generated_malicious_param_MinSum(user_param).cpu()
        print("turn malicious vector to malicious state")
        local_param = self.model.state_dict()
        start_idx = 0
        for key in local_param:
            local_param[key] = local_param[key]-malicious_param[start_idx:start_idx + len(local_param[key].data.view(-1))].reshape(local_param[key].data.shape)
            start_idx = start_idx + len(local_param[key].data.view(-1))
        return local_param


    def load_malicious_state(self,malicious_state):

        self.model.load_state_dict(malicious_state)

    def generated_malicious_param_MinMax(self,all_updates,dev_type='std'):
        model_re = torch.mean(all_updates, 0)
        all_updates=all_updates.cuda()
        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)
        #deviation=deviation.cuda()
        #aaa=1

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        max_distance = torch.max(distances)
        del distances
        model_re=model_re.cuda()
        deviation=deviation.cuda()
        #lamda_succ=lamda_succ.cuda()
        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            max_d = torch.max(distance)

            if max_d <= max_distance:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        mal_update = (model_re - lamda_succ * deviation)

        return mal_update

    def generated_malicious_param_MinSum(self,all_updates, dev_type='sign'):
        model_re=torch.mean(all_updates, 0)
        all_updates = all_updates.cuda()
        if dev_type == 'unit_vec':
            deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
        elif dev_type == 'sign':
            deviation = torch.sign(model_re)
        elif dev_type == 'std':
            deviation = torch.std(all_updates, 0)

        lamda = torch.Tensor([10.0]).float().cuda()
        # print(lamda)
        threshold_diff = 1e-5
        lamda_fail = lamda
        lamda_succ = 0

        distances = []
        for update in all_updates:
            distance = torch.norm((all_updates - update), dim=1) ** 2
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        scores = torch.sum(distances, dim=1)
        min_score = torch.min(scores)
        del distances
        model_re = model_re.cuda()
        deviation = deviation.cuda()

        while torch.abs(lamda_succ - lamda) > threshold_diff:
            mal_update = (model_re - lamda * deviation)
            distance = torch.norm((all_updates - mal_update), dim=1) ** 2
            score = torch.sum(distance)

            if score <= min_score:
                # print('successful lamda is ', lamda)
                lamda_succ = lamda
                lamda = lamda + lamda_fail / 2
            else:
                lamda = lamda - lamda_fail / 2

            lamda_fail = lamda_fail / 2

        # print(lamda_succ)
        mal_update = (model_re - lamda_succ * deviation)

        return mal_update
