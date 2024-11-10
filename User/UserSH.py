import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from User.User import Userbase


# Implementation for FedAvg clients


class UserNDSS(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs, localstep, n_attackers):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs, localstep)
        self.n_attackers = n_attackers
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
            global_param = self.get_params()
            param = param - global_param
            user_param = param[None, :] if len(user_param) == 0 else torch.cat((user_param, param[None, :]),
                                                                               0)
        print("generate malicious vector")
        malicious_param = self.generated_malicious_param(user_param)
        print("turn malicious vector to malicious state")
        local_param = self.model.state_dict()
        start_idx = 0
        for key in local_param:
            local_param[key] = malicious_param[start_idx:start_idx + len(local_param[key].data.view(-1))].reshape(
                local_param[key].data.shape)
            start_idx = start_idx + len(local_param[key].data.view(-1))
        return local_param

    def generated_malicious_param(self, all_updates):

        n_attackers = 1
        model_re = torch.mean(all_updates, 0)
        deviation = torch.sign(model_re)
        b = 2
        max_vector = torch.max(all_updates, 0)[0]
        min_vector = torch.min(all_updates, 0)[0]
        max_vector = max_vector.cuda()
        min_vector = min_vector.cuda()

        max_ = (max_vector > 0).type(torch.FloatTensor).cuda()
        min_ = (min_vector < 0).type(torch.FloatTensor).cuda()

        max_[max_ == 1] = b
        max_[max_ == 0] = 1 / b
        min_[min_ == 1] = b
        min_[min_ == 0] = 1 / b

        max_range = torch.cat((max_vector[:, None], (max_vector * max_)[:, None]), dim=1)
        min_range = torch.cat(((min_vector * min_)[:, None], min_vector[:, None]), dim=1)

        rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()

        max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
            [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
            [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

        mal_vec = (torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * max_rand.shape[1]).T.cuda() * max_rand + torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]).T.cuda() * min_rand).T
        return torch.squeeze(mal_vec, 0)

    def load_malicious_state(self, malicious_state):

        self.model.load_state_dict(malicious_state)
