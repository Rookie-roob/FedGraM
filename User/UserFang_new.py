import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserFang(Userbase):
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
        #print("collect benign users' state and turn to 1 dimension vector")
        for user in selected_users:
            param = user.get_params()
            global_param=self.get_params()
            param=global_param-param
            user_param = param[None, :] if len(user_param) == 0 else torch.cat((user_param, param[None, :]),
                                                                               0)
        #print("generate malicious vector")
        malicious_param = self.generated_malicious_param(user_param).cpu()
        #print("turn malicious vector to malicious state")
        local_param = self.model.state_dict()
        start_idx = 0
        for key in local_param:
            local_param[key] = local_param[key]-malicious_param[start_idx:start_idx + len(local_param[key].data.view(-1))].reshape(local_param[key].data.shape)
            start_idx = start_idx + len(local_param[key].data.view(-1))
        return local_param
    def generated_malicious_param(self, all_updates):

        n_attackers = 1
        model_re = torch.mean(all_updates, 0)
        model_std=torch.std(all_updates,0)
        deviation = torch.sign(model_re).cuda()


        max_low = model_re + 3*model_std
        max_high = model_re + 4 * model_std
        min_low = model_re-4*model_std
        min_high= model_re-3*model_std
        max_low=max_low.cuda()
        max_high = max_high.cuda()
        min_low = min_low.cuda()
        min_high = min_high.cuda()
        max_range = torch.cat((max_low[:,None],max_high[:,None]), dim=1)
        min_range = torch.cat((min_low[:, None], min_high[:, None]), dim=1)

        rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), n_attackers])).type(torch.FloatTensor).cuda()


        max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
            [max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
        min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack(
            [min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
        mal_update = (deviation < 0) * max_rand.T.cuda()  + (deviation > 0) * min_rand.T.cuda()

        return torch.squeeze(mal_update,0)

    def load_malicious_state(self,malicious_state):

        self.model.load_state_dict(malicious_state)
