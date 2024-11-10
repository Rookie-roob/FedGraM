import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from User.User import Userbase

# Implementation for FedAvg clients


class UserLIE(Userbase):
    def __init__(self, device, numeric_id, train_data, model, batch_size, learning_rate,
                 local_epochs,localstep, n_attackers):
        super().__init__(device, numeric_id, train_data, model, batch_size, learning_rate,
                         local_epochs,localstep)
        self.Zvalue={5:0.7031, 10:0.72575, 100:0.73319,50:0.70098,25:0.69993}
        self.n_attackers=n_attackers
        self.loss = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
    def setparam(self, selected_users):
        z=self.Zvalue[self.n_attackers]
        allparam=[]
        for user in selected_users:
            param = user.get_param()
            allparam = param[None, :] if len(allparam) == 0 else torch.cat((allparam, param[None, :]),
                                                                                    0)
        avg=torch.mean(allparam,dim=0)
        std=torch.std(allparam,dim=0)
        resulted_param=avg+z*std
        start_idx = 0

        local_para = self.model.state_dict()

        start_idx=0
        for key in local_para:
            '''
            if local_para[key].data.shape==torch.Size([]):
                continue
            '''
            print(local_para[key].data.shape)
            print(len(local_para[key].data.view(-1)))
            local_para[key]=resulted_param[start_idx:start_idx + len(local_para[key].data.view(-1))].reshape(local_para[key].data.shape)
            start_idx = start_idx + len(local_para[key].data.view(-1))
        '''
        for param in self.model.parameters():
            if start_idx==0 and self.id==49:
                print(param)
                print("before")
            param = resulted_param[start_idx:start_idx + len(param.data.view(-1))].reshape(param.data.shape)
            start_idx = start_idx + len(param.data.view(-1))
            param = param.to(self.device)
            if start_idx-len(param.data.view(-1))==0 and self.id==49:
                print(param)
                print("middle")
        for param in self.model.parameters():
            if  self.id==49:
                print(param)
                print("after")
                break
        '''
        self.model.load_state_dict(local_para)

    def set_param_new(self,selected_users):
        z = self.Zvalue[self.n_attackers]
        local_param=self.model.cpu().state_dict()
        users_param=[]
        for user in selected_users:
            users_param.append(user.model.cpu().state_dict())
        for key in local_param:
            param_temp = []
            if users_param[0][key].dim()!=0:
                #print(users_param[0][key].cpu().dim())
                #print(key)
                for user_param in users_param:
                    param_temp = user_param[key][None, :] if len(param_temp) == 0 else torch.cat((param_temp, user_param[key][None, :]), 0)
                avg = torch.mean(param_temp, dim=0)
                std = torch.std(param_temp, dim=0)
                local_param[key] = avg + z * std
            else:
                for user_param in users_param:
                    param_temp.append(user_param[key].item())
                avg=np.mean(param_temp)
                std=np.std(param_temp)
                local_param[key]=torch.tensor(avg+z*std)
        self.model.load_state_dict(local_param)
        print(str(self.id)+"  "+"over!")

    def get_malicious_model_state(self,selected_users):
        user_param = []
        #print("collect benign users' state and turn to 1 dimension vector")
        for user in selected_users:
            param = user.get_params()
            global_param = self.get_params()
            param = global_param - param
            user_param = param[None, :] if len(user_param) == 0 else torch.cat((user_param, param[None, :]),
                                                                               0)
        #print("generate malicious vector")
        malicious_param = self.generated_malicious_param(user_param)
        #print("turn malicious vector to malicious state")
        local_param = self.model.state_dict()
        start_idx = 0
        for key in local_param:
            local_param[key] = local_param[key] - malicious_param[
                                                  start_idx:start_idx + len(local_param[key].data.view(-1))].reshape(
                local_param[key].data.shape)
            start_idx = start_idx + len(local_param[key].data.view(-1))
        return local_param


    def load_malicious_state(self,malicious_state):

        self.model.load_state_dict(malicious_state)







    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def generated_gradients(self, all_updates):
        z=self.Zvalue[10]
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std
    def generated_malicious_param(self, all_updates):
        z=self.Zvalue[self.n_attackers]
        avg = torch.mean(all_updates, dim=0)
        std = torch.std(all_updates, dim=0)
        return avg + z * std