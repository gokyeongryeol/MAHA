import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader import *
from MAHA_classification import *
from utils import *
import pickle
import numpy as np

import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import time
import random

matplotlib.use('Agg')

def main(data, shot, phase, index, n_epoch, dp_rate, KL_w, lr, is_train):
    train_loader = Heterogeneous_dataset(batch_size=4, update_batch_size=shot, update_batch_size_eval=15, num_classes=5, index=index, data_source=data[:-3], is_train=is_train)

    model = MAHA(dim_input=84*84*3, ln=True, phase=phase, batch_size=4, num_inds=256, dp_rate=dp_rate).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    KL_list, NLL_list = [], []
    train_loss_list, test_loss_list = [], []
    train_accur_list, test_accur_list = [], []
     
    for epoch in range(n_epoch):
        model.train()
        
        image_batch, label_batch, sel_set = train_loader.generate_multidataset_batch(is_test=False)
        
        Cx = torch.tensor(image_batch[:, :5*shot, :], dtype=torch.float)
        Cy = torch.tensor(label_batch[:, :5*shot, :], dtype=torch.float)
        Tx = torch.tensor(image_batch, dtype=torch.float)
        Ty = torch.tensor(label_batch, dtype=torch.float)
            
        train_KL, train_NLL, train_accur = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device))
        train_loss = KL_w * train_KL + train_NLL
        
        KL_list.append(train_KL.item())
        NLL_list.append(train_NLL.item())
        
        train_loss_list.append(train_loss.item())
        train_accur_list.append(train_accur.item())
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
                    
        with torch.no_grad():
            model.eval()
        
            image_batch, label_batch, sel_set = train_loader.generate_multidataset_batch(is_test=True)
            
            Cx = torch.tensor(image_batch[:, :5*shot, :], dtype=torch.float)
            Cy = torch.tensor(label_batch[:, :5*shot, :], dtype=torch.float)
            Tx = torch.tensor(image_batch, dtype=torch.float)
            Ty = torch.tensor(label_batch, dtype=torch.float)
            
            test_NLL, test_accur = model(Cx.to(device), Cy.to(device), Tx.to(device), Ty.to(device))
            test_loss = test_NLL
            test_loss_list.append(test_loss.item())
            test_accur_list.append(test_accur.item())
            
        if (epoch+1) % 10 == 0:
            print('[Epoch %d] train_KL: %.3f, train_NLL: %.3f, train_loss: %.3f, test_loss: %.3f' 
                  % (epoch+1, train_KL, train_NLL, train_loss, test_loss))
            
            torch.save(model.state_dict(), '../models/'+data+str(shot)+phase+'/'+str(epoch+1)+'.pt')
            
            torch.save(KL_list, '../loss/'+data+str(shot)+phase+'/KL_list.pt')
            torch.save(NLL_list, '../loss/'+data+str(shot)+phase+'/NLL_list.pt')
            torch.save(train_loss_list, '../loss/'+data+str(shot)+phase+'/train_loss_list.pt')
            torch.save(test_loss_list, '../loss/'+data+str(shot)+phase+'/test_loss_list.pt')
            torch.save(train_accur_list, '../loss/'+data+str(shot)+phase+'/train_accur_list.pt')
            torch.save(test_accur_list, '../loss/'+data+str(shot)+phase+'/test_accur_list.pt')
        
        
def cluster_check(data, shot, phase, dp_rate):
    print('cluster check!')
    model = MAHA(dim_input=84*84*3, ln=True, phase=phase, batch_size=4, num_inds=256, dp_rate=dp_rate).to(device)
    model.load_state_dict(torch.load('../models/'+data+str(shot)+phase+'/18000.pt'))
    model.eval()
    
    test_loader = Heterogeneous_dataset(batch_size=4, update_batch_size=shot, update_batch_size_eval=15, num_classes=5, data_source=data, is_train=False)
    
    result_path = '../plots/'+data+str(shot)+phase
    plot_image_scatter(model, test_loader, 5, shot, data, result_path)
    
    
def regression(data, shot, phase, dp_rate):
    print('regression!')
    model = MAHA(dim_input=84*84*3, ln=True, phase=phase, batch_size=4, num_inds=256, dp_rate=dp_rate).to(device)
    model.load_state_dict(torch.load('../models/'+data+str(shot)+phase+'/18000.pt'))
    model.eval()
    
    test_loader = Heterogeneous_dataset(batch_size=4, update_batch_size=shot, update_batch_size_eval=15, num_classes=5, data_source=data, is_train=False)
    
    with torch.no_grad():
        accur_list = []
        for cnt in range(250):
            image_batch, label_batch, sel_set = test_loader.generate_multidataset_batch(is_test=True)
            
            Cx = torch.tensor(image_batch[:, :5*shot, :], dtype=torch.float)
            Cy = torch.tensor(label_batch[:, :5*shot, :], dtype=torch.float)
            
            _, accur = model(Cx.to(device), Cy.to(device), Cx.to(device), Cy.to(device))
           
            accur_list.append(np.array(accur.cpu()))
        mean = np.mean(np.array(accur_list))
        std = np.std(np.array(accur_list))
        print(mean, std)
        with open(data+str(shot)+phase+"accur_list.txt", 'w') as f:
            f.write(str(mean) + '/' + str(std) + '\n')

if __name__ == '__main__':  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    data = 'multidataset'
    shot = 1
    phase = 'pretrain'
    n_epoch = 100000
    index = None

    #data = 'multidataset'
    #shot = 5
    #phase = 'pretrain'
    #n_epoch = 100000
    #index = None
    
    #data = 'multidataset'
    #shot = 1
    #phase = 'finetuning'
    #n_epoch = 100000
    #index = 0

    #data = 'multidataset'
    #shot = 5
    #phase = 'finetuning'
    #n_epoch = 100000
    #index = 0

    tic = time.time()
    #hyperparameter search
    for itr in range(5):
        dp_rate = np.random.uniform(0.0, 0.5)
        KL_w = np.random.uniform(1e-4, 1e-1)
        lr = np.random.uniform(1e-6, 1e-4)
        is_train = False
        print(dp_rate, KL_w, lr, is_train)
        with open(data+str(itr)+'th'+str(shot)+phase, 'w') as f:
            f.write(str(dp_rate) + '/' + str(KL_w) + '/' + str(lr) + '\n')
        data_prime = data+str(itr)+'th'
        main(data_prime, shot, phase, index, 5000, dp_rate, KL_w, lr, is_train)
    
    #main(task, kind, phase, n_epoch, 256, 0.017521714799762885, 1.0, 4.1151246873666413e-05, 1.0, False)
    
    #with torch.no_grad():
    #    cluster_check(data, shot, phase, dp_rate)
    #with torch.no_grad():
    #    regression(data, shot, phase, dp_rate)
    toc = time.time()
    
    mon, sec = divmod(toc-tic, 60)
    hr, mon = divmod(mon, 60)
    print('total wall-clock time is ', int(hr),':',int(mon),':',int(sec))
    