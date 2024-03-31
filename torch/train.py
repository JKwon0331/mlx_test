import torch.nn as nn
import torch
import time
import numpy as np
from torch.utils.data import DataLoader

from utils_main import Dataset_mel, CosineAnnealingWarmUpRestarts
from model import Model

def train(model, optimizer, scheduler, epoch, tr_loader, val_loader, cv):
    start = 0
    
    device = 'mps'
    min_val_loss = float('inf')
    loss = {}
    loss['train'] = np.zeros(epoch)
    loss['val'] = np.zeros(epoch)
    
    model.to(device)

    print("Start to train classifier")

    model.train()
    for epo in range(epoch):
        if scheduler is not None:
            scheduler.step(epo)
        
        start_time = time.time()
        train_loss = 0

        ccnt = 0
        for src, answ in tr_loader:
            ccnt = ccnt + 1
            
            optimizer.zero_grad()
            prediction = model(src.to(device))#, src_mask.to(args.device))
            
            loss_g = nn.MSELoss()(prediction, answ.to(device))
            train_loss += (loss_g.detach().cpu().item())
            
            loss_g.backward()
            optimizer.step()
        
        elapsed = time.time() - start_time
        loss['train'][epo] = train_loss/len(tr_loader)

        with torch.no_grad():
            val_loss = 0
            for src, answ in val_loader:
                prediction = model(src.to(device))#, src_mask.to(args.device))
                val_loss += nn.MSELoss()(prediction, answ.to(device)).cpu().detach().item()
    
            if min_val_loss > val_loss/len(val_loader):
                min_val_loss = val_loss/len(val_loader)
        
            loss['val'][epo] = val_loss/len(val_loader)
        
        print('Sub {:s} | CV {:3d} | epoch {:3d} | {:5d} batches | '
              '{:5.2f} s | scheduler_lr {:5.7f} | optimizer_lr {:5.7f} | '
              ' train_g_loss {:5.5f} | val_g_loss {:5.5f} | '.format(
             'S001', cv, epo + 1, len(tr_loader), elapsed, scheduler.get_last_lr()[0],
            optimizer.param_groups[0]['lr'],
                        train_loss / len(tr_loader), val_loss/len(val_loader)))
        
        
        if 1000 == (epo + 1):
            print('Save_model')
            sm_time = time.time()
            G_dict = {'model': model.state_dict(),
                      'optim': optimizer.state_dict(),
                      'min_val_loss': min_val_loss,
                      'epoch': epo + 1
                      }
            if scheduler is not None:
                G_dict['sche'] = scheduler.state_dict()

            torch.save(G_dict, 'G_{:04d}'.format(epo + 1))
            print('Model_saved: {:2.3f}'.format((time.time()-sm_time)))
        
def main(cv):
    
    
    tr_data = np.random.rand(200, 130, 5000)* 10 
    tr_ans = np.random.rand(200, 130, 100)

    val_data = np.random.rand(20, 130, 5000)
    val_ans = np.random.rand(20, 130, 100)
    
    
    tr_dataset = Dataset_mel(x=tr_data, y=tr_ans, train = True)
    tr_loader = DataLoader(tr_dataset, batch_size= 4, shuffle=True)
    
    val_dataset = Dataset_mel(x=val_data, y=val_ans, train = False)
    val_loader = DataLoader(val_dataset, batch_size= 1, shuffle=False)

    model = Model(d_input = 5000, d_output = 100, d_layers = 1, d_model = 25*64, nhead = 25, e_layers = 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, 500, 1, 0.001, 0.0001, 100, 0.9)
   

    train(model, optimizer, scheduler, 1000, tr_loader, val_loader,cv)

if __name__ == "__main__":
    main(1)