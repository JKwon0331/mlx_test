
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from utils_main import batch_iterate, CosineAnnealingWarmUpRestarts
import os
from model import Model
import mlx.optimizers as optim
import scipy.io as sio
from mlx.utils import tree_flatten

def train(model, optimizer, scheduler, epoch, tr_dataset, val_dataset, cv):
    #gc.collect()
    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    start = 0
    min_val_loss = float('inf')
    loss = {}
    loss['train'] = mx.zeros(epoch)
    loss['val'] = mx.zeros(epoch)

    print("Start to train classifier")
    
    def loss_fn(model, x, y, ):
        return mx.mean(nn.losses.mse_loss(model(x),y))

    model.train()
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    for epo in range(epoch):
    
        if scheduler is not None:
            scheduler.step(epo)
            
        start_time = time.time()
        train_loss = 0
        
        tcnt = 0
        for src,answ in batch_iterate(4, tr_dataset, train = True, shuffle = True):
            if scheduler is not None:
                optimizer.learning_rate = scheduler.get_last_lr()

            tloss,grads = loss_and_grad_fn(model,src,answ)
            
            train_loss += tloss
            optimizer.update(model,grads)
            
            if scheduler is not None:
                optimizer.learning_rate = scheduler.get_last_lr()
            
            mx.eval(model.parameters(), optimizer.state)   
            tcnt = tcnt + 1
    
        elapsed = time.time() - start_time
        
        train_loss = train_loss/tcnt
        loss['train'][epo] = train_loss
        
        val_loss = 0
        model.eval()
        cnt = 0
        for src,answ in batch_iterate(1, val_dataset, train = False, shuffle = False):
            val_loss += loss_fn(model,src,answ)
            cnt = cnt + 1
            
        val_loss = val_loss/cnt
        loss['val'][epo] = val_loss
            
        if min_val_loss > val_loss:
            min_val_loss = val_loss
        
        print('Sub {:s} | CV {:3d} | epoch {:3d} | {:5d} batches | '
              '{:5.2f} s | scheduler_lr {:5.7f} | optimizer_lr {:5.7f} | '
              ' train_loss {:5.5f} | val_loss {:5.5f} | num_params {:5d}'.format(
             'S001', cv, epo + 1, tcnt, elapsed, scheduler.get_last_lr(),
            optimizer.learning_rate.item(),  np.array(train_loss), np.array(val_loss), num_params ))
        
        
        model.train()
        if (mx.array(1000) == (epo + 1)).sum():
            print('Save_model')
            sm_time = time.time()
            save_dir_now = 'epo_{:05d}'.format(epo+1) + '/CV' + str(cv) + '/'
            if not os.path.isdir(save_dir_now):
                os.makedirs(save_dir_now)
                
            G = {}
            G['val_loss'] = loss['train']
            G['tr_loss'] = loss['val']
            G['epo'] = epo+1
            G['min_val_loss'] = min_val_loss
            sio.savemat(save_dir_now + 'info.mat',G)

            model.save_weights(save_dir_now + 'weight.npz')
            print('Model_saved: {:2.3f}'.format((time.time()-sm_time)))       
       
       
def main(cv):
    init_fn1 = nn.init.uniform(low=-10, high = 10)
    init_fn2 = nn.init.uniform(low = -1, high = 1)
    tr_data = init_fn1(mx.zeros((200, 130, 5000)))
    tr_ans = init_fn2(mx.zeros((200, 130, 100)))

    val_data = init_fn1(mx.zeros((20, 130, 5000)))
    val_ans = init_fn2(mx.zeros((20, 130, 100)))
    
    tr_dataset = [tr_data, tr_ans]
    val_dataset = [val_data, val_ans]

    model = Model(d_input = 5000, d_output = 100, d_layers = 1, d_model = 25*64, nhead = 25, e_layers = 4)
    optimizer = optim.AdamW(learning_rate = 0.0001)
    scheduler = CosineAnnealingWarmUpRestarts(500, 1, 0.001, 0.0001, 100, 0.9)
    mx.eval(model.parameters())

    train(model, optimizer, scheduler, 1000, tr_dataset, val_dataset,cv)

if __name__ == "__main__":
    main(1)