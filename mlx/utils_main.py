
import numpy as mx
import math
import numpy as np
import mlx.core as mx
   
def batch_iterate(batch_size, dataset, train = False, shuffle = True):
    
    x_data, y_data= dataset
    datalen = 130
    dur = 60
    nData = x_data.shape[0]
    perm = np.arange(nData)
    if shuffle:
        np.random.shuffle(perm)
    perm = mx.array(perm)
    
    
    for s in range(0, nData, batch_size):
        ids = perm[s:s+batch_size]
        x = []
        y = []    
        
        for idx in range(len(ids)):
            ind = ids[idx]
            xx = x_data[ind,]
            yy = y_data[ind,]
            if train:
                max_start = max(0, datalen - dur)
                mel_start = mx.random.randint(0,max_start).item()
                
                xx = xx[mel_start : mel_start + dur]
                yy = yy[mel_start : mel_start + dur]
            
            x += [xx]
            y += [yy]
            
        
        yield mx.array(x), mx.array(y)
        

class CosineAnnealingWarmUpRestarts():
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.first = True
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmUpRestarts, self).__init__()
    
    def get_last_lr(self):
        if self.step_in_cycle == -1:
            return self.min_lr
        elif self.step_in_cycle < self.warmup_steps:
            return (self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr
        else:
            return self.min_lr + (self.max_lr - self.min_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = epoch
        

