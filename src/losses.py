import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from xitorch.optimize import equilibrium
import numpy as np
from math import exp
import math

        
class GroupMSE(torch.nn.Module):

    def forward(self, x_group):
        """
        x_group: [B,11,H,W]
        """
        x_ref = x_group[:,0,:,:]
        device = x_ref.device
        time_slices_number = x_group.shape[1]
        loss = 0#torch.tensor(0).to(device)
        for t in range(1,time_slices_number):
            loss += torch.mean((x_group[:,t,:,:] - x_ref) ** 2)
        return loss


class GroupMutualInformation(torch.nn.Module):
    """
    Mutual Information
    https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch/blob/5ebef416a7bfefda80b4bd8f58e6850eccebca2e/ViT-V-Net/losses.py#L288
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(GroupMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean() #average across batch

    def forward(self, x_group):
        """
        x_group: [B,11,H,W]
        """
        time_slices_number = x_group.shape[1]
        loss = 0

        # all to ref
        x_ref = x_group[:,0,:,:]
        for t in range(1,time_slices_number):
            loss += -self.mi(x_group[:,t,:,:],x_ref)

        # all to all
        # for i in range(0,time_slices_number):
        #     for j in range(0,time_slices_number):
        #         if j!=i:
        #             loss += -self.mi(x_group[:,i,:,:], x_group[:,j,:,:])
        
        return loss

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    https://github.com/junyuchen245/ViT-V-Net_for_3D_Image_Registration_Pytorch/blob/5ebef416a7bfefda80b4bd8f58e6850eccebca2e/ViT-V-Net/losses.py#L288
    """
    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma**2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1] # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab/nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean() #average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)


class GradGroup:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def loss(self, warp):
        #warp[B,11,2,160,160]
        dy = torch.abs(warp[:, :, :, 1:  ,:] - warp[:, :, :, :-1  ,:])
        dx = torch.abs(warp[:, :, :, :   ,1:] - warp[:, :, :, :   ,:-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

    def group_loss(self, warp):
        return self.loss(warp)

class T1GroupModelBased():
    def __init__(self):
        print(1) 

    def loss(self, y_group_pred,t_trigger_time,norm_matrix,y_group_mask,M0,T1,ref=0):
        """
        y_group_pred   [B,11,160,160]
        trigger_time   [B,11]
        norm_matrix    [B,11,2]
        y_group_mask   [B,160,160]
        T1             [B,1,160,160]
        """

        mse_loss = nn.MSELoss()
        device = y_group_pred.device #[B,11,160,160]
        norm_matrix = torch.tensor(norm_matrix).to(device)
        t_trigger_time = torch.tensor(t_trigger_time).to(device)
        y_group_mask = torch.tensor(y_group_mask).to(device)

        # # back to the original intensities a
        m = norm_matrix[:,:,1]
        m =m[:,:,None,None]
        M = norm_matrix[:,:,0]
        M =M[:,:,None,None]
        y_group_pred = y_group_pred*(M-m)+m  #[B,11,160,160]

        # convert the to first points to negative values
        y_group_pred[:,:2,:,:] = -1 * y_group_pred[:,:2,:,:] 

        # create syntetic data
        # y_synt = M0[y_group_mask.unsqueeze(1)>0] * ( 1-2*torch.exp(-t_trigger_time.unsqueeze(-1).unsqueeze(-1)/T1[y_group_mask.unsqueeze(1)>0]))
        # loss = mse_loss(y_group_pred[y_group_mask.unsqueeze(1).repeat([1,11,1,1])>0],y_synt)

        T1[T1<0] = torch.inf
        T1[T1>10000] = torch.inf
        y_synt = M0 * ( 1-2*torch.exp(-t_trigger_time.unsqueeze(-1).unsqueeze(-1)/T1)) #[B,11,160,160]
        loss = mse_loss(y_group_pred.float() * y_group_mask.unsqueeze(1),y_synt.float() * y_group_mask.unsqueeze(1))
        # loss = mse_loss(y_group_pred.float(),y_synt.float())
        

        return loss, y_synt, y_group_pred

class T1Group:
    """
    N-D T1 loss.
    """

    def __init__(self):
        print(1) 


    def estimate_T1_draft(self,Mz1,Mz2,t1,t2):
        """
        MZ1: [B,10,160,160]
        MZ2: [B,10,160,160]
        t1:  [B,10,1,1]
        t2:  [B,10,1,1]
        """
        device = Mz1.device

        
        indices1 = torch.logical_or(torch.logical_and(0<Mz1,Mz1<Mz2),torch.logical_and(Mz1<0,Mz2<Mz1)) #[B,10,160,160]
        indices2 = torch.logical_or(torch.logical_and(0>Mz1,0<Mz2),torch.logical_and(Mz2<0,0<Mz1)) #[B,10,160,160]
        indices3 = torch.logical_or(torch.logical_and(0>Mz2,Mz1<Mz2),torch.logical_and(Mz2>0,Mz2<Mz1)) #[B,10,160,160]
        
        H =torch.tensor(0.5).to(device)
        T1_min, T1_max = torch.zeros_like(Mz1).to(device) , torch.zeros_like(Mz1).to(device) 
        T1_min, T1_max = torch.where(indices1, 0., T1_min), torch.where(indices1, -t1/torch.log(H), T1_max)
        T1_min, T1_max = torch.where(indices2, -t1/torch.log(H), T1_min), torch.where(indices2, -t2/torch.log(H), T1_max)
        T1_min, T1_max = torch.where(indices3, -t2/torch.log(H), T1_min), torch.where(indices3, 70000., T1_max)

        T1_len = 5
        for i in range(5):  
            T1_grid = tensor_linspace(T1_min, T1_max, T1_len).to(device) #[B,10,160,160,T1_len]
            M01 = Mz1.unsqueeze(-1) / (1-2*torch.exp(-t1.unsqueeze(-1)/T1_grid))
            M02 = Mz2.unsqueeze(-1) / (1-2*torch.exp(-t2.unsqueeze(-1)/T1_grid))
            
            subtruction = torch.abs(M02-M01).squeeze() #[B,10,160,160,T1_len]
            subtruction = torch.flatten(subtruction,0,3) #[-1, T1_len]
            idx = softargmin(subtruction).reshape_as(Mz2)

            T1=torch.gather(T1_grid,-1,idx.to(torch.int64).unsqueeze(-1)).squeeze()
            
            index_m = torch.where(idx==0, 0, idx-1) #[B,10,160,160]
            index_M = torch.where(idx==T1_len-1, T1_len-1, idx+1) #[B,10,160,160]

            T1_min = torch.gather(T1_grid,-1,index_m.unsqueeze(-1)).squeeze()
            T1_max = torch.gather(T1_grid,-1,index_M.unsqueeze(-1)).squeeze()

        # return T1 values from the best indecies
        T1 = torch.gather(T1_grid,-1,idx.unsqueeze(-1)).squeeze()  #[8,10,160,160]
        M0 = Mz1 / (1-2*torch.exp(-t1/T1))
        return T1, M0
        

    def estimate_T1(self,Mz1,Mz2,t1,t2):
        """
        MZ1: [B,10,160,160]
        MZ2: [B,10,160,160]
        t1:  [B,10,1,1]
        t2:  [B,10,1,1]
        """
        device = Mz1.device

        T1_grid = torch.arange(700,1300,1).to(torch.float16).to(device) # new
        T1_grid.requires_grad = True

        M01 = Mz1.unsqueeze(-1) / (1-2*torch.exp(-t1.unsqueeze(-1)/T1_grid))
        M02 = Mz2.unsqueeze(-1) / (1-2*torch.exp(-t2.unsqueeze(-1)/T1_grid))

        subtruction = torch.abs(M02-M01).squeeze() #[B,10,160,160,T1_len]
        subtruction = torch.flatten(subtruction,0,3) #[-1, T1_len]
        idx = softargmin(subtruction).reshape_as(Mz2)


        T1=T1_grid[idx.long()]
        M0 = Mz1 / (1-2*torch.exp(-t1/T1))

        # import matplotlib.pyplot as plt
        # plt.imshow(T1[0,2,:,:].cpu(),cmap='jet',vmin=1,vmax=4000);plt.colorbar(); plt.savefig('debug_2_pic' + str(i)+'.png');plt.close()

        return T1, M0
        
    def estimate_optim(self, Mz, t):
        """
        Mz   [B,11,160,160]
        t   [B,11]
        """
        Mz_shape = Mz.shape
        m = Model(Mz_shape)
        # Instantiate optimizer
        opt = torch.optim.Adam(m.parameters(), lr=100)
        losses,losses_matrix, loss = training_loop(m, opt, t, Mz)
        T1, M0 = m.weights[0,:] , m.weights[1,:]
        return T1, M0, losses_matrix, loss

    def loss(self, y_group_pred,t_trigger_time,norm_matrix,y_group_mask,ref=0):
        """
        y_group_pred   [B,11,160,160]
        trigger_time   [B,11]
        norm_matrix    [B,11,2]
        y_group_mask   [B,160,160]
        """

        mse_loss = nn.MSELoss()
        device = y_group_pred.device
        norm_matrix = torch.tensor(norm_matrix).to(device)
        t_trigger_time = torch.tensor(t_trigger_time).to(device)
        y_group_mask = torch.tensor(y_group_mask).to(device)

        # back to the original intensities a
        m = norm_matrix[:,:,1]
        m =m[:,:,None,None]
        M = norm_matrix[:,:,0]
        M =M[:,:,None,None]
        y_group_pred = y_group_pred*(M-m)+m  #[B,11,160,160]

        # convert the to first points to negative values
        y_group_pred[:,:2,:,:] = -1 * y_group_pred[:,:2,:,:] 

        # find T1 for each two  points
        Mz1 = y_group_pred[:,ref      ,:,:].unsqueeze(1)            #[B,1,160,160]
        Mz2 = y_group_pred[:,ref+1  : ,:,:]                         #[B,10,160,160]
        t1 = t_trigger_time[:,ref     ,None,None].unsqueeze(1)      #[B,1,1,1]
        t2 = t_trigger_time[:,ref+1 : ,None,None]                   #[B,10,1,1]

        Mz = y_group_pred[:,ref+1  : ,:,:].clone().detach()
        t = t_trigger_time[:,ref+1 : ,None,None] .clone().detach()

        # T1_tensor, M0_tensor = self.estimate_T1(Mz1,Mz2,t1,t2)   #[B,10,160,160]    
        T1_tensor, M0_tensor, losses_tensor, loss = self.estimate_optim(Mz, t)   #[B,10,160,160]    


        # Trying to register to (Mz1=0,Mz2=2) : 

        # Mz12_ind_ref = 1
        # T1_tensor_ref = T1_tensor[:,Mz12_ind_ref,:,:].unsqueeze(1).repeat(1,T1_tensor.shape[1],1,1)
        # M0_tensor_ref = M0_tensor[:,Mz12_ind_ref,:,:].unsqueeze(1).repeat(1,T1_tensor.shape[1],1,1)
        # y_group_mask = y_group_mask.repeat(1,T1_tensor.shape[1],1,1)  #[B,160,160] => [B,10,160,160]
        # loss_T1 = mse_loss(T1_tensor[y_group_mask>0.1], T1_tensor_ref[y_group_mask>0.1])
        # loss_M0 = mse_loss(M0_tensor[y_group_mask>0.1], M0_tensor_ref[y_group_mask>0.1])
        # loss_T1 = losses_tensor[y_group_mask!=0]
        # loss_M0 = losses_tensor[y_group_mask!=0]

        # loss_T1_map = []
        # loss_T1_array = []
        # loss_M0_array = []

        # for ii in range(10):
        #     loss_T1_array.append(mse_loss(T1_tensor[:,ii,:,:][y_group_mask[:,ii,:,:]>0.1], T1_tensor_ref[:,ii,:,:][y_group_mask[:,ii,:,:]>0.1]).item())
        #     loss_M0_array.append(mse_loss(M0_tensor[:,ii,:,:][y_group_mask[:,ii,:,:]>0.1], M0_tensor_ref[:,ii,:,:][y_group_mask[:,ii,:,:]>0.1]).item())

        # loss_T1_map = torch.where(y_group_mask>0.1, torch.abs(T1_tensor - T1_tensor_ref) ,y_group_mask) 


        # Register to the mean :

        # mean_T1 = torch.mean(T1_tensor, dim=1, keepdim= True)  #[B,1,160,160]
        # mean_M0 = torch.mean(M0_tensor, dim=1, keepdim= True)  #[B,1,160,160]
        # mean_T1 = mean_T1.repeat(1,T1_tensor.shape[1],1,1)
        # mean_M0 = mean_M0.repeat(1,T1_tensor.shape[1],1,1)
        # y_group_mask = y_group_mask.unsqueeze(1).repeat(1,T1_tensor.shape[1],1,1).to(torch.float64)  #[B,160,160] => [B,10,160,160]
        # loss_T1 = mse_loss(T1_tensor[y_group_mask>0.1], mean_T1[y_group_mask>0.1])
        # loss_M0 = mse_loss(M0_tensor[y_group_mask>0.1], M0_tensor[y_group_mask>0.1])
        # loss_T1_map = torch.where(y_group_mask>0.1, torch.abs(T1_tensor - mean_T1) ,y_group_mask) 

        # return loss_T1, loss_M0, loss_T1_map, T1_tensor, loss_T1_array, loss_M0_array
        return loss

class IncreaseGroup:

    def __init__(self):
            print(1)

    def loss(self, y_group_pred):
        #y_group_pred.shape = [B,11,H,W]
        relu = nn.ReLU()
        dz = y_group_pred[:, 1:, :, :] - y_group_pred[:, :-1, :, :]
        
        return torch.mean(relu(-dz))






     


@torch.jit.script
def linspace(start, stop, num:int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    out = start[None] + steps*(stop - start)[None]
    return out

def tensor_linspace(start, end, steps=10):
    """
    https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def softargmin(input):
    device = input.device
    x,y = input.shape
    input = nn.functional.softmax(torch.max(input) - input * 100, dim=1).to(device)
    input[input<1] = 0
    indices = torch.linspace(0, 1, y).to(device).to(torch.int64)  #len 100
    result = torch.round(torch.sum((y - 1) * input * indices, dim=1))
    return result


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self,Mz_shape):
        
        super().__init__()
        # initialize weights with random numbers
        T1 = torch.ones(Mz_shape[0],1,Mz_shape[2],Mz_shape[3]) * 800.0
        M0 = torch.ones(Mz_shape[0],1,Mz_shape[2],Mz_shape[3]) * 100.0
        weights = torch.cat([T1.unsqueeze(0),M0.unsqueeze(0)],dim=0).to('cuda')
        # make weights torch parameters
        self.weights = nn.Parameter(weights).to('cuda')        
        
    def forward(self, t):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        T1,M0 = self.weights[0,:], self.weights[1,:]
        Mz = M0*(1-2*torch.exp(-t/T1))
        return Mz
        
def training_loop(model, optimizer,t,Mz ,n=100):
    "Training loop for torch model."
    
    losses = []
    for i in range(n):
        preds = model(t).to('cuda')
        loss = F.mse_loss(preds, Mz).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item()) 
    losses_matrix = torch.sum(((preds-Mz)**2).sqrt(),dim = 1)
    return losses, losses_matrix, loss

class DiceSC():
    def __init__(self):
        v=1

    def loss(self,y_pred,y_true,printflag=False):
        if len(y_pred.shape) == 3:
            y_pred = y_pred[np.newaxis,:]
            y_true = y_true[np.newaxis,:]

        vol_axes = [2,3]
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        dice_all = top / bottom
        if printflag:
            print('dice mean = {}'.format(torch.mean(top / bottom).item()))
            print('dice median = {}'.format(torch.median(top / bottom).item()))
            print('dice std = {}'.format(torch.std(top / bottom).item()))
        dice_info = [torch.median(top / bottom).item(), torch.mean(top / bottom).item(), torch.std(top / bottom).item()]
        return -dice , dice_info, dice_all.cpu().detach().numpy()

