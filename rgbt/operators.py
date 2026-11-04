"""Versioned reliability operators. No modifications to historical vendor modules."""
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'tricue-portable-v1'

class SCR(nn.Module):
    """Return native enhancement and the exact window moments used by SDGW."""
    def __init__(self, native):
        super().__init__(); self.native = native
    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z=x.float(); w=self.native.std; pad=(w.padding[1],)*2+(w.padding[0],)*2
            mean=F.conv2d(F.pad(z,pad,mode='reflect'),w.mean_kernel,groups=z.shape[1])
            second=F.conv2d(F.pad(z.square(),pad,mode='reflect'),w.mean_kernel,groups=z.shape[1])
            std=(second-mean.square()).clamp_min(w.eps).sqrt()
            enhanced=self.native.convl2l(z)*self.native.sig(self.native.conv1(std))
        return enhanced.to(x.dtype), (mean,std)

class Reliability(nn.Module):
    def __init__(self):
        super().__init__(); self.head=nn.Sequential(nn.Conv2d(5,16,1),nn.SiLU(),nn.Conv2d(16,2,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
    def forward(self,r,t,rs,ts):
        with torch.autocast(device_type=r.device.type,enabled=False):
            r,t=r.float(),t.float()
            # Channel-wise spatial normalization removes modality amplitude scale.
            def normalized(x,moments):
                offset=x.mean((-2,-1),keepdim=True)
                scale=x.var((-2,-1),unbiased=False,keepdim=True).add(1e-6).sqrt()
                mu,std=moments
                return (x-offset)/scale,((mu-offset)/scale).mean(1,keepdim=True),(std/scale).mean(1,keepdim=True)
            rn,rm,rv=normalized(r,rs);tn,tm,tv=normalized(t,ts)
            difference=(rn-tn).abs().mean(1,keepdim=True)
            weights=self.head(torch.cat([rm,rv,tm,tv,difference],1)).softmax(1)
            entropy=-(weights*weights.clamp_min(1e-8).log()).sum(1,keepdim=True)
            condition=torch.cat([weights,difference,entropy],1)
        self.diagnostics=dict(weight_sum_error=float((weights.sum(1)-1).abs().max().detach()),entropy=float(entropy.mean().detach()),weight_spatial_std=float(weights.std((-2,-1),unbiased=False).mean().detach()),difference=float(difference.mean().detach()))
        return weights,condition

class RCMF(nn.Module):
    def __init__(self,efc):
        super().__init__();self.efc=efc;self.reliability=Reliability()
    def forward(self,r,t,rs,ts):
        weights,condition=self.reliability(r,t,rs,ts)
        return self.efc([2*weights[:,0:1]*r,2*weights[:,1:2]*t]),condition

class FSPR(nn.Module):
    """Reuse the original high-pass response in both native paths, exactly once."""
    def __init__(self,hfp):
        super().__init__();self.hfp=hfp
        self.gate=nn.Sequential(nn.Conv2d(8,16,1),nn.SiLU(),nn.Conv2d(16,2,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
    def forward(self,x,condition=None):
        dtype=x.dtype
        with torch.autocast(device_type=x.device.type,enabled=False):
            z=x.float();h=self.hfp.high_pass(z);low=z-h
            maximum=F.adaptive_max_pool2d(h,self.hfp.patch).relu().sum((2,3),keepdim=True)
            average=F.adaptive_avg_pool2d(h,self.hfp.patch).relu().sum((2,3),keepdim=True)
            attention=torch.sigmoid(self.hfp.channel2(self.hfp.channel1(maximum)+self.hfp.channel1(average)))
            native=self.hfp.out(z*h+z*attention)
            content=torch.cat([z.mean(1,keepdim=True),z.var(1,unbiased=False,keepdim=True).add(1e-6).sqrt(),h.abs().mean(1,keepdim=True),low.abs().mean(1,keepdim=True)],1)
            if condition is None:condition=z.new_zeros(z.shape[0],4,*z.shape[-2:])
            elif condition.shape[-2:]!=z.shape[-2:]:condition=F.interpolate(condition.float(),size=z.shape[-2:],mode='bilinear',align_corners=False)
            alpha,beta=(.5*self.gate(torch.cat([content,condition],1)).tanh()).chunk(2,1)
            out=native+alpha*h+beta*low
        self.diagnostics=dict(reconstruction_error=float((z-h-low).abs().max().detach()),high_energy=float(h.square().mean().detach()),low_energy=float(low.square().mean().detach()),alpha=float(alpha.mean().detach()),beta=float(beta.mean().detach()))
        return out.to(dtype)
