import torch
from torch import nn
from torch.distributions import Categorical
from manager_policy import ManagerPolicy
from worker_policy import WorkerPolicy

DIMS={'pool':6,'order':17,'template':10,'subbatch':5,'machine':10,'slot':5}

class AttentionEncoder(nn.Module):
    """Typed entity attention with structural host-machine grounding and optional causal history.

    Store raw history in each source graph, so training recomputes precisely the
    collected history rather than using the encoder's current mutable history.
    """
    def __init__(self, width=32, temporal=False, history=8):
        super().__init__()
        self.proj=nn.ModuleDict({
            k:nn.Sequential(nn.Linear(v,width),nn.LayerNorm(width),nn.ReLU())
            for k,v in DIMS.items()
        })
        self.type_emb=nn.ParameterDict({k:nn.Parameter(torch.zeros(1,width)) for k in DIMS})
        self.context=nn.Sequential(nn.Linear(5,width),nn.LayerNorm(width),nn.ReLU())
        self.slot_host_proj=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.ReLU())
        self.subbatch_order_proj=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.ReLU())
        self.template_order_proj=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.ReLU())
        self.subbatch_machine_proj=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width),nn.ReLU())
        self.attention=nn.MultiheadAttention(width,4,batch_first=True,dropout=0)
        self.norm=nn.LayerNorm(width)
        self.ffn=nn.Sequential(nn.Linear(width,2*width),nn.ReLU(),nn.Linear(2*width,width))
        self.norm2=nn.LayerNorm(width)
        self.readout=nn.Sequential(nn.Linear(width,2*width),nn.LayerNorm(2*width),nn.ReLU())
        self.temporal=temporal
        self.history_length=history
        self.history=[]
        if temporal:
            self.history_proj=nn.Sequential(nn.Linear(sum(DIMS.values())+5,width),nn.LayerNorm(width),nn.ReLU())
            self.position=nn.Parameter(torch.zeros(history,width))
            self.transformer=nn.TransformerEncoder(nn.TransformerEncoderLayer(
                width,4,2*width,dropout=0,batch_first=True),1,enable_nested_tensor=False)
    def reset_history(self): self.history=[]
    def _inject_links(self, graph, h):
        edge_ms=('machine','has_slot','slot')
        if edge_ms in graph.edge_types and graph[edge_ms].edge_index.numel()>0:
            src,dst=graph[edge_ms].edge_index
            slot=h['slot'].clone()
            slot.index_add_(0,dst,self.slot_host_proj(h['machine'][src]))
            h['slot']=slot
        edge_os=('order','contains','subbatch')
        if edge_os in graph.edge_types and graph[edge_os].edge_index.numel()>0:
            src,dst=graph[edge_os].edge_index
            sub=h['subbatch'].clone()
            sub.index_add_(0,dst,self.subbatch_order_proj(h['order'][src]))
            h['subbatch']=sub
        edge_fm=('machine','fits_subbatch','subbatch')
        if edge_fm in graph.edge_types and graph[edge_fm].edge_index.numel()>0 and len(h['subbatch'])>0:
            src,dst=graph[edge_fm].edge_index
            sub=h['subbatch'].clone()
            deg=torch.zeros((len(sub),1),dtype=sub.dtype,device=sub.device)
            msg=torch.zeros_like(sub)
            msg.index_add_(0,dst,self.subbatch_machine_proj(h['machine'][src]))
            deg.index_add_(0,dst,torch.ones((len(dst),1),dtype=sub.dtype,device=sub.device))
            h['subbatch']=sub+msg/deg.clamp_min(1.0)
        edge_ot=('order','has_candidate','template')
        if edge_ot in graph.edge_types and graph[edge_ot].edge_index.numel()>0:
            src,dst=graph[edge_ot].edge_index
            tpl=h['template'].clone()
            tpl.index_add_(0,dst,self.template_order_proj(h['order'][src]))
            h['template']=tpl
    def forward(self, graph):
        h={k:self.proj[k](graph[k].x.clone() if graph[k].x.is_inference() else graph[k].x)+self.type_emb[k] for k in DIMS}
        keys=list(h); sizes=[len(h[k]) for k in keys]
        ctx=graph.event_context.reshape(1,5).to(h['pool'])
        if ctx.is_inference(): ctx=ctx.clone()
        h['pool']=h['pool']+self.context(ctx)
        self._inject_links(graph,h)
        tokens=torch.cat([h[k] for k in keys]).unsqueeze(0)
        attended,_=self.attention(tokens,tokens,tokens,need_weights=False)
        tokens=self.norm(tokens+attended)
        tokens=self.norm2(tokens+self.ffn(tokens)).squeeze(0)
        if self.temporal:
            if not hasattr(graph,'external_history'):
                raw=torch.cat([graph[k].x.mean(0) if len(graph[k].x) else
                               graph[k].x.new_zeros(DIMS[k]) for k in keys]+[ctx.flatten()])
                self.history=(self.history+[raw.detach().cpu().clone()])[-self.history_length:]
                graph.external_history=torch.stack(self.history)
            history=graph.external_history.clone().to(tokens)
            x=self.history_proj(history)+self.position[-len(history):]
            mask=torch.triu(torch.ones(len(x),len(x),device=x.device,dtype=torch.bool),1)
            temporal=self.transformer(x.unsqueeze(0),mask=mask)[0,-1]
            tokens=tokens+temporal
        chunks=tokens.split(sizes)
        result=dict(zip(keys,chunks))
        return result,self.readout(result['pool'].mean(0,keepdim=True))

def distribution(scores, valid, mode, exploration):
    valid=valid.to(device=scores.device,dtype=torch.bool)
    if not valid.any(): raise ValueError('Empty feasible action set')
    if not torch.isfinite(scores[valid]).all(): raise FloatingPointError('Nonfinite legal score')
    masked=scores.masked_fill(~valid,-torch.inf)
    if mode=='q':
        probs=valid.to(scores.dtype)*exploration/valid.sum()
        probs=probs.scatter_add(0,masked.argmax().reshape(1),scores.new_tensor([1-exploration]))
    else:
        # The mutation is part of the behaviour policy and its log probability.
        # No post-sampling action replacement with an incorrect policy gradient.
        probs=(1-exploration)*masked.softmax(0)+exploration*valid/valid.sum()
    return Categorical(probs=probs),masked

class SplitHead(ManagerPolicy):
    def __init__(self,width,mode='actor',exploration=0):
        super().__init__(2*width,width,10,width,width)
        self.mode=mode; self.exploration=exploration
    def forward(self,g,h_order,h_templates,templates,template_mask,order_W,**kwargs):
        _,scores=super().forward(g,h_order,h_templates,templates,template_mask,order_W)
        valid=torch.tensor([bool(template_mask.get(t,0)) for t in templates],device=g.device)
        return distribution(scores,valid,self.mode,self.exploration if self.training else 0)

class PlacementHead(WorkerPolicy):
    def __init__(self,width,mode='actor',exploration=0):
        super().__init__(width,width,width)
        self.mode=mode; self.exploration=exploration
    def forward(self,h_batch,h_slots,slot_keys,valid_slots,order_id=None):
        _,scores=super().forward(h_batch,h_slots,slot_keys,valid_slots,order_id)
        valid=torch.tensor([key in set(valid_slots) for key in slot_keys],device=h_batch.device)
        return distribution(scores,valid,self.mode,self.exploration if self.training else 0)
