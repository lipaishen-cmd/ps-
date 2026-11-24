import copy
import random
from dataclasses import dataclass, asdict
import torch
from torch import nn
from hppo_trainer import HPPOTrainer
from critics import SharedValueCritic
from . import METHODS
from .networks import AttentionEncoder, SplitHead, PlacementHead

@dataclass
class Config:
    width:int=32
    lr:float=0.0003
    gamma:float=1.0
    entropy:float=0.0015
    value_weight:float=0.25
    max_grad_norm:float=0.5
    mutation:float=0.05
    epsilon:float=0.1
    replay_size:int=2000
    batch_size:int=32
    updates:int=4
    target_tau:float=0.01
    history:int=8
    ga_population:int=4
    ga_generations:int=2
    ga_genes:int=512
    ga_mutation:float=0.15

class ExternalTrainer(HPPOTrainer):
    """Reuse only the existing episode collector; never call PPO.update_policies.

    Paper-specific adaptations are explicit in README; method IDs carry PCB.
    """
    def __init__(self,method,config=None,device=None):
        self.config=config or Config(); c=self.config
        if method not in METHODS: raise ValueError(method)
        if c.width%4 or c.width<=0: raise ValueError('width must be a positive multiple of 4')
        if not 0<=c.mutation<1 or not 0<=c.epsilon<1: raise ValueError('invalid exploration')
        if not 0<c.target_tau<=1 or c.batch_size<1 or c.replay_size<1: raise ValueError('invalid replay')
        target_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        super().__init__([], 'Pool_D',d_h=c.width,lr=c.lr,gamma=c.gamma,gae_lambda=0.95,device=target_device)
        self.method=method; self.method_id=METHODS[method]; self.replay=[]
        q=method in ('etdqn','iddqn')
        self.hgnn=AttentionEncoder(c.width,method=='transformer_a3c',c.history)
        self.manager_policy=SplitHead(c.width,'q' if q else 'actor',c.epsilon if q else (c.mutation if method=='am_a2c' else 0))
        self.worker_policy=PlacementHead(c.width,'q' if q else 'actor',c.epsilon if q else (c.mutation if method=='am_a2c' else 0))
        self.shared_critic=SharedValueCritic(2*c.width,c.width,c.width,c.width)
        self.manager_critic=self.worker_critic=self.shared_critic
        self.model=nn.ModuleList([self.hgnn,self.manager_policy,self.worker_policy,self.shared_critic]).to(self.device)
        self.all_params=list(self.model.parameters())
        self.optimizer=torch.optim.Adam(self.all_params,lr=c.lr)
        self.target=copy.deepcopy(self.model).eval().to(self.device)
        for p in self.target.parameters(): p.requires_grad_(False)
        self.training_updates=0
    def collect_dynamic_episode(self,*args,**kwargs):
        self.hgnn.reset_history()
        return super().collect_dynamic_episode(*args,**kwargs)
    def rows(self):
        rows=[]
        for role,b in [('split',self.manager_buffer),('placement',self.worker_buffer)]:
            b.compute_returns_and_advantages(0.)
            for i in range(len(b)):
                r=dict(role=role,graph=b.graph_list[i].clone().cpu(),action=b.actions[i],
                       reward=b.rewards[i],done=b.dones[i],ret=b.returns[i],
                       adv=b.advantages[i],
                       tau=b.taus[i] if role=='split' else 1)
                if role=='split':
                    r.update(index=b.frontier_idx_list[i],templates=b.templates_list[i],
                             mask=b.mask_list[i],order_W=b.order_W_list[i])
                else:
                    r.update(index=b.order_idx_list[i],sub_idx=b.sub_idx_list[i],
                             slot_keys=b.slot_keys_list[i],valid_slots=b.valid_slots_list[i])
                rows.append(r)
        return rows
    def forward_row(self,r,model=None):
        model=self.model if model is None else model
        dev=next(model.parameters()).device
        h,g=model[0](r['graph'].to(dev))
        order=h['order'][r['index']:r['index']+1]
        if r['role']=='split':
            dist,q=model[1](g,order,h['template'],r['templates'],r['mask'],r['order_W'])
            value=model[3](g.detach(),order.detach()).squeeze()
        else:
            batch=h['subbatch'][r['sub_idx']:r['sub_idx']+1]
            dist,q=model[2](batch,h['slot'],r['slot_keys'],r['valid_slots'])
            value=model[3](g.detach(),order.detach(),batch.detach()).squeeze()
        return dist,q,value
    def actor_loss(self):
        rows=self.rows()
        if not rows: return None
        adv_by_role={'split':[],'placement':[]}
        evaluated=[]
        for r in rows:
            dist,_,value=self.forward_row(r)
            ret=value.new_tensor(r['ret'])
            raw_adv=value.new_tensor(r.get('adv',r['ret']-float(value.detach())))
            adv_by_role[r['role']].append(raw_adv)
            evaluated.append((r,dist,value,ret,raw_adv))
        stats={}
        for role,vals in adv_by_role.items():
            if len(vals)>1:
                t=torch.stack(vals)
                stats[role]=(t.mean(),t.std(unbiased=False).clamp_min(1e-6))
            else:
                stats[role]=None
        role_losses={'split':[],'placement':[]}
        for r,dist,value,ret,raw_adv in evaluated:
            st=stats[r['role']]
            adv=(raw_adv-st[0])/st[1] if st is not None else raw_adv.clamp(-5.0,5.0)
            action=torch.tensor(r['action'],dtype=torch.long,device=dist.probs.device)
            v_loss=nn.functional.smooth_l1_loss(value,ret,beta=1.0)
            step_loss=(-dist.log_prob(action)*adv
                       +self.config.value_weight*v_loss
                       -self.config.entropy*dist.entropy())
            role_losses[r['role']].append(step_loss)
        active=[torch.stack(v).mean() for v in role_losses.values() if v]
        return torch.stack(active).mean() if active else None
    def _sample_replay_batch(self):
        split_items=[x for x in self.replay if x[0]['role']=='split']
        place_items=[x for x in self.replay if x[0]['role']=='placement']
        bs=min(len(self.replay),self.config.batch_size)
        if not split_items or not place_items:
            return random.sample(self.replay,bs)
        n_split=min(len(split_items),max(1,bs//2))
        n_place=min(len(place_items),bs-n_split)
        return random.sample(split_items,n_split)+random.sample(place_items,n_place)
    def optimize(self,loss):
        self.optimizer.zero_grad(); loss.backward()
        norm=nn.utils.clip_grad_norm_(self.all_params,self.config.max_grad_norm,error_if_nonfinite=True)
        self.optimizer.step(); self.training_updates+=1
        return {'loss':float(loss.detach()),'gradient_norm':float(norm)}
    def update_policies(self):
        if self.method in ('am_a2c','transformer_a3c'):
            loss=self.actor_loss(); out={} if loss is None else self.optimize(loss)
        else:
            if self.method=='iddqn':
                rows=self.rows()
                for i,r in enumerate(rows):
                    nxt=rows[i+1] if i+1<len(rows) and rows[i+1]['role']==r['role'] else None
                    if not r['done'] and nxt is None:
                        raise RuntimeError('Nonterminal Q transition missing next state')
                    self.replay.append((r,nxt if not r['done'] else None))
            self.replay=self.replay[-self.config.replay_size:]
            out={}
            for _ in range(self.config.updates):
                if not self.replay: break
                losses={'split':[],'placement':[]}
                for r,nxt in self._sample_replay_batch():
                    _,q,_=self.forward_row(r)
                    with torch.no_grad():
                        if self.method=='etdqn': y=q.new_tensor(r['teacher_return'])
                        else:
                            y=q.new_tensor(r['reward'])
                            if nxt is not None:
                                _,online,_=self.forward_row(nxt)
                                _,target,_=self.forward_row(nxt,self.target)
                                y=y+self.gamma**r['tau']*target[online.argmax()]
                    losses[r['role']].append(nn.functional.smooth_l1_loss(q[r['action']],y))
                active=[torch.stack(v).mean() for v in losses.values() if v]
                out=self.optimize(torch.stack(active).mean())
                if self.method=='iddqn':
                    with torch.no_grad():
                        for t,p in zip(self.target.parameters(),self.model.parameters()):
                            t.lerp_(p,self.config.target_tau)
        self.manager_buffer.reset(); self.worker_buffer.reset()
        return out
    def save(self,path,metadata):
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.save(dict(format='external-rl-pcb-v1',method=self.method,config=asdict(self.config),
                        device=str(self.device),
                        model=self.model.state_dict(),target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(),updates=self.training_updates,
                        replay=self.replay,metadata=metadata,
                        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),
                        cuda_rng=cuda_rng),path)
    @classmethod
    def load(cls,path,device=None,restore_rng=False):
        # Only load locally generated, trusted checkpoints (graphs require pickle).
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['format']!='external-rl-pcb-v1': raise ValueError('checkpoint schema mismatch')
        obj=cls(payload['method'],Config(**payload['config']),device=device)
        obj.model.load_state_dict(payload['model']); obj.target.load_state_dict(payload['target'])
        obj.optimizer.load_state_dict(payload['optimizer']); obj.replay=payload['replay']
        obj.training_updates=payload['updates']
        if restore_rng:
            if 'python_rng' in payload: random.setstate(payload['python_rng'])
            if 'torch_rng' in payload: torch.set_rng_state(payload['torch_rng'])
            if payload.get('cuda_rng') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(payload['cuda_rng'])
        return obj,payload['metadata']
