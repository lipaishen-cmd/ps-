"""Evolutionary fixed-target teacher, adapted to the current PCB event.

Fitness uses only the current event reward from the common physical runner.
No rollout sees a future arrival. Extra simulator calls are counted explicitly.
Individual 0 in Generation 0 is seeded with the current Q-network's greedy policy
(gene < 0 -> argmax Q), and subsequent individuals explore via mutation/crossover
around the Q-guided elite so the teacher monotonically improves upon the online Q.
"""
import copy
import random
import torch
from torch import nn
from torch.distributions import Categorical
from dynamic_two_agent_runner import DynamicTwoAgentRunner

class TapeHead(nn.Module):
    def __init__(self,head,tape):
        super().__init__(); self.head=head; self.tape=tape
    def forward(self,*args,**kwargs):
        _,q=self.head(*args,**kwargs)
        legal=torch.where(torch.isfinite(q))[0]
        i=self.tape['index']; self.tape['index']+=1
        if i>=len(self.tape['genes']): raise RuntimeError('GA chromosome too short')
        g=self.tape['genes'][i]
        if g<0.0:
            choice=q.argmax()
        else:
            choice=legal[min(len(legal)-1,int(g*len(legal)))]
        probs=torch.zeros_like(q); probs[choice]=1
        return Categorical(probs=probs),q

def teach_event(trainer,manager):
    c=trainer.config
    if c.ga_population<2 or c.ga_generations<1: raise ValueError('invalid GA budget')
    q_seed=[-1.0 for _ in range(c.ga_genes)]
    population=[q_seed]
    for _ in range(c.ga_population-1):
        population.append([random.random() if random.random()<max(c.ga_mutation,0.25) else -1.0
                           for _ in range(c.ga_genes)])
    best=None; calls=0; successful=0; scored=[]
    for _ in range(c.ga_generations):
        scored=[]
        for genes in population:
            trial=copy.deepcopy(manager)
            # Only run_next_event; never collect_dynamic_episode here.
            tape={'index':0,'genes':genes}
            runner=DynamicTwoAgentRunner(trial,trainer.hgnn,
                TapeHead(trainer.manager_policy,tape),TapeHead(trainer.worker_policy,tape),
                trainer.shared_critic,device=trainer.device,reward_mode=trainer.reward_mode,reward_scale=10.)
            with torch.no_grad(): d=runner.run_next_event(deterministic=True)
            calls+=1
            fitness=float(d.reward) if d.committed else -float('inf')
            scored.append((fitness,genes,d))
            if d.committed:
                successful+=1
                if best is None or fitness>best[0]: best=(fitness,genes,d)
        scored.sort(key=lambda x:x[0],reverse=True)
        elites=[x[1] for x in scored[:max(1,c.ga_population//2)]]
        population=[list(elites[0])]
        while len(population)<c.ga_population:
            a,b=random.choice(elites),random.choice(elites)
            child=[random.random() if random.random()<c.ga_mutation else
                   (x if random.random()<.5 else y) for x,y in zip(a,b)]
            population.append(child)
    if best is not None:
        d=best[2]
        for role,records in [('split',d.manager_records),('placement',d.worker_records)]:
            ret=0.
            for idx in reversed(range(len(records))):
                record=records[idx]
                reward=(float(record['step_reward']) if role=='placement' else 0.)
                if idx==len(records)-1: reward+=float(d.reward)
                tau=record['tau'] if role=='split' else 1
                ret=reward+trainer.gamma**tau*ret
                r=dict(role=role,graph=record['graph'].clone().cpu(),action=record['action_idx'],
                       teacher_return=ret,index=record['frontier_idx'] if role=='split' else record['order_idx'])
                if role=='split': r.update(templates=record['templates'],mask=record['mask'],order_W=record['order_W'])
                else: r.update(sub_idx=record['sub_idx'],slot_keys=record['slot_keys'],valid_slots=record['valid_slots'])
                trainer.replay.append((r,None))
        trainer.replay=trainer.replay[-c.replay_size:]
    return dict(teacher_simulations=calls,teacher_successes=successful)
