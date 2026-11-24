"""A3C with process-isolated environments and asynchronous gradient delivery.

The parent owns Adam state. Each worker snapshots parameters BEFORE sampling;
the completed local gradient is applied without a barrier and its staleness is
logged. Whole-episode n-step returns preserve the shared reward boundaries.
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from contextlib import nullcontext
from pathlib import Path
import time
import torch
from .trainer import ExternalTrainer,Config

_WORKER_CACHE={}

def local_gradient(state,config,path,seed,max_events):
    from .run import seed_all,train_episode
    seed_all(seed)
    key=tuple(sorted(config.items()))
    trainer=_WORKER_CACHE.get(key)
    if trainer is None:
        trainer=ExternalTrainer('transformer_a3c',Config(**config),device='cpu')
        _WORKER_CACHE[key]=trainer
    trainer.manager_buffer.reset(); trainer.worker_buffer.reset()
    trainer.model.load_state_dict(state); start=time.perf_counter()
    result=train_episode(trainer,Path(path),max_events)
    loss=trainer.actor_loss()
    if loss is None: gradients=[None for p in trainer.model.parameters()]
    else:
        trainer.optimizer.zero_grad(); loss.backward()
        gradients=[None if p.grad is None else p.grad.detach().cpu().numpy().copy() for p in trainer.model.parameters()]
        result['loss']=float(loss.detach())
    trainer.manager_buffer.reset(); trainer.worker_buffer.reset()
    result['elapsed_seconds']=time.perf_counter()-start
    return gradients,result

def train_async(trainer,paths,episodes,workers,seed,max_events,offset=0,total_episodes=None,pool=None):
    from dataclasses import asdict
    if workers<2: raise ValueError('A3C requires at least two asynchronous workers')
    total=total_episodes if total_episodes is not None else (offset+episodes)
    logs=[]; submitted=0; pending={}
    ctx=nullcontext(pool) if pool is not None else ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'))
    with ctx as active_pool:
        def submit():
            nonlocal submitted
            ep=offset+submitted; submitted+=1
            state={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
            future=active_pool.submit(local_gradient,state,asdict(trainer.config),str(paths[ep%len(paths)]),seed*100000+ep,max_events)
            pending[future]=(ep,trainer.training_updates)
        for _ in range(min(workers,episodes)): submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                ep,version=pending.pop(future); gradients,result=future.result()
                staleness=trainer.training_updates-version
                result.update(episode=ep,parameter_staleness=staleness)
                trainer.optimizer.zero_grad()
                has_grad=False
                for param,grad in zip(trainer.model.parameters(),gradients):
                    if grad is not None:
                        param.grad=torch.from_numpy(grad).to(device=param.device,dtype=param.dtype).clone(); has_grad=True
                if has_grad:
                    norm=torch.nn.utils.clip_grad_norm_(trainer.model.parameters(),trainer.config.max_grad_norm,error_if_nonfinite=True)
                    trainer.optimizer.step(); trainer.training_updates+=1
                    result['gradient_norm']=float(norm)
                cmax=result.get('final_cmax')
                cmax_s=f"{cmax:.2f}" if cmax is not None else "FAIL"
                loss_v=result.get('loss')
                loss_s=f"{loss_v:.4f}" if loss_v is not None else "-"
                print(f"  [Ep {ep+1}/{total}] ok={result['completed']} cmax={cmax_s} "
                      f"loss={loss_s} staleness={staleness} time={result['elapsed_seconds']:.1f}s",flush=True)
                logs.append(result)
                if submitted<episodes: submit()
    return logs
