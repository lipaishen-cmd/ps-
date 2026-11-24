"""Training/evaluation CLI, using the active split manifest and shared simulator."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from . import ROOT, METHODS
from .trainer import ExternalTrainer, Config
from dynamic_validator import load_dynamic_orders_and_events
from dynamic_event_manager import DynamicEventManager
from dynamic_two_agent_runner import DynamicTwoAgentRunner

PROTECTED=('agent/dynamic_event_manager.py','agent/scope_selector.py',
           'environment/decoder.py','agent/atomic_transaction.py','agent/reward.py',
           'agent/dynamic_two_agent_runner.py','agent/partition_template_generator.py',
           'agent/graph_builder.py','agent/template_encoding.py','environment/time_model.py')
PAPER_BOUNDARY_STATEMENT=(
    '为统一动作空间与物理约束，本文参考 AM-A2C、Transformer-A3C、ETDQN 及注意力增强 DDQN 的'
    '表征与更新机制，在相同的 PCB 分批—放置双阶段决策框架、合法动作掩码和标量奖励下构建适配基线'
    '（分别记作 AM-A2C-PCB、TRANSFORMER-A3C-PCB、ETDQN-PCB 和 ATTENTION-DDQN-PCB）。'
)
RUN_ARTIFACTS=('best.pt','last.pt','manifest.json','train.csv','results.csv')
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fingerprint(): return {p:digest(ROOT/p) for p in PROTECTED}
def default_device(): return 'cuda' if torch.cuda.is_available() else 'cpu'
def canonical_run_dir(method,stage,seed):
    return ROOT/'result'/'external_rl'/method/f'stage{stage}'/f'seed_{seed}'
def sim_multiplier_for(method,config):
    return (1+config.ga_population*config.ga_generations) if method=='etdqn' else 1
def resolve_budget(method,mode,episodes,validate_every,budget_mode,config):
    mult=sim_multiplier_for(method,config)
    if mode=='smoke':
        eff_ep=2 if method=='transformer_a3c' else 1
        return eff_ep,eff_ep,mult
    if budget_mode=='matched_sim':
        eff_ep=max(1,episodes//mult)
        eff_val=max(1,round(validate_every/mult))
        return eff_ep,eff_val,mult
    return episodes,validate_every,mult
def scenarios(split,stage):
    pointer=json.loads((ROOT/'data/active_experiment_package.json').read_text(encoding='utf-8'))
    folder=ROOT/pointer['dataset_relative_path']; path=folder/pointer['manifest_filename']
    if digest(path)!=pointer['manifest_sha256']: raise ValueError('Active manifest hash mismatch')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    rows=manifest['splits'][split]
    result=[]
    for r in rows:
        if stage and not r['filepath'].startswith('stage'+str(stage)+'/'): continue
        p=(folder/r['filepath']).resolve()
        if not p.is_relative_to(folder.resolve()): raise ValueError('Path outside dataset')
        if digest(p)!=r['sha256']: raise ValueError('Scenario hash mismatch: '+str(p))
        result.append(p)
    if not result: raise ValueError('Empty split/stage')
    return result,pointer
def manager_for(path,max_events=None):
    orders,events=load_dynamic_orders_and_events(str(path),'Sheet1')
    if max_events is not None: events=events[:max_events]
    return DynamicEventManager('Pool_D',orders,events,instance_name='Sheet1',formal_mode=True)
def seed_all(seed):
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def outcome(manager,decisions):
    success=(len(decisions)==len(manager.events) and all(d.committed for d in decisions)
             and not manager.pending_event_ids and manager.active_context is None)
    return dict(completed=success,final_cmax=float(manager.current_solution.makespan) if success else None,
                controlled_failures=sum(not d.committed for d in decisions),
                expansions=sum(d.expansion_count for d in decisions),
                events=len(decisions),split_actions=sum(len(d.manager_records) for d in decisions),
                placement_actions=sum(len(d.worker_records) for d in decisions))
def train_episode(trainer,path,max_events=None,clear_buffer=True):
    trainer.model.train(); manager=manager_for(path,max_events)
    extra={'teacher_simulations':0,'teacher_successes':0}
    if trainer.method=='etdqn':
        from .evolution import teach_event
        decisions=[]
        while manager.pending_event_ids:
            report=teach_event(trainer,manager)
            for key in extra: extra[key]+=report[key]
            runner=DynamicTwoAgentRunner(manager,trainer.hgnn,trainer.manager_policy,
                trainer.worker_policy,trainer.shared_critic,device=trainer.device,
                reward_mode=trainer.reward_mode,reward_scale=10.)
            with torch.no_grad(): d=runner.run_next_event(deterministic=False)
            decisions.append(d)
            if not d.committed:
                manager.rollback_active(); break
    else: decisions=trainer.collect_dynamic_episode(manager,clear_buffer=clear_buffer)
    result=outcome(manager,decisions); result.update(extra)
    result['env_simulations']=result['events']+result['teacher_simulations']
    return result
def evaluate(trainer,paths,seed,split,stage,max_events=None):
    flags=[m.training for m in trainer.model]
    trainer.model.eval(); rows=[]
    try:
        with torch.no_grad():
            for path in paths:
                manager=manager_for(path,max_events); start=time.perf_counter()
                decisions=trainer.collect_dynamic_episode(manager,deterministic=True,record_rollout=False)
                row=outcome(manager,decisions)
                row.update(method_id=trainer.method_id,seed=seed,split=split,stage=stage,
                           scenario=str(path),elapsed_seconds=time.perf_counter()-start,
                           checkpoint_type='external_pcb_adaptation',
                           evidence_status='SMOKE_ONLY' if max_events else 'ADAPTED_METHOD_NOT_PAPER_REPLICATION')
                rows.append(row)
    finally:
        for m,flag in zip(trainer.model,flags): m.train(flag)
    return rows
def score(rows):
    ok=[r['final_cmax'] for r in rows if r['completed']]
    return (len(ok),-sum(ok)/len(ok) if ok else -float('inf'),-max(ok) if ok else -float('inf'))
def write_csv(path,rows):
    if not rows: return
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
def read_csv_rows(path):
    if not Path(path).is_file(): return []
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--mode',choices=['train','evaluate','smoke'],default='train')
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,default=None,help='Optional output dir (defaults to result/external_rl/<method>/stage<S>/seed_<seed>)')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--stage',type=int,choices=[1,2,3],default=1)
    p.add_argument('--seed',type=int,choices=[1,2,3,4,5],default=1)
    p.add_argument('--episodes',type=int,default=1800)
    p.add_argument('--budget-mode',choices=['matched_sim','episodes'],default='matched_sim')
    p.add_argument('--device',type=str,default=default_device())
    p.add_argument('--resume',action='store_true',help='Resume training from --output/last.pt or --checkpoint')
    p.add_argument('--fresh-start',action='store_true',help='Clear previous run artifacts in --output and start from episode 0')
    p.add_argument('--formal',action='store_true',help='Require completed paper-fidelity and matched-budget audit')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--validate-every',type=int,default=10)
    args=p.parse_args(argv)
    if args.formal:
        p.error('Formal paper-comparison release is not certified: see fidelity.json and README. '
                'These runnable PCB adaptations must not be presented as exact paper reproductions.')
    if args.episodes<1 or args.validate_every<1: p.error('positive episode/validation budget required')
    if args.resume and args.fresh_start: p.error('--resume and --fresh-start cannot be used together')
    if args.mode!='train' and args.resume: p.error('--resume is only valid in --mode train')
    explicit_output=args.output is not None
    if args.output is None:
        base_dir=canonical_run_dir(args.method,args.stage,args.seed)
        if args.mode=='train': args.output=base_dir
        elif args.mode=='smoke': args.output=ROOT/'result'/'external_rl'/args.method/f'smoke_stage{args.stage}_seed{args.seed}'
        else: args.output=base_dir/f'eval_{args.split}'
    hashes=fingerprint()
    train_paths,pointer=scenarios('train',args.stage)
    val_paths,_=scenarios('val',args.stage)
    if args.mode=='evaluate':
        ckpt=args.checkpoint or (canonical_run_dir(args.method,args.stage,args.seed)/'best.pt')
        if not ckpt.is_file(): p.error(f'Evaluation checkpoint not found: {ckpt}')
        if args.fresh_start and args.output.exists():
            for name in RUN_ARTIFACTS: (args.output/name).unlink(missing_ok=True)
            shutil.rmtree(args.output/'tensorboard',ignore_errors=True)
        args.output.mkdir(parents=True,exist_ok=args.fresh_start)
        trainer,old=ExternalTrainer.load(ckpt,device=args.device)
        if trainer.method!=args.method or old['shared_files']!=hashes or old['dataset']!=pointer:
            raise ValueError('Method/environment/dataset checkpoint mismatch')
        if old.get('mode')=='smoke': raise ValueError('Smoke checkpoint cannot enter full evaluation')
        if old['stage']!=args.stage: raise ValueError('Checkpoint stage mismatch')
        if old['seed']!=args.seed: raise ValueError('Checkpoint seed mismatch')
        paths,_=scenarios(args.split,args.stage)
        rows=evaluate(trainer,paths,args.seed,args.split,args.stage)
        meta=dict(method=args.method,method_id=trainer.method_id,config=asdict(trainer.config),
                  device=str(trainer.device),seed=args.seed,stage=args.stage,split=args.split,
                  protocol='external-pcb-adaptation-v1',shared_files=hashes,dataset=pointer,
                  mode='evaluate',checkpoint=str(ckpt),
                  paper_boundary_statement=PAPER_BOUNDARY_STATEMENT)
        (args.output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        write_csv(args.output/'results.csv',rows)
        if hashes!=fingerprint(): raise RuntimeError('Shared environment changed during evaluation')
        print(json.dumps({'method':trainer.method_id,'split':args.split,'stage':args.stage,'seed':args.seed,
                          'successful':sum(r['completed'] for r in rows),'total':len(rows)},ensure_ascii=False))
        return
    config=Config(**json.loads(args.config.read_text(encoding='utf-8'))) if args.config else Config()
    eff_episodes,eff_val_every,sim_mult=resolve_budget(
        args.method,args.mode,args.episodes,args.validate_every,args.budget_mode,config)
    has_last=(args.output/'last.pt').is_file()
    has_manifest=(args.output/'manifest.json').is_file()
    auto_resume=(args.mode=='train' and not explicit_output and not args.fresh_start and has_last and has_manifest)
    do_resume=args.resume or auto_resume
    start_ep=0; logs=[]; best=None; cum_sims=0
    resumed_ok=False
    if do_resume:
        resume_ckpt=args.checkpoint if args.checkpoint else (args.output/'last.pt')
        if not resume_ckpt.is_file(): p.error(f'Resume checkpoint not found: {resume_ckpt}')
        try:
            trainer,old=ExternalTrainer.load(resume_ckpt,device=args.device,restore_rng=True)
            resumed_ok=True
        except (RuntimeError,ValueError) as exc:
            if args.resume or not auto_resume:
                raise
            print(f"[ExternalRL] 检测到旧版不兼容断点 ({exc.__class__.__name__})，自动清理旧产物并从 Episode 0 全新开始: {args.output}",flush=True)
            for name in RUN_ARTIFACTS: (args.output/name).unlink(missing_ok=True)
            shutil.rmtree(args.output/'tensorboard',ignore_errors=True)
            shutil.rmtree(args.output/'tensorboard_sync',ignore_errors=True)
    if resumed_ok:
        if trainer.method!=args.method or old['shared_files']!=hashes or old['dataset']!=pointer:
            raise ValueError('Method/environment/dataset checkpoint mismatch on resume')
        if old.get('mode')!='train': raise ValueError('Only train checkpoints can be resumed')
        if old['stage']!=args.stage or old['seed']!=args.seed:
            raise ValueError('Resume stage/seed mismatch')
        if old.get('budget_mode','episodes')!=args.budget_mode:
            raise ValueError('Resume budget-mode mismatch')
        config=trainer.config
        start_ep=int(old.get('completed_episodes',0))
        if eff_episodes<start_ep:
            raise ValueError('Resume target episodes smaller than already completed episodes')
        cum_sims=int(old.get('cumulative_env_simulations',0))
        raw_best=old.get('best_score')
        if raw_best is not None:
            b=tuple(raw_best)
            best=(b[0],b[1],b[1]) if len(b)==2 else b
        else:
            best=None
        logs=[r for r in read_csv_rows(args.output/'train.csv') if int(r['episode'])<start_ep]
    else:
        if args.checkpoint and not auto_resume:
            p.error('Pass --resume with --checkpoint to resume training, or omit --checkpoint for fresh training.')
        if args.fresh_start and args.output.exists():
            for name in RUN_ARTIFACTS: (args.output/name).unlink(missing_ok=True)
            shutil.rmtree(args.output/'tensorboard',ignore_errors=True)
            shutil.rmtree(args.output/'tensorboard_sync',ignore_errors=True)
        elif args.output.exists() and any((args.output/name).exists() for name in ('best.pt','last.pt','train.csv','results.csv')):
            p.error(f'Run artifacts already exist in {args.output}; pass --resume to continue or --fresh-start to overwrite.')
        args.output.mkdir(parents=True,exist_ok=True)
        seed_all(args.seed)
        trainer=ExternalTrainer(args.method,config,device=args.device)
    print(f"[ExternalRL] method={trainer.method_id} | stage={args.stage} | seed={args.seed} | "
          f"device={trainer.device} | budget={args.budget_mode} ({start_ep}->{eff_episodes} ep, sim_mult={sim_mult}x) | "
          f"output={args.output}")
    meta=dict(method=args.method,method_id=trainer.method_id,config=asdict(config),
              device=str(trainer.device),seed=args.seed,
              protocol='external-pcb-adaptation-v1',shared_files=hashes,dataset=pointer,
              formal_ready=False,stage=args.stage,source_revision='see source_manifest.json',
              fidelity='Adaptation; not an exact reproduction of published method',
              paper_boundary_statement=PAPER_BOUNDARY_STATEMENT,
              budget_mode=args.budget_mode,requested_episodes=args.episodes,
              effective_episodes=eff_episodes,effective_validate_every=eff_val_every,
              sim_multiplier=sim_mult,completed_episodes=start_ep,
              cumulative_env_simulations=cum_sims,best_score=list(best) if best is not None else None,
              command={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              parameter_count=sum(p.numel() for p in trainer.model.parameters()),
              run_id=str(args.output.resolve()),
              source_hashes={str(p.relative_to(ROOT)):digest(p) for p in Path(__file__).parent.rglob('*.py')},
              mode=args.mode)
    (args.output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    smoke=args.mode=='smoke'; max_events=1 if smoke else None
    if smoke: train_paths=train_paths[:1]; val_paths=val_paths[:1]
    tb_dir=args.output/'tensorboard'
    shutil.rmtree(args.output/'tensorboard_sync',ignore_errors=True)
    had_tb_events=tb_dir.is_dir() and any(tb_dir.glob('*tfevents*'))
    writer=SummaryWriter(log_dir=str(tb_dir)) if (SummaryWriter is not None and not smoke) else None
    if writer is not None:
        print(f"[TensorBoard] 实时监控日志目录: {tb_dir}",flush=True)
    def log_tb_episode(row):
        if writer is None: return
        step=int(row['episode'])+1
        if row.get('final_cmax') not in (None,'','None','null'):
            writer.add_scalar('Train/Makespan_Cmax',float(row['final_cmax']),step)
        if row.get('loss') not in (None,'','None','null'):
            writer.add_scalar('Train/Total_Loss',float(row['loss']),step)
        if row.get('gradient_norm') not in (None,'','None','null'):
            writer.add_scalar('Train/Grad_Norm',float(row['gradient_norm']),step)
        writer.add_scalar('Train/Cumulative_Env_Simulations',int(row['cumulative_env_simulations']),step)
        if row.get('parameter_staleness') not in (None,'','None','null'):
            writer.add_scalar('Train/Parameter_Staleness',float(row['parameter_staleness']),step)
        writer.flush()
    if writer is not None and logs and not had_tb_events:
        for row in logs: log_tb_episode(row)
    def save_progress(completed_ep,val_rows=None):
        nonlocal best
        meta['completed_episodes']=completed_ep
        meta['cumulative_env_simulations']=cum_sims
        if val_rows is not None:
            current=score(val_rows)
            improved=best is None or current>best
            if improved:
                best=current
                meta['best_score']=list(best)
                trainer.save(args.output/'best.pt',meta)
            val_ok=sum(r['completed'] for r in val_rows)
            val_cmax=-current[1] if current[0]>0 else None
            val_s=f"{val_cmax:.2f}" if val_cmax is not None else "FAIL"
            best_s=f"{-best[1]:.2f}" if best is not None and best[0]>0 else "FAIL"
            tag=" *NEW BEST*" if improved else ""
            print(f"[Val @ Ep {completed_ep}/{eff_episodes}] ok={val_ok}/{len(val_rows)} "
                  f"val_cmax={val_s} | best={best_s} | sims={cum_sims}{tag}",flush=True)
            if writer is not None:
                writer.add_scalar('Val/Autonomous_Feasibility_Rate_Percent',100.0*val_ok/max(1,len(val_rows)),completed_ep)
                if val_cmax is not None:
                    writer.add_scalar('Val/Actor_Makespan_Cmax',float(val_cmax),completed_ep)
                writer.flush()
        trainer.save(args.output/'last.pt',meta)
        (args.output/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        write_csv(args.output/'train.csv',logs)
    try:
        if args.method=='transformer_a3c':
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            from .asynchronous import train_async
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
                for start in range(start_ep,eff_episodes,eff_val_every):
                    count=min(eff_val_every,eff_episodes-start)
                    chunk=train_async(trainer,train_paths,count,args.workers,args.seed,max_events,start,eff_episodes,pool=pool)
                    for row in chunk:
                        row.setdefault('env_simulations',row.get('events',0)+row.get('teacher_simulations',0))
                        cum_sims+=int(row['env_simulations'])
                        row['cumulative_env_simulations']=cum_sims
                        row['sim_multiplier']=sim_mult
                        log_tb_episode(row)
                        logs.append(row)
                    val_rows=evaluate(trainer,val_paths,args.seed,'val',args.stage,max_events)
                    save_progress(start+count,val_rows)
        else:
            batch_n=len(train_paths) if (args.method=='am_a2c' and not smoke) else 1
            for ep in range(start_ep,eff_episodes):
                path=train_paths[ep%len(train_paths)]; start_t=time.perf_counter()
                is_batch_start=(ep==start_ep) or (ep%batch_n==0)
                result=train_episode(trainer,path,max_events,clear_buffer=is_batch_start)
                if (ep+1)%batch_n==0 or ep==eff_episodes-1:
                    result.update(trainer.update_policies())
                cum_sims+=int(result['env_simulations'])
                elapsed=time.perf_counter()-start_t
                result.update(episode=ep,scenario=str(path),elapsed_seconds=elapsed,
                              cumulative_env_simulations=cum_sims,sim_multiplier=sim_mult)
                cmax=result.get('final_cmax')
                cmax_s=f"{cmax:.2f}" if cmax is not None else "FAIL"
                loss_v=result.get('loss')
                loss_s=f"{loss_v:.4f}" if loss_v is not None else "-"
                print(f"  [Ep {ep+1}/{eff_episodes}] ok={result['completed']} cmax={cmax_s} "
                      f"loss={loss_s} sims={cum_sims} time={elapsed:.1f}s",flush=True)
                log_tb_episode(result)
                logs.append(result)
                if (ep+1)%eff_val_every==0 or ep==eff_episodes-1:
                    val_rows=evaluate(trainer,val_paths,args.seed,'val',args.stage,max_events)
                    save_progress(ep+1,val_rows)
        if not (args.output/'best.pt').is_file():
            val_rows=evaluate(trainer,val_paths,args.seed,'val',args.stage,max_events)
            save_progress(eff_episodes,val_rows)
    finally:
        if writer is not None: writer.close()
    restored,_=ExternalTrainer.load(args.output/'best.pt',device=args.device)
    rows=evaluate(restored,val_paths,args.seed,'val',args.stage,max_events)
    write_csv(args.output/'train.csv',logs); write_csv(args.output/'results.csv',rows)
    if hashes!=fingerprint(): raise RuntimeError('Shared environment changed during run')
    print(json.dumps({'method':trainer.method_id,'device':str(trainer.device),
                      'updates':trainer.training_updates,'effective_episodes':eff_episodes,
                      'cumulative_env_simulations':cum_sims,
                      'successful':sum(r['completed'] for r in rows),'total':len(rows)},ensure_ascii=False))
if __name__=='__main__': main()
