import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
import torch
from baselines.external_rl import METHODS
from baselines.external_rl.trainer import ExternalTrainer,Config
from baselines.external_rl.networks import distribution
from baselines.external_rl.run import (
    scenarios,manager_for,train_episode,evaluate,seed_all,fingerprint,resolve_budget,canonical_run_dir,main as cli_main
)
from baselines.external_rl.stage_launcher import run_stage_script
from baselines.external_rl.evolution import teach_event

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.paths,_=scenarios('train',1)
    def setUp(self): seed_all(1)
    def test_method_ids_and_matched_budget(self):
        self.assertEqual(METHODS['iddqn'],'ATTENTION-DDQN-PCB')
        cfg=Config()
        self.assertEqual(resolve_budget('am_a2c','train',1800,10,'matched_sim',cfg),(1800,10,1))
        self.assertEqual(resolve_budget('etdqn','train',1800,10,'matched_sim',cfg),(200,1,9))
        self.assertEqual(resolve_budget('etdqn','train',1800,10,'episodes',cfg),(1800,10,9))
        self.assertEqual(resolve_budget('etdqn','smoke',1800,10,'matched_sim',cfg),(1,1,9))
    def test_masks_and_mutation_likelihood(self):
        q=torch.tensor([2.,1000.,-1.],requires_grad=True)
        mask=torch.tensor([True,False,True])
        d,_=distribution(q,mask,'actor',.2)
        self.assertEqual(d.probs[1].item(),0.)
        self.assertTrue(torch.all(d.sample((100,))!=1))
        expected=.8*torch.softmax(q[[0,2]],0)+.1
        torch.testing.assert_close(d.probs[[0,2]],expected)
        (-d.log_prob(torch.tensor(0))).backward()
        self.assertEqual(q.grad[1].item(),0.)
        with self.assertRaises(ValueError): distribution(q,torch.zeros(3,dtype=torch.bool),'q',.1)
    def test_variable_q_candidates(self):
        for count in [1,7,93]:
            q=torch.arange(count,dtype=torch.float)
            d,_=distribution(q,torch.ones(count,dtype=torch.bool),'q',.1)
            self.assertEqual(d.probs.argmax().item(),count-1)
    def test_real_actor_gradient_and_readonly_eval(self):
        t=ExternalTrainer('am_a2c',device='cpu'); train_episode(t,self.paths[0],1)
        before={k:v.clone() for k,v in t.model.state_dict().items()}
        length=len(t.manager_buffer); rng=torch.get_rng_state().clone()
        evaluate(t,self.paths[:1],1,'val',1,1)
        self.assertEqual(length,len(t.manager_buffer))
        for k,v in t.model.state_dict().items(): torch.testing.assert_close(v,before[k])
        torch.testing.assert_close(torch.get_rng_state(),rng)
        t.update_policies()
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in t.model.state_dict().items() if k.startswith('0.')))
    def test_temporal_replay_uses_collected_history(self):
        t=ExternalTrainer('transformer_a3c',device='cpu'); train_episode(t,self.paths[0],1)
        row=t.rows()[-1]; self.assertGreater(len(row['graph'].external_history),1)
        a=t.forward_row(row)[1]; t.hgnn.reset_history(); b=t.forward_row(row)[1]
        torch.testing.assert_close(a,b)
    def test_ddqn_checkpoint_roundtrip_and_target(self):
        t=ExternalTrainer('iddqn',device='cpu'); train_episode(t,self.paths[0],1); t.update_policies()
        self.assertTrue(t.replay)
        self.assertTrue(all(not p.requires_grad for p in t.target.parameters()))
        self.assertTrue(all(nxt is None if r['done'] else nxt['role']==r['role'] for r,nxt in t.replay))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt'; t.save(path,{'shared_files':fingerprint()})
            loaded,meta=ExternalTrainer.load(path,device='cpu')
            for a,b in zip(t.model.parameters(),loaded.model.parameters()): torch.testing.assert_close(a,b)
            self.assertEqual(len(t.replay),len(loaded.replay))
    def test_teacher_isolates_live_manager(self):
        t=ExternalTrainer('etdqn',Config(ga_population=2,ga_generations=1),device='cpu')
        m=manager_for(self.paths[0],1)
        ids=list(m.pending_event_ids); makespan=m.current_solution.makespan
        report=teach_event(t,m)
        self.assertEqual(list(m.pending_event_ids),ids)
        self.assertEqual(m.current_solution.makespan,makespan)
        self.assertIsNone(m.active_context)
        self.assertEqual(report['teacher_simulations'],2)
        self.assertTrue(t.replay)
        self.assertTrue(all('teacher_return' in r for r,_ in t.replay))
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_device_end_to_end(self):
        for method,cfg in [('am_a2c',Config()),('iddqn',Config()),('etdqn',Config(ga_population=2,ga_generations=1))]:
            t=ExternalTrainer(method,cfg,device='cuda')
            self.assertEqual(t.device.type,'cuda')
            res=train_episode(t,self.paths[0],1)
            self.assertGreater(res['env_simulations'],0)
            t.update_policies()
            rows=evaluate(t,self.paths[:1],1,'val',1,1)
            self.assertEqual(len(rows),1)
            self.assertEqual(next(t.model.parameters()).device.type,'cuda')
    def test_resume_training_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'run_out'
            cli_main(['--method','am_a2c','--mode','train','--stage','1','--seed','1',
                      '--episodes','2','--validate-every','2','--budget-mode','episodes',
                      '--device','cpu','--output',str(out)])
            meta1=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(meta1['completed_episodes'],2)
            cli_main(['--method','am_a2c','--mode','train','--stage','1','--seed','1',
                      '--episodes','4','--validate-every','2','--budget-mode','episodes',
                      '--device','cpu','--resume','--output',str(out)])
            meta2=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(meta2['completed_episodes'],4)
            with (out/'train.csv').open(encoding='utf-8-sig',newline='') as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),4)
            self.assertEqual([int(r['episode']) for r in rows],[0,1,2,3])
    def test_stage_launcher_and_canonical_paths(self):
        import sys
        p=canonical_run_dir('am_a2c',2,3)
        self.assertTrue(str(p).endswith(str(Path('result')/'external_rl'/'am_a2c'/'stage2'/'seed_3')))
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'stage_launcher_out'
            old_argv=sys.argv
            try:
                sys.argv=['train_stage2.py','--episodes','1','--validate-every','1','--device','cpu','--output',str(out)]
                run_stage_script(fixed_stage=2,fixed_method='am_a2c',selected_seed=3)
            finally:
                sys.argv=old_argv
            meta=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['stage'],2)
            self.assertEqual(meta['seed'],3)
            self.assertEqual(meta['method'],'am_a2c')
if __name__=='__main__': unittest.main()
