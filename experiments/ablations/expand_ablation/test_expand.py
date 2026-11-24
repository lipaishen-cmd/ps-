"""Unit and regression tests for L11 (DAPPO-Expand) C2 ablation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]
for p in (str(THIS_DIR), str(PROJECT_ROOT), str(PROJECT_ROOT / "agent"), str(PROJECT_ROOT / "environment")):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_models import POOL_D
from dynamic_event_manager import DynamicEventManager
from dynamic_validator import load_dynamic_orders_and_events
from expanded_graph_builder import PreExpandedPCBGraphBuilder, TimedOnDemandPCBGraphBuilder
from dappo_expand_trainer import (
    DAPPO_EXPAND_METHOD_ID,
    DAPPOExpandTrainer,
    ExpandedTwoAgentRunner,
)
from evaluate_expand import compute_convergence_indicators, load_config, select_scenarios
from hppo_trainer import HPPOTrainer
from partition_template_generator import generate_partition_templates


class L11ExpandAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, _ = load_config()
        scenarios, _ = select_scenarios(split="test", stages=(1,), scenario_ids=("scenario_test_000",))
        cls.scenario_item = scenarios[0]
        cls.init_orders, cls.events = load_dynamic_orders_and_events(
            str(cls.scenario_item["path"]), cls.config["sheet_name"]
        )

    def _build_split_graphs(self):
        manager = DynamicEventManager(
            pool_type=POOL_D,
            initial_orders=self.init_orders,
            events=self.events,
            instance_name=self.config["sheet_name"],
            k_scope=self.config["default_k_scope"],
            k_seed=self.config["default_k_seed"],
            delta_k=self.config["delta_k"],
            formal_mode=True,
        )
        context = manager.trigger_next_event()
        new_order = context.event.order
        orders = list(context.orders.values())
        frontier_idx = [o.order_id for o in orders].index(new_order.order_id)
        templates = generate_partition_templates(
            new_order,
            max_templates=self.config["max_partition_templates"],
            machines=manager.machines_dict,
        )
        mask = {tpl: 1 for tpl in templates}

        ondemand_builder = TimedOnDemandPCBGraphBuilder(
            orders=orders, pool_type=POOL_D, machines_dict=manager.machines_dict
        )
        expanded_builder = PreExpandedPCBGraphBuilder(
            orders=orders, pool_type=POOL_D, machines_dict=manager.machines_dict
        )
        g_ondemand = ondemand_builder.build_graph(
            solution=context.base_solution,
            current_frontier_idx=frontier_idx,
            candidate_templates=templates,
            template_mask=mask,
            event_id=context.event.event_id,
            event_time=context.event_time,
            scope_ratio=context.scope_ratio,
        )
        g_expanded = expanded_builder.build_graph(
            solution=context.base_solution,
            current_frontier_idx=frontier_idx,
            candidate_templates=templates,
            template_mask=mask,
            event_id=context.event.event_id,
            event_time=context.event_time,
            scope_ratio=context.scope_ratio,
        )
        return manager, context, new_order, templates, g_ondemand, g_expanded

    def test_split_stage_pre_expansion_vs_on_demand(self) -> None:
        _, _, new_order, templates, g_ondemand, g_expanded = self._build_split_graphs()
        expected_subbatches = sum(len(tpl) for tpl in templates)
        self.assertGreater(expected_subbatches, 0)
        self.assertEqual(g_ondemand["subbatch"].x.size(0), 0)
        self.assertEqual(g_ondemand[("template", "instantiates", "subbatch")].edge_index.size(1), 0)

        self.assertEqual(g_expanded["subbatch"].x.size(0), expected_subbatches)
        self.assertEqual(g_expanded["subbatch"].x.size(1), 5)
        self.assertEqual(
            g_expanded[("template", "instantiates", "subbatch")].edge_index.size(1),
            expected_subbatches,
        )
        self.assertEqual(
            g_expanded[("order", "contains", "subbatch")].edge_index.size(1),
            expected_subbatches,
        )
        # Precedence edges only occur within the same candidate template, never across templates.
        expected_precedence = sum(max(0, len(tpl) - 1) for tpl in templates)
        prec_ei = g_expanded[("subbatch", "precedes", "subbatch")].edge_index
        self.assertEqual(prec_ei.size(1), expected_precedence)
        tpl_ids = g_expanded["subbatch"].template_id.tolist()
        for col in range(prec_ei.size(1)):
            u, v = int(prec_ei[0, col].item()), int(prec_ei[1, col].item())
            self.assertEqual(tpl_ids[u], tpl_ids[v])
            self.assertEqual(u + 1, v)

        self.assertGreater(g_expanded[("subbatch", "fits_machine", "machine")].edge_index.size(1), 0)
        self.assertGreater(g_expanded[("subbatch", "can_insert", "slot")].edge_index.size(1), 0)
        self.assertGreater(g_expanded.num_total_edges, g_ondemand.num_total_edges)
        del new_order

    def test_hgnn_forward_and_end_to_end_gradient_on_expanded_graph(self) -> None:
        _, _, new_order, templates, _, g_expanded = self._build_split_graphs()
        trainer = DAPPOExpandTrainer(orders=[*self.init_orders, *(e.order for e in self.events)], pool_type=POOL_D, d_h=32)
        trainer.hgnn.train()
        trainer.manager_policy.train()
        trainer.optimizer_encoder.zero_grad()
        trainer.optimizer_split.zero_grad()

        embeddings, global_emb = trainer.hgnn(g_expanded)
        self.assertEqual(
            trainer.hgnn.last_consumed_edge_types,
            trainer.hgnn.PROPAGATED_EDGE_TYPES,
        )
        frontier_idx = int((g_expanded[("order", "has_candidate", "template")].edge_index[0, 0]).item())
        h_order = embeddings["order"][frontier_idx : frontier_idx + 1]
        h_templates = embeddings["template"]
        dist, _ = trainer.manager_policy(
            g=global_emb,
            h_order=h_order,
            h_templates=h_templates,
            templates=templates,
            template_mask={tpl: 1 for tpl in templates},
            order_W=new_order.W,
        )
        loss = -dist.log_prob(torch.tensor(0))
        loss.backward()

        hgnn_grad = any(
            p.grad is not None and torch.any(p.grad.abs() > 0)
            for p in trainer.hgnn.parameters()
        )
        mgr_grad = any(
            p.grad is not None and torch.any(p.grad.abs() > 0)
            for p in trainer.manager_policy.parameters()
        )
        self.assertTrue(hgnn_grad)
        self.assertTrue(mgr_grad)

    def test_trainer_parameter_parity_and_checkpoint_isolation(self) -> None:
        all_orders = [*self.init_orders, *(e.order for e in self.events)]
        main_trainer = HPPOTrainer(orders=all_orders, pool_type=POOL_D, d_h=32)
        exp_trainer = DAPPOExpandTrainer(orders=all_orders, pool_type=POOL_D, d_h=32)

        main_actor_params = sum(p.numel() for p in main_trainer.actor_params)
        exp_actor_params = sum(p.numel() for p in exp_trainer.actor_params)
        self.assertEqual(main_actor_params, 8802)
        self.assertEqual(exp_actor_params, main_actor_params)
        self.assertEqual(exp_trainer.method_id, DAPPO_EXPAND_METHOD_ID)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            main_ckpt = tmp / "main.pt"
            exp_ckpt = tmp / "expand.pt"
            main_trainer.save_checkpoint(str(main_ckpt), extra_meta={"stage": 1, "seed": 1})
            exp_trainer.save_checkpoint(str(exp_ckpt), extra_meta={"stage": 1, "seed": 1})

            loaded_payload = exp_trainer.load_checkpoint(str(exp_ckpt), load_optimizer=True)
            self.assertEqual(loaded_payload["method_id"], DAPPO_EXPAND_METHOD_ID)
            self.assertEqual(loaded_payload["graph_construction_mode"], "pre_expand_all_candidate_templates")

            with self.assertRaises(ValueError):
                exp_trainer.load_checkpoint(str(main_ckpt), load_optimizer=False)
            with self.assertRaises(ValueError):
                main_trainer.load_checkpoint(str(exp_ckpt), load_optimizer=False)

    def test_config_and_single_event_runner_telemetry(self) -> None:
        self.assertEqual(self.config["schema"], "expand_ablation_v1")
        self.assertEqual(self.config["claim"], "C2")
        all_orders = [*self.init_orders, *(e.order for e in self.events)]
        trainer = DAPPOExpandTrainer(orders=all_orders, pool_type=POOL_D, d_h=32)
        manager = DynamicEventManager(
            pool_type=POOL_D,
            initial_orders=self.init_orders,
            events=self.events,
            instance_name=self.config["sheet_name"],
            k_scope=self.config["default_k_scope"],
            k_seed=self.config["default_k_seed"],
            delta_k=self.config["delta_k"],
            formal_mode=True,
        )
        runner = ExpandedTwoAgentRunner(
            manager=manager,
            hgnn=trainer.hgnn,
            manager_policy=trainer.manager_policy,
            worker_policy=trainer.worker_policy,
            value_critic=trainer.shared_critic,
            max_templates=self.config["max_partition_templates"],
        )
        decision = runner.run_next_event(deterministic=True)
        self.assertTrue(decision.committed)
        tel = getattr(decision, "graph_telemetry", None)
        self.assertIsNotNone(tel)
        self.assertGreater(tel["split_graph_build_ms"], 0.0)
        self.assertGreater(tel["split_subbatch_nodes_mean"], 0.0)
        self.assertGreater(tel["event_hgnn_forward_ms"], 0.0)

    def test_convergence_indicators(self) -> None:
        val_hist = [
            {"episode": 10, "mean_actor_cmax": 400.0, "autonomous_feasibility_rate": 1.0},
            {"episode": 20, "mean_actor_cmax": 320.0, "autonomous_feasibility_rate": 1.0},
            {"episode": 30, "mean_actor_cmax": 301.0, "autonomous_feasibility_rate": 1.0},
            {"episode": 40, "mean_actor_cmax": 300.0, "autonomous_feasibility_rate": 1.0},
        ]
        res = compute_convergence_indicators(val_hist)
        self.assertEqual(res["episodes_to_within_1pct_final"], 30.0)
        self.assertIsNotNone(res["normalized_val_auc"])
        self.assertGreater(res["normalized_val_auc"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
