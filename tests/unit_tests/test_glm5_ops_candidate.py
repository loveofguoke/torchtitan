# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import unittest

import torch

from torchtitan.models.glm5.ops_candidate.communication.bucket_planner import (
    plan_communication_buckets,
)
from torchtitan.models.glm5.ops_candidate.communication.fsdp_layer_grouping import (
    ShardedLayerProfile,
    plan_fsdp_layer_groups,
)
from torchtitan.models.glm5.ops_candidate.communication.overlap_pipeline import (
    run_chunked_overlap,
)
from torchtitan.models.glm5.ops_candidate.communication.redistribution_audit import (
    MeshAxis,
    RedistributionTransition,
    audit_partial_to_replicate,
)
from torchtitan.models.glm5.ops_candidate.memory.workspace import (
    TensorWorkspacePool,
)
from torchtitan.models.glm5.ops_candidate.moe.expert_placement import (
    ExpertLoad,
    plan_expert_rank_placement,
)
from torchtitan.models.glm5.ops_candidate.pipeline.stage_partition import (
    PipelineUnitProfile,
    ideal_non_interleaved_1f1b_bubble_fraction,
    plan_contiguous_pipeline_stages,
)
from torchtitan.models.glm5.ops_candidate.recompute.selective_recompute import (
    CandidateActivation,
    RecomputeDecision,
    plan_selective_recompute,
)
from torchtitan.models.glm5.ops_candidate.serving.paged_kv_cache import (
    PagedKVBlockTable,
)


class _ImmediateHandle:
    def __init__(self, value):
        self.value = value

    def wait(self):
        return self.value


class Glm5CandidateWorkspaceTest(unittest.TestCase):
    def test_reuses_an_exact_shape_cpu_tensor(self) -> None:
        pool = TensorWorkspacePool(max_cached_bytes=1024)
        original = pool.acquire(
            (4, 8),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        pointer = original.data_ptr()

        pool.release(original)
        reused = pool.acquire(
            (4, 8),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertEqual(reused.data_ptr(), pointer)
        self.assertEqual(pool.cached_bytes, 0)

    def test_respects_capacity_and_tensor_contract(self) -> None:
        pool = TensorWorkspacePool(max_cached_bytes=8)
        too_large = torch.empty((4,), dtype=torch.float32)
        with_grad = torch.empty((1,), dtype=torch.float32, requires_grad=True)

        pool.release(too_large)
        pool.release(with_grad)

        self.assertEqual(pool.cached_bytes, 0)


class Glm5CandidateCommunicationTest(unittest.TestCase):
    def test_bucket_planner_aligns_and_preserves_parameter_spans(self) -> None:
        buckets = plan_communication_buckets(
            (("a", 3), ("b", 5), ("c", 2)),
            target_numel=8,
            alignment=4,
            reverse_registration_order=False,
        )

        self.assertEqual([(b.start, b.end) for b in buckets], [(0, 8), (8, 12)])
        self.assertEqual(
            [span.name for bucket in buckets for span in bucket.parameters],
            ["a", "b", "c"],
        )

    def test_audits_multi_axis_partial_to_replicate(self) -> None:
        audit = audit_partial_to_replicate(
            RedistributionTransition(
                "attention_output",
                (MeshAxis("fsdp", 2), MeshAxis("tp", 4)),
                ("fsdp", "tp"),
            )
        )

        self.assertTrue(audit.flatten_candidate)
        self.assertEqual(audit.current_all_reduce_count, 2)
        self.assertEqual(audit.flattened_all_reduce_count, 1)

    def test_fsdp_grouping_preserves_policy_boundaries(self) -> None:
        plan = plan_fsdp_layer_groups(
            (
                ShardedLayerProfile("dense.0", 10, 20, "dense"),
                ShardedLayerProfile("dense.1", 10, 20, "dense"),
                ShardedLayerProfile("moe.2", 30, 60, "moe"),
            ),
            layers_per_group=2,
        )

        self.assertEqual(
            [[layer.name for layer in group.layers] for group in plan.groups],
            [["dense.0", "dense.1"], ["moe.2"]],
        )
        self.assertEqual(plan.removable_data_collectives, 2)

    def test_overlap_prototype_makes_wait_points_visible(self) -> None:
        result = run_chunked_overlap(
            (1, 2),
            issue_dispatch=lambda _, value: _ImmediateHandle(value + 10),
            compute=lambda _, value: value * 2,
            issue_combine=lambda _, value: _ImmediateHandle(value - 1),
            merge=sum,
        )

        self.assertEqual(result.output, 44)
        self.assertEqual(
            [(event.chunk_id, event.phase) for event in result.events[:4]],
            [
                (0, "dispatch-issued"),
                (1, "dispatch-issued"),
                (0, "dispatch-ready"),
                (0, "compute-complete"),
            ],
        )


class Glm5CandidatePipelineTest(unittest.TestCase):
    def test_balances_contiguous_measured_stage_costs(self) -> None:
        plan = plan_contiguous_pipeline_stages(
            (
                PipelineUnitProfile("embedding", 4.0, 0.0),
                PipelineUnitProfile("layer.0", 2.0, 2.0),
                PipelineUnitProfile("layer.1", 2.0, 2.0),
                PipelineUnitProfile("loss", 1.0, 3.0),
            ),
            num_stages=2,
        )

        self.assertEqual(
            [[unit.name for unit in stage.units] for stage in plan.stages],
            [["embedding", "layer.0"], ["layer.1", "loss"]],
        )
        self.assertEqual(plan.max_stage_ms, 8.0)
        self.assertAlmostEqual(
            ideal_non_interleaved_1f1b_bubble_fraction(
                num_stages=8,
                num_microbatches=8,
            ),
            7 / 15,
        )


class Glm5CandidateMoETest(unittest.TestCase):
    def test_hot_experts_are_spread_across_ranks(self) -> None:
        plan = plan_expert_rank_placement(
            tuple(
                ExpertLoad(expert_id, count)
                for expert_id, count in enumerate((8, 7, 1, 1))
            ),
            num_ranks=2,
        )

        self.assertEqual(
            [assignment.predicted_tokens for assignment in plan.assignments],
            [9, 8],
        )
        self.assertLess(plan.imbalance_ratio, 1.1)


class Glm5CandidateRecomputeTest(unittest.TestCase):
    def test_discrete_outputs_are_mandatory(self) -> None:
        decisions = plan_selective_recompute(
            (
                CandidateActivation("topk", "discrete", 16, 1.0),
                CandidateActivation("projection", "tensor", 32, 100.0),
                CandidateActivation("norm", "tensor", 16, 2.0),
            ),
            save_budget_bytes=48,
        )

        self.assertEqual(decisions["topk"], RecomputeDecision.SAVE)
        self.assertEqual(decisions["projection"], RecomputeDecision.SAVE)
        self.assertEqual(decisions["norm"], RecomputeDecision.RECOMPUTE)


class Glm5CandidatePagedKVTest(unittest.TestCase):
    def test_forked_partial_block_uses_copy_on_write(self) -> None:
        table = PagedKVBlockTable(num_blocks=4, block_size=4)
        source = table.allocate_tokens("source", 3)
        table.fork("source", "target")
        target = table.allocate_tokens("target", 1)

        self.assertEqual(source.block_ids, (0,))
        self.assertEqual(target.block_ids, (1,))
        self.assertEqual(len(target.copies), 1)
        self.assertEqual(target.copies[0].source_block, 0)
        self.assertEqual(target.copies[0].target_block, 1)
        self.assertEqual(target.copies[0].num_tokens, 3)
        self.assertEqual(table.slot_mapping("source"), (0, 1, 2))
        self.assertEqual(table.slot_mapping("target"), (4, 5, 6, 7))

if __name__ == "__main__":
    unittest.main()
