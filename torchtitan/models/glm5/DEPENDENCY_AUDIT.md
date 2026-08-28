# GLM-5 dependency and regression contract

This document defines the interfaces that connect the device-independent
TorchTitan GLM-5 implementation to TorchTitanTurbo and torchtitan-test. Read it
before changing GLM-5 code or an upstream common component used by GLM-5.

## Repository ownership

- `torchtitan`: model mathematics, configuration, state-dict conversion, and
  PyTorch-native parallelism integration. Do not add NPU-only patches here.
- `TorchTitanTurbo`: NPU compatibility patches, graph workarounds, and
  NPU-specific optimized implementations.
- `torchtitan-test`: launchers, parity, precision, checkpoint, stability,
  smoke, graph, performance, combination experiments, reports, and artifacts.

The three repositories are source-installed together. There is no stable ABI
between arbitrary commits, so every cross-repository change must record and
test the exact three source revisions used by an experiment.

## Interfaces consumed downstream

| TorchTitan surface | Downstream consumers | Required audit |
|---|---|---|
| `Glm5Model.forward`, token/position/mask shapes | parity traces, fixed-token input, CP/PP smoke and precision capture | parity CPU tests, single/PP/CP smoke, fixed-input contract |
| `glm5_configs`, `glm5_debugmodel`, model config fields | Turbo GLM patch, every test launcher | Turbo config conversion, command generation, all topology validation |
| `make_router_config`, `TokenChoiceTopKRouter` behavior | `NpuGlm5TokenChoiceTopKRouter` | reference-vs-NPU router contract test, EP smoke/parity |
| `Glm5StateDictAdapter` keys and placements | HF parity, checkpoint conversion, TP-sharded export | adapter unit tests, HF round trip, TP distributed adapter test |
| `parallelize.py` and `sharding.py` | smoke, precision, graph and performance topology suites | single plus every affected topology; inspect TP/CP/PP/EP combinations |
| trainer/config/dataloader/checkpoint CLI fields used by GLM | all torchtitan-test workflows | search every generated command and update all experiment families |
| artifact-visible module names | exploratory parity checkpoints and reports | trace schema compatibility or an explicit schema-version bump |

## Required change procedure

1. Identify whether the changed symbol is model-local or comes from
   `torchtitan.models.common`, Trainer, config, data, checkpoint, or distributed
   infrastructure.
2. Search TorchTitanTurbo for imports, subclasses, monkey patches, copied
   forward logic, and config conversion involving that symbol.
3. Search torchtitan-test for imports, generated CLI arguments, trace paths,
   fixture contracts, report fields, and compatibility assertions.
4. Update or add a behavioral contract test. A successful import is not enough
   for a copied or patched algorithm.
5. Run focused CPU tests, then the affected GPU/NPU smoke and numerical
   experiment. Model-math changes require parity/precision evidence; launcher
   changes require command and lifecycle tests.
6. Update this file and the corresponding downstream dependency documents when
   an interface or repository responsibility changes.

Do not silently catch an interface-mismatch `ImportError`. Optional model
absence and an installed-but-incompatible API are different states and must be
reported separately.
