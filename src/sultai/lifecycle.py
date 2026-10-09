"""Bounded same-feature handover mechanics, separate from formation evidence.

All schedules and seeds below were declared before held-out evaluation. The
host and additional module both use the original input's tanh features: they
could be algebraically combined, but this assay permits only task SGD updates.
"""

import hashlib
import json
import math
import random
from typing import Callable, Sequence

from .repair import Adapter, CHANNELS, Example, PARAMETERS, finite, vector

INITIAL_STEPS = 8
HOLD_STEPS = 16
TAPER_STEPS = 48
POST_STEPS = 64
REMOVAL_STEP = INITIAL_STEPS + HOLD_STEPS + TAPER_STEPS
TOTAL_STEPS = REMOVAL_STEP + POST_STEPS
BATCH_SIZE = 16
LEARNING_RATE = 0.8
CONDITIONING_COUNT = 64
AUDIT_COUNT = 64
POST_REMOVAL_MSE_TARGET = 0.01
TASK_COEFFICIENTS = (0.25, -0.20, 0.15, 0.10)
ARMS = ("no_growth", "cold_zero_taper", "random_taper",
        "early_static_retained", "early_static_taper", "formed_taper")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


class _Trainable:
    """272 allocated values, including zero-valued values; plain SGD has no state."""

    def __init__(self, adapter: Adapter | None = None):
        adapter = Adapter.zero() if adapter is None else adapter
        self.weights = [list(row) for row in adapter.weights]
        self.bias = list(adapter.bias)

    def residual(self, features: Sequence[float]) -> tuple[float, ...]:
        return tuple(finite(sum(w * x for w, x in zip(row, features)) + bias)
                     for row, bias in zip(self.weights, self.bias))

    def serialized(self) -> dict:
        return {"weights": self.weights, "bias": self.bias}


class _Model:
    def __init__(self):
        self.host = _Trainable()
        self.optimizer_state: dict = {}  # Stateless SGD, including auxiliary modules.

    def add(self, adapter: Adapter, *, trainable: bool) -> None:
        if hasattr(self, "temporary"):
            raise ValueError("additional module already allocated")
        self.temporary = _Trainable(adapter) if trainable else adapter

    def remove(self) -> None:
        if hasattr(self, "temporary"):
            del self.temporary  # Actual object/weight removal, not an alpha-zero mask.

    @property
    def parameters(self) -> int:
        return PARAMETERS * (1 + int(hasattr(self, "temporary")))

    def predict(self, h: Sequence[float], alpha: float) -> tuple[float, ...]:
        features = tuple(math.tanh(value) for value in h)
        host_delta = self.host.residual(features)
        extra = (0.0,) * CHANNELS
        if hasattr(self, "temporary") and alpha:
            if isinstance(self.temporary, _Trainable):
                extra = self.temporary.residual(features)
            else:
                extra = tuple(v - x for v, x in zip(self.temporary.apply(h), h))
        return vector(tuple(x + delta + alpha * aux
                            for x, delta, aux in zip(h, host_delta, extra)))

    def serialized(self) -> dict:
        payload = {"host": self.host.serialized(), "optimizer_state": self.optimizer_state}
        if hasattr(self, "temporary"):
            module = self.temporary
            payload["adapter"] = (module.serialized() if isinstance(module, _Trainable)
                                  else json.loads(module.to_json()))
        # Return detached data, so later changes cannot silently alter evidence.
        return json.loads(json.dumps(payload, allow_nan=False))


def _sgd_step(model: _Model, batch: Sequence[Example], alpha: float) -> int:
    """Exact minibatch gradient of mean output MSE. Never consumes teacher weights."""
    if not batch:
        raise ValueError("SGD requires a nonempty task batch")
    gradient = [[0.0] * CHANNELS for _ in range(CHANNELS)]
    bias_gradient = [0.0] * CHANNELS
    scale = 2.0 / (len(batch) * CHANNELS)
    for example in batch:
        features = tuple(math.tanh(value) for value in example.h)
        prediction = model.predict(example.h, alpha)
        for i, (actual, target) in enumerate(zip(prediction, example.target)):
            error = (actual - target) * scale
            bias_gradient[i] += error
            for j, value in enumerate(features):
                gradient[i][j] += error * value
    modules = [(model.host, 1.0)]
    if hasattr(model, "temporary") and isinstance(model.temporary, _Trainable):
        modules.append((model.temporary, alpha))
    for module, influence in modules:
        for i in range(CHANNELS):
            module.bias[i] = finite(module.bias[i] - LEARNING_RATE * influence * bias_gradient[i])
            for j in range(CHANNELS):
                module.weights[i][j] = finite(module.weights[i][j] -
                                               LEARNING_RATE * influence * gradient[i][j])
    return len(modules)


def _alpha(completed_steps: int) -> float:
    """Influence at a completed-step boundary; deletion occurs at alpha zero."""
    elapsed = completed_steps - INITIAL_STEPS
    if elapsed <= HOLD_STEPS:
        return 1.0
    return max(0.0, 1.0 - (elapsed - HOLD_STEPS) / TAPER_STEPS)


def _examples(seed: int, name: str, count: int, target: Adapter) -> tuple[Example, ...]:
    rng = random.Random(seed)
    return tuple(Example(f"lifecycle:{name}:{i}", h, target.apply(h))
                 for i in range(count)
                 for h in [tuple(rng.uniform(-2.0, 2.0) for _ in range(CHANNELS))])


def _random_adapter() -> Adapter:
    rng = random.Random(7004)
    return Adapter(tuple(tuple(rng.uniform(-0.025, 0.025) for _ in range(CHANNELS))
                         for _ in range(CHANNELS)),
                   tuple(rng.uniform(-0.025, 0.025) for _ in range(CHANNELS)))


def _train_arm(name: str, former: Callable[[Sequence[Example]], Adapter],
               training: Sequence[Example], conditioning: Sequence[Example],
               audit_inputs: Sequence[Sequence[float]]) -> tuple[dict, dict]:
    """This function receives audit inputs only: no held-out target access exists."""
    model = _Model()
    if name.startswith("early_static"):
        model.add(Adapter.zero(), trainable=True)
    counts = {"before": PARAMETERS, "at_step_zero": model.parameters,
              "peak": model.parameters}
    predictions: dict[str, tuple[tuple[float, ...], ...]] = {}
    stage_state: dict[str, dict] = {}
    host_updates = auxiliary_updates = 0
    residual_digest = None
    insertion_unchanged = True
    withdrawal = None

    def record(stage: str, step: int, alpha: float) -> None:
        predictions[stage] = tuple(model.predict(h, alpha) for h in audit_inputs)
        stage_state[stage] = {"completed_host_updates": step, "alpha": alpha,
                              "allocated_parameters": model.parameters}

    for step in range(TOTAL_STEPS + 1):
        influence = 1.0 if name == "early_static_retained" else _alpha(step)
        if name == "no_growth":
            influence = 0.0
        if step == INITIAL_STEPS:
            record("pre_add", step, influence)
            before = _digest(model.host.serialized())
            if name == "formed_taper":
                residual = tuple(Example(example.sample_id, example.h,
                                         tuple(x + target - host for x, target, host in
                                               zip(example.h, example.target,
                                                   model.predict(example.h, 0.0))))
                                 for example in conditioning)
                residual_digest = _digest([(e.sample_id, e.h, e.target) for e in residual])
                candidate = former(residual)
                if not isinstance(candidate, Adapter):
                    raise TypeError("former must return a repair.Adapter")
                model.add(candidate, trainable=False)
                del candidate
                # Same fixed host: withdrawal changes only adapter influence.
                withdrawal = {"with_adapter": tuple(model.predict(h, 1.0) for h in audit_inputs),
                              "without_adapter": tuple(model.predict(h, 0.0) for h in audit_inputs)}
            elif name == "cold_zero_taper":
                model.add(Adapter.zero(), trainable=True)
            elif name == "random_taper":
                model.add(_random_adapter(), trainable=True)
            insertion_unchanged = before == _digest(model.host.serialized())
            counts["peak"] = max(counts["peak"], model.parameters)
            record("immediate_add", step, influence)
        if step == INITIAL_STEPS + HOLD_STEPS + TAPER_STEPS // 2:
            record("mid_taper", step, influence)
        if step == REMOVAL_STEP:
            record("pre_remove", step, influence)
            if name not in ("no_growth", "early_static_retained"):
                model.remove()
            record("immediate_remove", step, influence)
        if step == TOTAL_STEPS:
            record("post_learning", step, influence)
            break
        batch = training[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
        updated = _sgd_step(model, batch, influence)
        host_updates += 1
        auxiliary_updates += updated - 1
    counts["final"] = model.parameters
    final = model.serialized()
    serial_count = sum(len(row) for row in final["host"]["weights"]) + len(final["host"]["bias"])
    if "adapter" in final:
        serial_count += sum(len(row) for row in final["adapter"]["weights"]) + len(final["adapter"]["bias"])
    report = {
        "counts": counts, "serialized_parameter_count": serial_count,
        "host_updates": host_updates, "auxiliary_gradient_updates": auxiliary_updates,
        "training_examples": len(training), "training_example_ids_sha256": _digest([e.sample_id for e in training]),
        "training_pairs_sha256": _digest([(e.sample_id, e.h, e.target) for e in training]),
        "initial_host_sha256": _digest(_Trainable().serialized()),
        "host_unchanged_at_insertion": insertion_unchanged,
        "formation_conditioning_sha256": residual_digest,
        "formation_examples": len(conditioning) if name == "formed_taper" else 0,
        "additional_module_training": "frozen" if name == "formed_taper" else
                                      "absent" if name == "no_growth" else "task SGD",
        "stage_state": stage_state, "final_serialization_keys": sorted(final),
        "final_serialization_sha256": _digest(final),
        "adapter_absent": "adapter" not in final,
        "optimizer_state_empty": final["optimizer_state"] == {},
        "retained_budget_matches_host": counts["final"] == PARAMETERS,
    }
    return report, {"predictions": predictions, "withdrawal": withdrawal}


def _prediction_mse(predictions: Sequence[Sequence[float]], audit: Sequence[Example]) -> float:
    if len(predictions) != len(audit) or not audit:
        raise ValueError("audit prediction count mismatch")
    return finite(sum((actual - target) ** 2 for prediction, example in zip(predictions, audit)
                      for actual, target in zip(prediction, example.target)) /
                  (len(audit) * CHANNELS))


def _run_task(former: Callable[[Sequence[Example]], Adapter], training: Sequence[Example],
              conditioning: Sequence[Example], audit: Sequence[Example], formation_label: str) -> dict:
    if len(training) != TOTAL_STEPS * BATCH_SIZE or len(conditioning) != CONDITIONING_COUNT or len(audit) != AUDIT_COUNT:
        raise ValueError("fixed lifecycle example counts required")
    ids = [e.sample_id for group in (training, conditioning, audit) for e in group]
    if len(ids) != len(set(ids)):
        raise ValueError("lifecycle splits must have unique, disjoint sample IDs")
    inputs = tuple(example.h for example in audit)
    # Every arm finishes all training before any audit target is inspected.
    trained = {name: _train_arm(name, former, training, conditioning, inputs) for name in ARMS}
    arms = {}
    audit_queries = 0
    for name, (report, raw) in trained.items():
        report["mse"] = {stage: _prediction_mse(predictions, audit)
                         for stage, predictions in raw["predictions"].items()}
        audit_queries += len(raw["predictions"]) * len(audit)
        if raw["withdrawal"] is not None:
            report["frozen_host_withdrawal_mse"] = {
                key: _prediction_mse(predictions, audit)
                for key, predictions in raw["withdrawal"].items()}
            audit_queries += 2 * len(audit)
        report["post_removal_target_met"] = (report["adapter_absent"] and
                                            report["mse"]["post_learning"] <= POST_REMOVAL_MSE_TARGET)
        arms[name] = report
    return {
        "assay": "planted same-feature handover mechanics",
        "formation_label": formation_label,
        "insertion_policy": "raw candidate inserted on fixed schedule; no admission selection in this assay",
        "config": {"initial_steps": INITIAL_STEPS, "hold_steps": HOLD_STEPS,
                   "taper_steps": TAPER_STEPS, "removal_step": REMOVAL_STEP,
                   "post_steps": POST_STEPS, "total_host_updates": TOTAL_STEPS,
                   "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
                   "loss": "mean over examples and output channels MSE",
                   "task_coefficients": TASK_COEFFICIENTS,
                   "training_seed": 7001, "conditioning_seed": 7002, "audit_seed": 7003,
                   "random_adapter_seed": 7004, "conditioning_examples": CONDITIONING_COUNT,
                   "audit_examples": AUDIT_COUNT, "post_removal_mse_target": POST_REMOVAL_MSE_TARGET},
        "arms": arms,
        "cost": {"host_gradient_steps": TOTAL_STEPS * len(ARMS),
                 "auxiliary_gradient_steps": sum(a["auxiliary_gradient_updates"] for a in arms.values()),
                 "task_example_gradient_uses": len(training) * len(ARMS),
                 "formation_calls": 1, "formation_conditioning_examples": CONDITIONING_COUNT,
                 "selection_queries": 0, "final_audit_queries": audit_queries,
                 "fixed_candidate_count": 1},
        "acceptance": {"formed_post_removal_target_met": arms["formed_taper"]["post_removal_target_met"],
                       "all_final_removal_arms_absent": all(arms[name]["adapter_absent"] for name in ARMS
                                                           if name != "early_static_retained"),
                       "insertion_preserves_host_weights": all(a["host_unchanged_at_insertion"] for a in arms.values()),
                       "serialized_counts_match_allocations": all(a["serialized_parameter_count"] == a["counts"]["final"]
                                                                   for a in arms.values()),
                       "no_adapter_optimizer_state": all(a["optimizer_state_empty"] for a in arms.values()),
                       "common_future_examples": len({a["training_pairs_sha256"] for a in arms.values()}) == 1,
                       "common_host_update_count": all(a["host_updates"] == TOTAL_STEPS for a in arms.values())},
        "summary": {"post_learning_mse": {name: a["mse"]["post_learning"] for name, a in arms.items()},
                    "formed_immediate_add_mse": arms["formed_taper"]["mse"]["immediate_add"],
                    "formed_immediate_remove_mse": arms["formed_taper"]["mse"]["immediate_remove"],
                    "formed_peak_parameters": arms["formed_taper"]["counts"]["peak"],
                    "formed_final_parameters": arms["formed_taper"]["counts"]["final"],
                    "retained_static_final_parameters": arms["early_static_retained"]["counts"]["final"]},
        "limits": ["full host allocated from step zero, not near-empty growth",
                   "host and adapter share nonlinear features and can be algebraically combined",
                   "host only learns from task gradients; no teacher weight copying or folding",
                   "full-host SGD can move residuals outside the generator's four-template family",
                   "retained early static arm has unmatched 544 final parameters",
                   "removing 272 temporary values is not half-size success against a tuned reference",
                   "no learned removal policy, unseen-family transfer, or developmental advantage claim"],
    }


def run_lifecycle(former: Callable[[Sequence[Example]], Adapter], *, formation_label: str = "frozen learned paired generator") -> dict:
    """Run fixed task SGD; former sees only current-host residual conditioning pairs."""
    from .formation import materialize
    target = materialize(TASK_COEFFICIENTS)
    return _run_task(former, _examples(7001, "training", TOTAL_STEPS * BATCH_SIZE, target),
                     _examples(7002, "conditioning", CONDITIONING_COUNT, target),
                     _examples(7003, "audit", AUDIT_COUNT, target), formation_label)


def affine_folding_diagnostic() -> dict:
    """A zero-bias residual h + W h folds exactly into following map L."""
    weights = ((0.2, -0.3), (0.4, 0.1))
    following = ((0.7, -0.2), (-0.5, 0.8))
    identity_plus = tuple(tuple(w + int(i == j) for j, w in enumerate(row))
                          for i, row in enumerate(weights))
    folded = tuple(tuple(sum(following[i][k] * identity_plus[k][j] for k in range(2))
                         for j in range(2)) for i in range(2))
    probes = ((-1.5, 0.4), (0.0, 0.0), (0.8, 1.2))
    errors = []
    for h in probes:
        residual = tuple(sum(row[j] * h[j] for j in range(2)) for row in identity_plus)
        sequential = tuple(sum(row[j] * residual[j] for j in range(2)) for row in following)
        combined = tuple(sum(row[j] * h[j] for j in range(2)) for row in folded)
        errors.extend(abs(a - b) for a, b in zip(sequential, combined))
    return {"max_absolute_error": max(errors), "identity_verified": max(errors) < 1e-12,
            "folded_linear_map": folded, "adapter_bias": (0.0, 0.0),
            "interpretation": "linear reparameterization control; no added representational capacity",
            "convolution_caveat": "nonzero bias folding requires boundary semantics"}


def signed_probe_diagnostic(epsilon: float = 0.25) -> dict:
    """Passive ambiguity and sign-reversed probe equality survive nonlinearity."""
    epsilon = finite(epsilon)
    if epsilon <= 0:
        raise ValueError("positive epsilon required")
    # Shared downstream nonlinear readout g(u)=u+u^2; mirrored upstream signs.
    def a(z: float) -> float:
        return z + z * z
    def b(z: float) -> float:
        return -z + z * z
    passive_a, passive_b = (0.0, a(0.0), 1.0), (0.0, b(0.0), 1.0)
    change_a = a(epsilon) - a(0.0)
    change_b = b(epsilon) - b(0.0)
    return {"epsilon": epsilon, "passive_evidence_a": passive_a,
            "passive_evidence_b": passive_b, "passive_identical": passive_a == passive_b,
            "a_plus": a(epsilon), "b_minus": b(-epsilon),
            "signed_probe_identity_verified": a(epsilon) == b(-epsilon),
            "a_plus_change": change_a, "b_plus_change": change_b,
            "sum_same_sign_changes": finite(change_a + change_b),
            "finite_changes_are_opposite": change_a == -change_b,
            "interpretation": "g(u)=u+u^2 counterexample; finite nonlinear changes need not be opposite",
            "limit": "this fixture does not show that every probe distinguishes every host"}


def run_diagnostics() -> dict:
    return {"affine_folding": affine_folding_diagnostic(), "signed_probe": signed_probe_diagnostic()}
