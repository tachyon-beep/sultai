"""Bounded seen-family lineage assay; oracle pairs contain gradient information.

All settings below are fixed before evaluation. No task fitting occurs in
FrozenGenerator.form; full ridge teachers are offline training supervision.
"""

from dataclasses import asdict, dataclass
import math
import random
from typing import Mapping, Sequence

from .repair import Adapter, CHANNELS, Episode, Example, FrozenHost, finite, fit, mse


@dataclass(frozen=True)
class Protocol:
    train_seeds: tuple[int, ...] = tuple(range(1000, 1032))
    dev_seeds: tuple[int, ...] = tuple(range(2000, 2008))
    test_seeds: tuple[int, ...] = tuple(range(3000, 3008))
    conditioning_count: int = 64
    selection_count: int = 24
    test_count: int = 64
    coefficient_bound: float = 0.35
    input_bound: float = 2.0
    teacher_ridge: float = 1e-8
    meta_ridge: float = 0.01
    admission_improvement: float = 1e-10
    sgd_learning_rate: float = 4.0
    probe_count: int = 32
    shuffled_lineage_rotation: int = 1
    mismatched_rotation: int = 1


FIXED = Protocol()
# Existing known-positive unit-test tolerance, promoted to the intended gate
# after the first audit by independent review; no model/data settings changed.
REPRESENTABILITY_TOLERANCE = 1e-12
FAMILY = "public-four-template-nonlinear-16"
EVIDENCE_MODES = ("coarse", "paired", "paired_probes")
COST_KEYS = (
    "oracle_conditioning_queries", "identity_probe_queries", "teacher_fits",
    "teacher_fit_examples", "meta_regression_fits", "meta_training_rows",
    "full_ridge_fits", "full_ridge_fit_examples", "sgd_steps", "sgd_example_steps",
    "generator_forwards", "retrieval_distance_evaluations", "selection_queries",
    "final_audit_queries", "affine_ridge_fits", "affine_ridge_fit_examples",
)


def costs(**entries: int) -> dict[str, int]:
    if set(entries) - set(COST_KEYS) or any(v < 0 for v in entries.values()):
        raise ValueError("invalid cost entry")
    return {key: entries.get(key, 0) for key in COST_KEYS}


def add_costs(*ledgers: Mapping[str, int]) -> dict[str, int]:
    return {key: sum(ledger.get(key, 0) for ledger in ledgers) for key in COST_KEYS}


def public_templates() -> tuple[Adapter, ...]:
    """Four mutually orthogonal parameter directions, each squared norm 16."""
    zero = (0.0,) * CHANNELS
    diagonal = tuple(tuple(float(i == j) for j in range(CHANNELS)) for i in range(CHANNELS))
    shift = tuple(tuple(float(j == (i + 1) % CHANNELS) for j in range(CHANNELS))
                  for i in range(CHANNELS))
    alternate = tuple(tuple((1.0 if i % 2 == 0 else -1.0) * (j == (i + 2) % CHANNELS)
                            for j in range(CHANNELS)) for i in range(CHANNELS))
    return (Adapter(diagonal, zero), Adapter(shift, zero), Adapter(alternate, zero),
            Adapter((zero,) * CHANNELS, tuple(1.0 if i % 2 == 0 else -1.0
                                            for i in range(CHANNELS))))


def materialize(coefficients: Sequence[float]) -> Adapter:
    if len(coefficients) != 4:
        raise ValueError("expected four public-template coefficients")
    coefficients = tuple(finite(c) for c in coefficients)
    templates = public_templates()
    return Adapter(tuple(tuple(sum(c * a.weights[i][j] for c, a in zip(coefficients, templates))
                               for j in range(CHANNELS)) for i in range(CHANNELS)),
                   tuple(sum(c * a.bias[i] for c, a in zip(coefficients, templates))
                         for i in range(CHANNELS)))


def make_episode(seed: int, *, healthy: bool = False) -> Episode:
    """Seed is fixture construction only; it never enters generator evidence."""
    rng = random.Random(seed)
    coefficients = tuple(rng.uniform(-FIXED.coefficient_bound, FIXED.coefficient_bound)
                         for _ in range(4))
    planted = Adapter.zero() if healthy else materialize(coefficients)
    lineage = f"procedural-lineage-{seed}" + ("-healthy" if healthy else "")
    def examples(split: str, count: int) -> tuple[Example, ...]:
        result = []
        for index in range(count):
            h = tuple(rng.uniform(-FIXED.input_bound, FIXED.input_bound) for _ in range(CHANNELS))
            result.append(Example(f"{lineage}:{split}:{index}", h, planted.apply(h)))
        return tuple(result)
    return Episode(lineage, FAMILY, examples("conditioning", FIXED.conditioning_count),
                   examples("selection", FIXED.selection_count), examples("test", FIXED.test_count))


def guard_lineage_splits(splits: Mapping[str, Sequence[Episode]]) -> None:
    """Shared family is intentional; lineage AND sample IDs must be disjoint."""
    lineages: set[str] = set()
    samples: set[str] = set()
    for episodes in splits.values():
        for episode in episodes:
            if episode.host_lineage in lineages:
                raise ValueError("host lineage overlap")
            lineages.add(episode.host_lineage)
            ids = {x.sample_id for part in (episode.conditioning, episode.selection, episode.test)
                   for x in part}
            if samples & ids:
                raise ValueError("sample ID overlap")
            samples.update(ids)


def _basis(h: Sequence[float]) -> tuple[tuple[float, ...], ...]:
    x = tuple(math.tanh(v) for v in h)
    signs = tuple(1.0 if i % 2 == 0 else -1.0 for i in range(CHANNELS))
    return (x, tuple(x[(i + 1) % CHANNELS] for i in range(CHANNELS)),
            tuple(signs[i] * x[(i + 2) % CHANNELS] for i in range(CHANNELS)), signs)


def conditioning_evidence(conditioning: Sequence[Example], mode: str) -> tuple[float, ...]:
    """Coarse loses input/residual pairing; paired cross-moments recover gradients.

    Residuals are used exactly as supplied, including outside-family residuals.
    The probe bank executes identity host outputs; it adds no evidence features.
    """
    if mode not in EVIDENCE_MODES or not conditioning:
        raise ValueError("known mode and nonempty conditioning required")
    sums = [0.0] * 10
    for example in conditioning:
        residual = tuple(t - h for t, h in zip(example.target, example.h))
        basis = _basis(example.h)
        sums[0] += sum(example.h)
        sums[1] += sum(v * v for v in example.h)
        sums[2] += sum(basis[0])
        sums[3] += sum(v * v for v in basis[0])
        sums[4] += sum(residual)
        sums[5] += sum(v * v for v in residual)
        for j in range(4):
            sums[6 + j] += sum(q * r for q, r in zip(basis[j], residual))
    if mode == "paired_probes":
        host = FrozenHost()
        for channel in range(CHANNELS):
            for sign in (-1.0, 1.0):
                probe = tuple(sign if i == channel else 0.0 for i in range(CHANNELS))
                if host.output(probe) != probe:
                    raise ValueError("identity probe fixture changed")
    width = 6 if mode == "coarse" else 10
    return tuple(finite(value / (len(conditioning) * CHANNELS)) for value in sums[:width])


def _teacher_coefficients(adapter: Adapter) -> tuple[float, ...]:
    # Projection of OFFLINE teacher weights for regression supervision only.
    return tuple(finite((sum(w * q for row, template in zip(adapter.weights, a.weights)
                             for w, q in zip(row, template)) +
                         sum(b * q for b, q in zip(adapter.bias, a.bias))) / CHANNELS)
                 for a in public_templates())


def _regress(rows: Sequence[Sequence[float]], targets: Sequence[Sequence[float]],
             ridge: float = FIXED.meta_ridge) -> tuple[tuple[float, ...], ...]:
    width = len(rows[0]) + 1
    outputs = len(targets[0])
    augmented = [[ridge if i == j else 0.0 for j in range(width)] + [0.0] * outputs
                 for i in range(width)]
    for row, target in zip(rows, targets):
        x = (1.0,) + tuple(row)
        for i in range(width):
            for j in range(width):
                augmented[i][j] += x[i] * x[j]
            for j in range(outputs):
                augmented[i][width + j] += x[i] * target[j]
    for column in range(width):
        pivot = max(range(column, width), key=lambda i: abs(augmented[i][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = finite(augmented[column][column])
        if divisor == 0:
            raise ValueError("singular meta regression")
        augmented[column] = [finite(v / divisor) for v in augmented[column]]
        for row in range(width):
            if row != column:
                factor = augmented[row][column]
                augmented[row] = [finite(a - factor * b) for a, b in
                                  zip(augmented[row], augmented[column])]
    return tuple(tuple(row[width:]) for row in augmented)


@dataclass(frozen=True)
class FrozenGenerator:
    mode: str
    weights: tuple[tuple[float, ...], ...]

    def form(self, conditioning: Sequence[Example]) -> Adapter:
        x = (1.0,) + conditioning_evidence(conditioning, self.mode)
        return materialize(tuple(finite(sum(value * row[j] for value, row in zip(x, self.weights)))
                                 for j in range(4)))


@dataclass(frozen=True)
class TeacherBank:
    evidence: tuple[tuple[float, ...], ...]
    adapters: tuple[Adapter, ...]


@dataclass(frozen=True)
class AffineControl:
    """Descriptive272-weight affine residual; distinct from primary Adapter."""
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def apply(self, h: Sequence[float]) -> tuple[float, ...]:
        return tuple(finite(value + sum(w * x for w, x in zip(row, h)) + b)
                     for value, row, b in zip(h, self.weights, self.bias))


def _fit_affine(conditioning: Sequence[Example]) -> AffineControl:
    solution = _regress(tuple(x.h for x in conditioning),
                        tuple(tuple(t - h for t, h in zip(x.target, x.h)) for x in conditioning),
                        FIXED.teacher_ridge)
    return AffineControl(tuple(tuple(solution[1 + j][i] for j in range(CHANNELS))
                               for i in range(CHANNELS)), tuple(solution[0]))


def train_generators(training: Sequence[Episode]) -> tuple[dict[str, FrozenGenerator], TeacherBank, dict[str, int]]:
    if not training:
        raise ValueError("nonempty training lineages required")
    teachers = tuple(fit(e.conditioning, FIXED.teacher_ridge) for e in training)
    targets = tuple(_teacher_coefficients(a) for a in teachers)
    rows = {mode: tuple(conditioning_evidence(e.conditioning, mode) for e in training)
            for mode in EVIDENCE_MODES}
    generators = {mode: FrozenGenerator(mode, _regress(rows[mode], targets)) for mode in EVIDENCE_MODES}
    n = sum(len(e.conditioning) for e in training)
    return generators, TeacherBank(rows["paired"], teachers), costs(
        oracle_conditioning_queries=4 * n, identity_probe_queries=FIXED.probe_count * len(training),
        teacher_fits=len(training), teacher_fit_examples=n,
        meta_regression_fits=3, meta_training_rows=3 * len(training))


def _sgd(conditioning: Sequence[Example], steps: int) -> Adapter:
    weights = [[0.0] * CHANNELS for _ in range(CHANNELS)]
    bias = [0.0] * CHANNELS
    scale = 2 * FIXED.sgd_learning_rate / (len(conditioning) * CHANNELS)
    for _ in range(steps):
        gw = [[0.0] * CHANNELS for _ in range(CHANNELS)]
        gb = [0.0] * CHANNELS
        for example in conditioning:
            x = tuple(math.tanh(v) for v in example.h)
            for i in range(CHANNELS):
                error = example.h[i] + sum(w * q for w, q in zip(weights[i], x)) + bias[i] - example.target[i]
                gb[i] += error
                for j in range(CHANNELS):
                    gw[i][j] += error * x[j]
        for i in range(CHANNELS):
            bias[i] -= scale * gb[i]
            for j in range(CHANNELS):
                weights[i][j] -= scale * gw[i][j]
    return Adapter(tuple(tuple(row) for row in weights), tuple(bias))


def _retrieve(evidence: Sequence[float], bank: TeacherBank, neighbors: int) -> Adapter:
    indices = sorted(range(len(bank.evidence)), key=lambda i:
                     (sum((a - b) ** 2 for a, b in zip(evidence[6:], bank.evidence[i][6:])), i))[:neighbors]
    # Fixed equal-weight mean; distances use paired gradient moments only.
    return Adapter(tuple(tuple(sum(bank.adapters[k].weights[i][j] for k in indices) / len(indices)
                               for j in range(CHANNELS)) for i in range(CHANNELS)),
                   tuple(sum(bank.adapters[k].bias[i] for k in indices) / len(indices)
                         for i in range(CHANNELS)))


@dataclass(frozen=True)
class Candidate:
    name: str
    adapter: Adapter | AffineControl
    ledger: tuple[tuple[str, int], ...]


def candidate_bank(conditioning: Sequence[Example], generators: Mapping[str, FrozenGenerator],
                   teachers: TeacherBank, *,
                   shuffled_conditioning: Sequence[Example]) -> tuple[Candidate, ...]:
    """All candidates are complete before any selection or test examples enter."""
    n = len(conditioning)
    result = []
    def append(name: str, adapter: Adapter | AffineControl, **ledger: int) -> None:
        result.append(Candidate(name, adapter, tuple(costs(**ledger).items())))
    append("no_op", Adapter.zero())
    for mode in EVIDENCE_MODES:
        append(mode, generators[mode].form(conditioning), oracle_conditioning_queries=n,
               identity_probe_queries=FIXED.probe_count if mode == "paired_probes" else 0,
               generator_forwards=1)
    append("full_ridge", fit(conditioning, FIXED.teacher_ridge), oracle_conditioning_queries=n,
           full_ridge_fits=1, full_ridge_fit_examples=n)
    append("affine_ridge", _fit_affine(conditioning), oracle_conditioning_queries=n,
           affine_ridge_fits=1, affine_ridge_fit_examples=n)
    for steps in (1, 2, 4):
        append(f"sgd_{steps}", _sgd(conditioning, steps), oracle_conditioning_queries=n * steps,
               sgd_steps=steps, sgd_example_steps=n * steps)
    for name, count in (("retrieval", 1), ("interpolation_3", 3)):
        evidence = conditioning_evidence(conditioning, "paired")
        append(name, _retrieve(evidence, teachers, count), oracle_conditioning_queries=n,
               retrieval_distance_evaluations=len(teachers.adapters))
    append("shuffled_conditioning", generators["paired"].form(shuffled_conditioning),
           oracle_conditioning_queries=len(shuffled_conditioning), generator_forwards=1)
    mismatched = mismatched_residuals(conditioning)
    append("mismatched_pairing", generators["paired"].form(mismatched),
           oracle_conditioning_queries=n, generator_forwards=1)
    return tuple(result)


def mismatched_residuals(conditioning: Sequence[Example]) -> tuple[Example, ...]:
    """Preserve identity and residual marginals; break input/residual pairing."""
    if not conditioning:
        raise ValueError("nonempty conditioning required")
    result = []
    for i, x in enumerate(conditioning):
        donor = conditioning[(i + FIXED.mismatched_rotation) % len(conditioning)]
        residual = tuple(t - h for t, h in zip(donor.target, donor.h))
        result.append(Example(x.sample_id, x.h, tuple(h + r for h, r in zip(x.h, residual))))
    return tuple(result)


@dataclass(frozen=True)
class Admission:
    adapter: Adapter | AffineControl
    acted: bool
    raw_selection_mse: float
    no_op_selection_mse: float
    selection_queries: int


def admit(candidate: Adapter | AffineControl, selection: Sequence[Example], *, baseline_loss: float | None = None) -> Admission:
    raw = mse(candidate, selection)
    baseline = mse(Adapter.zero(), selection) if baseline_loss is None else finite(baseline_loss)
    acted = raw < baseline - FIXED.admission_improvement
    return Admission(candidate if acted else Adapter.zero(), acted, raw, baseline,
                     len(selection) * (2 if baseline_loss is None else 1))


def _evaluate(episodes: Sequence[Episode], generators: Mapping[str, FrozenGenerator],
              teachers: TeacherBank) -> dict:
    if len(episodes) < 2:
        raise ValueError("conditioning derangement requires at least two lineages")
    # Complete every bank in the phase before reading any selection/test pairs.
    banks = tuple(candidate_bank(e.conditioning, generators, teachers,
                                 shuffled_conditioning=episodes[(i + FIXED.shuffled_lineage_rotation) % len(episodes)].conditioning)
                  for i, e in enumerate(episodes))
    rows = []
    for index, (episode, candidates) in enumerate(zip(episodes, banks)):
        baseline_selection = mse(Adapter.zero(), episode.selection)
        # Admission is finished for every fixed candidate before final audits.
        decisions = tuple(admit(c.adapter, episode.selection, baseline_loss=baseline_selection)
                          for c in candidates)
        raw = tuple(mse(c.adapter, episode.test) for c in candidates)
        admitted = tuple(mse(d.adapter, episode.test) for d in decisions)
        no_op_test = raw[0]
        methods = {}
        for candidate, decision, raw_loss, admitted_loss in zip(candidates, decisions, raw, admitted):
            ledger = add_costs(dict(candidate.ledger), costs(selection_queries=decision.selection_queries,
                                                           final_audit_queries=2 * len(episode.test)))
            methods[candidate.name] = {
                "candidate_count": 0 if candidate.name == "no_op" else 1,
                "raw_selection_mse": decision.raw_selection_mse,
                "no_op_selection_mse": decision.no_op_selection_mse,
                "raw_test_mse": raw_loss, "admitted_test_mse": admitted_loss,
                "acted": decision.acted, "realized_harm": decision.acted and admitted_loss > no_op_test,
                "harm_mse_delta": admitted_loss - no_op_test, "costs": ledger,
            }
        rows.append({"host_lineage": episode.host_lineage, "family": episode.bottleneck_family,
                     "shuffled_conditioning_donor": episodes[(index + FIXED.shuffled_lineage_rotation) % len(episodes)].host_lineage,
                     "methods": methods, "costs": add_costs(
                         *(m["costs"] for m in methods.values()),
                         costs(selection_queries=len(episode.selection)))})
    summary = {}
    for method in rows[0]["methods"]:
        values = [row["methods"][method] for row in rows]
        summary[method] = {
            "raw_mean_mse": finite(sum(v["raw_test_mse"] for v in values) / len(values)),
            "admitted_mean_mse": finite(sum(v["admitted_test_mse"] for v in values) / len(values)),
            "acted_lineages": sum(v["acted"] for v in values),
            "realized_harms": sum(v["realized_harm"] for v in values),
        }
    return {"lineages": rows, "summary": summary, "costs": add_costs(*(row["costs"] for row in rows))}


def formation_acceptance(development: Mapping, test: Mapping, healthy: Mapping) -> dict[str, bool]:
    """Check declared targets, including the conventional positive control.

    The ridge gate was added during independent review after the initial audit
    to enforce already-intended representability, using the existing unit-test
    tolerance. It is an acceptance-report correction, not held-out tuning.
    """
    return {
        "paired_reduces_test_mean_mse": test["summary"]["paired"]["raw_mean_mse"] < test["summary"]["no_op"]["raw_mean_mse"],
        "full_ridge_representability": bool(test["lineages"]) and all(
            row["methods"]["full_ridge"]["raw_test_mse"] < REPRESENTABILITY_TOLERANCE
            for row in test["lineages"]),
        "healthy_all_no_op": all(v["acted_lineages"] == 0 for v in healthy["summary"].values()),
        "lineage_and_sample_isolation": True,  # run_formation's guard completed first.
        "finite_metrics": all(math.isfinite(v["raw_mean_mse"]) and math.isfinite(v["admitted_mean_mse"])
                              for phase in (development, test, healthy) for v in phase["summary"].values()),
    }


def run_formation() -> tuple[dict, FrozenGenerator]:
    training = tuple(make_episode(seed) for seed in FIXED.train_seeds)
    development = tuple(make_episode(seed) for seed in FIXED.dev_seeds)
    evaluation = tuple(make_episode(seed) for seed in FIXED.test_seeds)
    guard_lineage_splits({"train": training, "development": development, "test": evaluation})
    generators, teachers, offline = train_generators(training)
    # No settings are selected from development; evaluate it as a separate audit.
    dev = _evaluate(development, generators, teachers)
    test = _evaluate(evaluation, generators, teachers)
    healthy = _evaluate(tuple(make_episode(seed, healthy=True) for seed in FIXED.test_seeds), generators, teachers)
    config = asdict(FIXED)
    config.update({"lineages": {"train": 32, "development": 8, "test": 8},
                   "healthy_control_lineages": 8, "shared_family": FAMILY,
                   "adapter_stored_parameters": 272, "effective_output_dimension": 4,
                   "retrieval_retained_teacher_parameters": 32 * 272,
                   "public_template_materialized_values": 4 * 272,
                   "generator_parameters": {mode: sum(len(row) for row in g.weights)
                                            for mode, g in generators.items()},
                   "template_squared_parameter_norms": [16, 16, 16, 16],
                   "template_definitions": ["W[i,i]=1", "W[i,(i+1)%16]=1",
                                             "W[i,(i+2)%16]=(-1)^i", "b[i]=(-1)^i"],
                   "evidence_dimensions": {"coarse": 6, "paired": 10, "paired_probes": 10},
                   "coarse_features": ["mean_h", "mean_h_squared", "mean_tanh_h",
                                       "mean_tanh_h_squared", "mean_residual", "mean_residual_squared"],
                   "paired_features": "coarse plus four mean basis-response/residual cross-moments",
                   "probe_bank": "32 identity outputs at signed coordinate unit vectors; no added features",
                   "retrieval": "Euclidean distance on four paired moments; training teachers only",
                   "interpolation": "equal-weight mean of nearest three training teachers",
                   "affine_control": "full272 ridge using h rather than tanh(h); descriptive",
                   "shuffle": "within each phase lineage i receives conditioning from (i+1)%8; no held-out training",
                   "mispair": "rotate residuals by one conditioning sample, target_i=h_i+(target_j-h_j)",
                   "meta_intercept": "included and ridge-regularized like every coefficient",
                   "full_ridge_representability_tolerance": REPRESENTABILITY_TOLERANCE,
                   "representability_gate_provenance": "independent-review correction after first audit; existing1e-12 unit-test tolerance; no model/data tuning",
                   "hyperparameter_selection": "none; all constants predeclared",
                   "candidate_bank": "one candidate per method plus no-op, K=1 except no-op K=0"})
    all_costs = add_costs(offline, dev["costs"], test["costs"], healthy["costs"])
    report = {
        "assay": "seen-family synthetic lineage holdout", "config": config,
        "development": dev, "test": test, "healthy": healthy,
        "costs": {"offline": offline, "development": dev["costs"], "test": test["costs"],
                  "healthy": healthy["costs"], "total": all_costs,
                  "units": "example-output queries; repeated conditioning reads/SGD passes charged per method",
                  "selection_baseline": "one shared no-op selection pass per lineage, charged at lineage level",
                  "audit": "raw and admitted adapter outputs both executed for every method",
                  "meta_regression": "3 pivoted normal-equation solves: 7/11/11 feature columns,4 outputs",
                  "teachers": "32 conventional full272 ridge fits; retrieval retains32 teacher adapters"},
        "acceptance": formation_acceptance(dev, test, healthy),
        "summary": {"paired_raw_test_mean_mse": test["summary"]["paired"]["raw_mean_mse"],
                    "paired_admitted_test_mean_mse": test["summary"]["paired"]["admitted_mean_mse"],
                    "no_op_test_mean_mse": test["summary"]["no_op"]["raw_mean_mse"],
                    "full_ridge_test_mean_mse": test["summary"]["full_ridge"]["raw_mean_mse"],
                    "affine_ridge_test_mean_mse": test["summary"]["affine_ridge"]["raw_mean_mse"],
                    "paired_acted_lineages": test["summary"]["paired"]["acted_lineages"],
                    "paired_realized_harms": test["summary"]["paired"]["realized_harms"]},
        "limitations": [
            "Direct oracle residual observations are privileged evidence.",
            "Paired cross-moments contain task-gradient information.",
            "Identity-host probes add no identifying information.",
            "Shared public four-template family: no unseen-family or trained-host transfer claim.",
            "Admission is empirical, with no calibrated population harm-rate guarantee.",
            "All baseline comparisons are descriptive and retained, including unfavorable outcomes.",
            "Frozen form accepts outside-family residuals unchanged but emits only four-template adapters.",
            "Conditioning derangement uses another lineage's permitted conditioning in the same phase.",
            "Residual mismatching preserves identity and residual marginals while breaking their pairing.",
        ],
    }
    return report, generators["paired"]
