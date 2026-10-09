# Bounded technical concept demonstrator

This is a CPU-only, standard-library synthetic instrument for the proposed
one-site repair. TCD means **technical concept demonstrator**. This demonstrator
checks software and measurement boundaries; it does not validate the long-term
Sultai development or parameter-efficiency goal.

## Run without installing anything

From the Sultai repository root, with Python 3.10 or newer:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m sultai.smoke
# Or run both:
sh scripts/smoke.sh
```

The module prints deterministic JSON. The test suite and smoke each complete
in less than a second on the current host; no GPU, network, external library,
service, model download, or training campaign is used. These commands do not
install the optional package metadata in `pyproject.toml`.

## What is implemented

The frozen host is an identity readout at a known 16-channel site. Activations
are independently sampled from [-2, 2] by a local seeded pseudorandom generator.
The adapter computes `h + alpha * (W tanh(h) + b)`: W has 256 learned values,
b has 16, and alpha is a finite external control, fixed to 1 in the smoke.
The host has zero learned values. Adapter construction copies values into
immutable tuples and rejects incorrect shapes and non-finite values. Tests
cover zero weights, alpha=0, output overflow, deterministic execution, JSON
round trips, and a nonzero fixture's failure of an affine finite-difference
identity.

The planted oracle target uses fixed weights expressible by this adapter:
0.35 on the diagonal, 0.08 on the next channel cyclically, and alternating
biases of +/-0.025. Those weights exist only inside synthetic target generation.
The optimizer accepts paired examples, never the fixture's planted weights,
host internals, selection labels, or test labels. Targets are synthetic oracle
observations; the instrument does not establish their availability on a real
host. The healthy fixture uses target=h, and the no-repair reporting arm is
the zero adapter.

The method is **conventional adapter optimization on new-host paired examples**.
It solves a 17-feature ridge system for all 16 outputs using pivoted elimination.
The fixed tanh provides nonlinearity in h; fitting is linear in W and b. The
predeclared ridges are `(1e-8, 1e-3, 0.1)`, with no tuning after inspecting
results. A numerical feature-rank diagnostic uses a fixed absolute tolerance
of 1e-10. The positive and healthy fixtures have rank 17; a rank-deficient
test demonstrates finite fitting with positive ridge. Normal equations and
this simple rank estimate suit the small bounded fixture; they are not a
general-purpose solver or an estimate of the matrix condition number.

Each episode has 64 conditioning, 24 selection, and 32 test examples with
unique, disjoint sample IDs. Only conditioning pairs fit candidates; only
selection pairs choose the winner, with first-candidate tie breaking. The
test phase follows freezing the candidate bank and winner. It evaluates
three predeclared arms: no repair, the first fixed candidate, and best-of-K.
Test results never feed back into fitting or selection. When the first
candidate also wins best-of-K, its evaluation is still charged separately.

## Accounting and observed smoke result

Per episode, K=3 consumes three ridge solves and 192 conditioning-example
uses. Selection costs 72 candidate-example forward queries, or 1,152 scalar
predictions. The final three-arm test phase costs 96 forward queries, or
1,536 scalar predictions. These count evaluation requests on available
oracle pairs; no external host service is queried. A repair forward pass has
16 tanh evaluations and 256 weight products. Candidate formation, selection,
and inference are separately counted; these counts do not equate their
computational cost or provide research-grade runtime comparisons.

The candidate bank holds 816 adapter values before selection; a single retained
repair holds 272. There is no retained optimizer state. The analytical solve
uses temporary feature/Gram/right-hand-side arrays. This does not demonstrate
physical model removal or reduction of an existing host's parameters.

On the initial run, planted held-out MSE was approximately 0.0689634 for no
repair and 1.21236e-20 for both the first fixed candidate and best-of-K.
Healthy-control MSE was exactly zero for all three arms. These are expected
instrument results for representable oracle fixtures, not evidence of real
defect repairability. Small floating-point differences across Python/platform
versions are possible; repeat calls in one environment are deterministic.

## Boundary and diagnostic tests

The tests adversarially reverse selection/test rankings and require selection
to follow selection labels. Changing test targets cannot change fitted weights
or the winner. Changing selection labels cannot change the preselection bank.
Shuffled conditioning targets and mismatched pairs change the fitted repair
and worsen held-out fit in the fixed fixture. Frozen episode/host values remain
unchanged after fitting.

`guard_group_splits` rejects shared host lineage **or** shared bottleneck family
between any outer partitions, including a chain of shared identifiers across
three partitions. Its tests use disjoint pools and deliberate violations.
The smoke itself runs two synthetic episodes with within-episode splits; it
does not claim an outer train/validation/test benchmark over independent real
host families. Synthetic group names are labels, not empirical independence
evidence. A future benchmark must call the guard with its actual outer pools.

The separate `ShadowProbe` diagnostic demonstrates a passive ambiguity:
both downstream maps D_A(z)=z and D_B(z)=-z observe z=0, output=0, target=1.
The same fixed positive probe z=0.25 produces opposite output signs; repairs
z=+1 and z=-1 respectively hit the target. Healthy and zero-response cases
are also checked. This scalar demonstration is not a trained probing policy
and is not fed to the ridge benchmark. It does not implement the three
telemetry-condition comparison in DESIGN.md.

## Remaining work

Repair retrieval/interpolation, deterministic learned generators, conditional
diffusion, coarse/paired/probed telemetry comparisons, real-host integration,
common-future learning, handback schedules, physically removable structure,
and the half-final-parameter comparison remain future work. The ridge reference
is not forward-only generated repair formation. No unimplemented method is
reported as a measured baseline.
