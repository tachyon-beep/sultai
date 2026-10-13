"""J1-J3 statistician arithmetic from the Gate 3 attempt-4 receipts (read-only).

Run: python3 -I J1-J3-calc.py
Every number printed cites the receipt field it came from.
"""
from __future__ import annotations

import json
import math
import statistics as st
from itertools import product
from pathlib import Path

ROOT = Path("/home/john/sultai/docs/results/gate3-pilot-2026-10-13-linked-3")
SEEDS = [3000, 3001, 3002, 3003, 3004, 3005]
Z95, Z80, INFL = 1.645, 0.8416, 1.25          # contracts.py:185 uses 1.645 * 1.25
X_CODE = 0.02                                  # contracts.py:43
PER_ANCHOR_S, CEIL_12H, CEIL_CODE = 965.0, 12 * 3600, 16 * 3600  # draft §8, §10; pilot.py:353


def sd(xs):
    return st.stdev(xs)


def load(p):
    return json.load(open(ROOT / p))


summary = load("execution-summary.json")
ka = summary["bottleneck_known_answer"]

# ---- single-arm gains and the measured within-anchor paired difference ----
rows = {}
for w in ("w2", "w4"):
    p5 = [ka[w][str(s)]["P5"]["gain"] for s in SEEDS]
    p6 = [ka[w][str(s)]["P6"]["gain"] for s in SEEDS]
    d = [ka[w][str(s)]["P6_minus_P5"] for s in SEEDS]
    rho = (sd(p5) ** 2 + sd(p6) ** 2 - sd(d) ** 2) / (2 * sd(p5) * sd(p6))
    rows[w] = dict(p5=p5, p6=p6, d=d, sd5=sd(p5), sd6=sd(p6), sdd=sd(d), rho=rho)

# P1 attenuated hosts: module endpoint below the intact loss (posthoc arms A1 = intact, B3 = fitted)
p1 = summary["posthoc_P1_low_alpha"]
b3_over_intact = [p1[str(s)]["arms"]["A1"] - p1[str(s)]["arms"]["B3"] for s in SEEDS]
b3_end = [p1[str(s)]["arms"]["B3"] for s in SEEDS]
intact = [load(f"materiality-{s}/result.json")["payload"]["intact_loss"] for s in SEEDS]
w4p5_end = [ka["w4"][str(s)]["P5"]["noop_loss"] - ka["w4"][str(s)]["P5"]["gain"] for s in SEEDS]

# ---- fit-seed floor and the selection bootstrap SE (what it actually is) ----
seed_sds, sel_se = [], []
for w, s, fam in product((2, 4), SEEDS, ("P5", "P6")):
    o = load(f"oracle{w}-{s}/result.json")["payload"][fam]["oracle"]
    b = o["best"]
    vals = [c["selection_loss"] for c in o["candidates"] if c["rate"] == b["rate"] and c["steps"] == 800]
    seed_sds.append(sd(vals))                       # three fit seeds at the selected rate, 800 steps
    sel_se.append(o["selection_paired_bootstrap_se"])  # SE of (best-200 - best-800) on 3,000 selection ex.
for s in SEEDS:
    o = load(f"posthoc-{s}/result.json")["payload"]["families"]["P1"][0]["oracle"]
    b = o["best"]
    vals = [c["selection_loss"] for c in o["candidates"] if c["rate"] == b["rate"] and c["steps"] == 800]
    seed_sds.append(sd(vals))
    sel_se.append(o["selection_paired_bootstrap_se"])


def hw(s, n):
    return Z95 * INFL * s / math.sqrt(n)


def mde(s, n):
    return (Z95 + Z80) * INFL * s / math.sqrt(n)


def n_star(s, x=X_CODE):
    return math.ceil((Z95 * INFL * s / (x / 2)) ** 2)


def sign_threshold(n=32, alpha=0.05):
    k = n
    while sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n <= alpha:
        k -= 1
    return k + 1


def sign_power_shift(n=32, power=0.8):
    k = sign_threshold(n)
    p = 0.5
    while sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1)) < power:
        p += 0.001
    # shift in SD units such that P(D > X) = p under a normal
    return k, p, st.NormalDist().inv_cdf(p)


def main():
    print("## Single-arm gains and the measured paired difference (execution-summary.json .bottleneck_known_answer)")
    for w, r in rows.items():
        print(f"{w}: SD(P5 gain)={r['sd5']:.4f}  SD(P6 gain)={r['sd6']:.4f}  SD(P6-P5)={r['sdd']:.4f}  "
              f"mean(P6-P5)={st.mean(r['d']):.4f}  implied rho={r['rho']:.3f}")
    print(f"P1 intact-B3 gain (posthoc_P1_low_alpha.*.arms A1-B3): SD={sd(b3_over_intact):.4f} "
          f"range {min(b3_over_intact):.3f}-{max(b3_over_intact):.3f}")
    print(f"intact dev loss (materiality-*/result.json .payload.intact_loss): SD={sd(intact):.4f} span={max(intact)-min(intact):.3f}")
    print(f"B3 endpoint SD={sd(b3_end):.4f} span={max(b3_end)-min(b3_end):.3f}; w4 P5 endpoint SD={sd(w4p5_end):.4f}")
    print(f"fit-seed SD at selected rate (oracle*/posthoc* candidates, 3 seeds, 800 steps): "
          f"median={st.median(seed_sds):.4f} max={max(seed_sds):.4f}; two one-seed arms floor sqrt2*median={math.sqrt(2)*st.median(seed_sds):.4f}")
    print(f"selection_paired_bootstrap_se (best200-best800, 3,000 selection ex.): median={st.median(sel_se):.4f} "
          f"max={max(sel_se):.4f}; scaled to 5,000 dev x sqrt(3/5): {st.median(sel_se)*math.sqrt(0.6):.4f}")

    sigma_arm = sd(b3_over_intact)  # best proxy for a 272-parameter module's gain SD on the unimpaired host
    print(f"\n## Detectable margins, sigma_arm={sigma_arm:.4f} (intact-B3), SD(D)=sigma*sqrt(2(1-rho))")
    k, p, z = sign_power_shift()
    print(f"sign test n=32: need >= {k}/32 above X (one-sided 5%); 80% power needs P(D>X)={p:.3f} -> shift {z:.2f} SD(D)")
    print(f"{'rho':>8} {'SD(D)':>7} {'hw6':>7} {'mde6':>7} {'hw32':>7} {'mde32':>7} {'sign32':>7} {'n*(.02)':>8} {'n*(.015)':>8} {'n*(.01)':>8}")
    cases = [(0.0, sigma_arm * math.sqrt(2)), (0.5, sigma_arm), (0.9, sigma_arm * math.sqrt(0.2)),
             (rows['w2']['rho'], rows['w2']['sdd']), (rows['w4']['rho'], rows['w4']['sdd'])]
    for rho, s in cases:
        print(f"{rho:8.3f} {s:7.4f} {hw(s,6):7.4f} {mde(s,6):7.4f} {hw(s,32):7.4f} {mde(s,32):7.4f} {z*s:7.4f} "
              f"{n_star(s,0.02):8d} {n_star(s,0.015):8d} {n_star(s,0.01):8d}")
    n_12h = int(CEIL_12H // PER_ANCHOR_S)
    print(f"\n12 GPU-h at {PER_ANCHOR_S:.0f} s/anchor -> n<= {n_12h}; code ceiling {CEIL_CODE} s -> n<= {int(CEIL_CODE//PER_ANCHOR_S)}")
    for x in (0.02, 0.015, 0.01):
        s32 = (x / 2) * math.sqrt(32) / (Z95 * INFL)
        s44 = (x / 2) * math.sqrt(n_12h) / (Z95 * INFL)
        print(f"X={x}: n*<=32 iff SD(D)<={s32:.4f}; n*<={n_12h} iff SD(D)<={s44:.4f}")

    print("\n## D3 (a gain, not a two-arm difference): single-arm gain SDs")
    for name, g in (("w2 P5", rows['w2']['p5']), ("w4 P5", rows['w4']['p5']), ("intact-B3", b3_over_intact)):
        print(f"{name}: SD={sd(g):.4f}  hw32={hw(sd(g),32):.4f}  n*(.02)={n_star(sd(g))}")
    comp = sd(b3_end)
    print(f"if finishing compresses the host like B3 (endpoint SD {comp:.4f}), gain SD ~ sqrt(2)*{comp:.4f}={math.sqrt(2)*comp:.4f}: "
          f"hw32={hw(math.sqrt(2)*comp,32):.4f} n*(.02)={n_star(math.sqrt(2)*comp)}")

    print("\n## D5 (paired by data stream): two regimes")
    s_host = math.sqrt(2) * sd(intact)
    s_comp = math.sqrt(2) * sd(b3_end)
    for name, s in (("host seed spread kept", s_host), ("endpoints compressed as B3", s_comp)):
        print(f"{name}: SD(D5)={s:.4f} hw32={hw(s,32):.4f} mde32={mde(s,32):.4f} n*(.02)={n_star(s)}")

    print("\n## Accuracy (NOT in receipts; binomial with assumed p=0.75)")
    for n in (3000, 5000):
        print(f"n={n}: single-arm SE={100*math.sqrt(0.75*0.25/n):.2f} pts; paired-difference SE at 5% discordant={100*math.sqrt(0.05/n):.2f} pts, at 10%={100*math.sqrt(0.10/n):.2f} pts")
    print("loss per-example paired SE on 5,000 dev ~0.002-0.004 nats (scaled selection SE; B3 contrasts 0.0034-0.0038 on 3,000);")
    print("outside prior 25-35 acc pts/nat -> 0.02 nats ~ 0.5-0.7 pts, 0.01 nats ~ 0.25-0.35 pts")
    for eff_n, eff_p in ((0.02, 0.5), (0.02, 0.7)):
        for se_n, se_p in ((0.002, 0.45), (0.004, 0.32)):
            snr_n, snr_p = eff_n / se_n, eff_p / se_p
            print(f"effect {eff_n} nats = {eff_p} pts: loss SNR {snr_n:.1f}, acc SNR {snr_p:.1f}, variance ratio {(snr_n/snr_p)**2:.0f}x")


if __name__ == "__main__":
    main()
