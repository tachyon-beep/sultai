"""Independent recomputation of the J1-J3 statistician's numbers from the attempt-4 receipts.

Run: python3 -I validator-check.py   (read-only; reads docs/results/gate3-pilot-2026-10-13-linked-3/)
Written independently of J1-J3-calc.py: Pearson rho computed directly, SDs from raw per-anchor
receipts (not execution-summary.json), chi-square interval on sigma_arm, true power sizing.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

ROOT = Path("/home/john/sultai/docs/results/gate3-pilot-2026-10-13-linked-3")
SEEDS = (3000, 3001, 3002, 3003, 3004, 3005)
Z95, Z80, INFL, X = 1.6448536, 0.8416212, 1.25, 0.02


def load(rel: str):
    with open(ROOT / rel) as fh:
        return json.load(fh)


def sd(xs):
    return st.stdev(xs)


def pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den


def hw(s, n):
    return Z95 * INFL * s / math.sqrt(n)


def n_star(s, x=X):
    return math.ceil((Z95 * INFL * s / (x / 2)) ** 2)


def n_power(s, delta, inflate=INFL):
    """One-sided alpha=.05, 80% power, paired one-sample z-test for mean shift delta."""
    return math.ceil(((Z95 + Z80) * inflate * s / delta) ** 2)


def main():
    print("=== 1. Per-anchor receipts (oracle{w}-{seed}/result.json .payload) ===")
    gains = {}
    for w in (2, 4):
        p5, p6, d = [], [], []
        for s in SEEDS:
            pl = load(f"oracle{w}-{s}/result.json")["payload"]
            p5.append(pl["P5"]["gain"])
            p6.append(pl["P6"]["gain"])
            d.append(pl["paired_gain_difference"])
            assert abs((pl["P6"]["gain"] - pl["P5"]["gain"]) - pl["paired_gain_difference"]) < 1e-12
        rho = pearson(p5, p6)
        gains[w] = (p5, p6, d)
        print(f"w{w}: SD(P5)={sd(p5):.5f} SD(P6)={sd(p6):.5f} SD(P6-P5)={sd(d):.5f} mean(P6-P5)={st.mean(d):.5f} "
              f"Pearson rho(P5,P6)={rho:.4f}  range P5 gain {min(p5):.3f}-{max(p5):.3f}")
    print(f"single-arm gain range across w2/w4 P5/P6: {min(min(g[0]+g[1]) for g in gains.values()):.3f}"
          f"-{max(max(g[0]+g[1]) for g in gains.values()):.3f}")

    print("\n=== 2. Intact host and posthoc P1 low-alpha arms (materiality-*/ and posthoc-*/result.json) ===")
    intact = [load(f"materiality-{s}/result.json")["payload"]["intact_loss"] for s in SEEDS]
    a1, b3, se_post = [], [], []
    for s in SEEDS:
        fam = [f for f in load(f"posthoc-{s}/result.json")["payload"]["families"]["P1"] if f["magnitude"] == 0.1]
        assert len(fam) == 1
        a1.append(fam[0]["arm_losses"]["A1"])
        b3.append(fam[0]["arm_losses"]["B3"])
        se_post.append(fam[0]["oracle"]["selection_paired_bootstrap_se"])
    a1_minus_b3 = [x - y for x, y in zip(a1, b3)]
    print(f"max|A1 - intact_loss| = {max(abs(x - y) for x, y in zip(a1, intact)):.2e}  (A1 is the intact host)")
    print(f"SD(intact)={sd(intact):.5f} span={max(intact)-min(intact):.4f}")
    print(f"SD(A1-B3)={sd(a1_minus_b3):.5f} range {min(a1_minus_b3):.4f}-{max(a1_minus_b3):.4f}")
    print(f"SD(B3 endpoint)={sd(b3):.5f} span={max(b3)-min(b3):.4f}; Pearson rho(A1,B3)={pearson(a1,b3):.3f}")
    w4end = [load(f"oracle4-{s}/result.json")["payload"]["P5"]["oracle_loss"] for s in SEEDS]
    w4end_chk = [load(f"oracle4-{s}/result.json")["payload"]["P5"]["noop_loss"]
                 - load(f"oracle4-{s}/result.json")["payload"]["P5"]["gain"] for s in SEEDS]
    assert max(abs(x - y) for x, y in zip(w4end, w4end_chk)) < 1e-9
    print(f"SD(w4 P5 endpoint = oracle_loss)={sd(w4end):.5f}")

    print("\n=== 3. Fit-seed SD and selection_paired_bootstrap_se (oracle*/ and posthoc-*/ .oracle) ===")
    seed_sds, sel_se = [], []
    for w in (2, 4):
        for s in SEEDS:
            pl = load(f"oracle{w}-{s}/result.json")["payload"]
            for fam in ("P5", "P6"):
                o = pl[fam]["oracle"]
                br = o["best"]["rate"]
                vals = [c["selection_loss"] for c in o["candidates"] if c["rate"] == br and c["steps"] == 800]
                assert len(vals) == 3
                seed_sds.append(sd(vals))
                sel_se.append(o["selection_paired_bootstrap_se"])
    for s in SEEDS:
        fam = [f for f in load(f"posthoc-{s}/result.json")["payload"]["families"]["P1"] if f["magnitude"] == 0.1][0]
        o = fam["oracle"]
        br = o["best"]["rate"]
        vals = [c["selection_loss"] for c in o["candidates"] if c["rate"] == br and c["steps"] == 800]
        seed_sds.append(sd(vals))
        sel_se.append(o["selection_paired_bootstrap_se"])
    print(f"fit-seed SD (3 seeds, selected rate, 800 steps; {len(seed_sds)} oracles): median={st.median(seed_sds):.4f} "
          f"min={min(seed_sds):.4f} max={max(seed_sds):.4f}; sqrt2*median={math.sqrt(2)*st.median(seed_sds):.4f}")
    print(f"selection_paired_bootstrap_se ({len(sel_se)} oracles): min={min(sel_se):.4f} median={st.median(sel_se):.4f} "
          f"max={max(sel_se):.4f}; x sqrt(3/5)={st.median(sel_se)*math.sqrt(0.6):.4f}")
    print(f"  bottleneck oracles only: min={min(sel_se[:24]):.4f} max={max(sel_se[:24]):.4f}")
    print(f"  posthoc P1 oracles (the report's 'B3 contrasts'): {min(se_post):.4f}-{max(se_post):.4f}")

    print("\n=== 4. Sizing table (sigma_arm = SD(A1-B3)) ===")
    sig = sd(a1_minus_b3)
    cases = [(0.0, sig * math.sqrt(2)), (0.5, sig), (0.9, sig * math.sqrt(0.2)),
             (pearson(*gains[2][:2]), sd(gains[2][2])), (pearson(*gains[4][:2]), sd(gains[4][2]))]
    print(f"{'rho':>6} {'SD(D)':>8} {'hw6':>7} {'hw32':>7} {'mde32':>7} {'n*.02':>6} {'n*.015':>7} {'n*.01':>6} {'npow(.02)':>9}")
    for rho, s in cases:
        print(f"{rho:6.3f} {s:8.5f} {hw(s,6):7.4f} {hw(s,32):7.4f} {(Z95+Z80)*INFL*s/math.sqrt(32):7.4f} "
              f"{n_star(s,0.02):6d} {n_star(s,0.015):7d} {n_star(s,0.01):6d} {n_power(s,0.02):9d}")
    print(f"half-width thresholds: n*<=32 iff SD<={0.01*math.sqrt(32)/(Z95*INFL):.5f}; n*<=44 iff SD<={0.01*math.sqrt(44)/(Z95*INFL):.5f}")
    print(f"12 GPU-h / 965 s = {12*3600/965:.2f} -> n<=44; 16 h -> {16*3600/965:.2f}; code cap also count<=200 (pilot.py:353)")
    # chi-square 95% interval on sigma_arm with 5 df, propagated to the rho=0.9 n*
    lo_chi, hi_chi = 12.8325, 0.8312  # chi2_{0.975,5}, chi2_{0.025,5}
    sig_lo, sig_hi = sig * math.sqrt(5 / lo_chi), sig * math.sqrt(5 / hi_chi)
    print(f"sigma_arm={sig:.5f}; 95% chi-square interval (5 df) {sig_lo:.4f}-{sig_hi:.4f}; "
          f"rho=0.9 SD(D) {sig_lo*math.sqrt(0.2):.4f}-{sig_hi*math.sqrt(0.2):.4f} -> n*(.02) "
          f"{n_star(sig_lo*math.sqrt(0.2))}-{n_star(sig_hi*math.sqrt(0.2))}")
    print(f"rho needed for n*<=32 at sigma_arm={sig:.4f}: rho>={1-(0.01*math.sqrt(32)/(Z95*INFL))**2/(2*sig**2):.4f}")

    print("\n=== 5. Sign test n=32 (exact binomial) ===")
    n = 32
    k = min(j for j in range(n + 1) if sum(math.comb(n, i) for i in range(j, n + 1)) / 2**n <= 0.05)
    p = 0.5
    while sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)) < 0.8:
        p += 0.0005
    print(f"threshold >= {k}/32 (size {sum(math.comb(n,i) for i in range(k,n+1))/2**n:.4f}); 80% power at P(D>X)={p:.3f}, "
          f"shift {st.NormalDist().inv_cdf(p):.3f} SD")

    print("\n=== 6. D3 and D5 ===")
    for name, g in (("w2 P5", gains[2][0]), ("w4 P5", gains[4][0]), ("A1-B3", a1_minus_b3)):
        print(f"D3 proxy {name}: SD={sd(g):.4f} hw32={hw(sd(g),32):.4f} n*={n_star(sd(g))}")
    comp = math.sqrt(2) * sd(b3)
    print(f"D3 compressed-as-B3: SD={comp:.4f} hw32={hw(comp,32):.4f} n*={n_star(comp)}")
    for name, s in (("D5 host spread kept sqrt2*SD(intact)", math.sqrt(2) * sd(intact)),
                    ("D5 compressed sqrt2*SD(B3)", comp)):
        print(f"{name}: SD={s:.4f} hw32={hw(s,32):.4f} n*={n_star(s)}  [sqrt2 assumes independent endpoints]")

    print("\n=== 7. Accuracy (not in receipts; binomial, assumed p=0.75) ===")
    for m in (3000, 5000):
        print(f"n={m}: single-arm SE={100*math.sqrt(0.1875/m):.3f} pts; paired at 5%/10% discordant "
              f"{100*math.sqrt(0.05/m):.3f}/{100*math.sqrt(0.10/m):.3f} pts")
    ratios = [((0.02/se_n) / (eff_p/se_p))**2 for eff_p in (0.5, 0.7) for se_n in (0.002, 0.004) for se_p in (0.45, 0.32)]
    print(f"variance ratios over the 2x2x2 grid: min={min(ratios):.1f} max={max(ratios):.1f}")
    print(f"guard hw32 at 0.4-pt anchor SD: {hw(0.4,32):.3f} pts (report says 0.18)")

    print("\n=== 8. Receipts that carry none of these numbers ===")
    print("02-magnitudes.json keys:", sorted(load("02-magnitudes.json").keys()))
    print("03-posthoc-known-answer.json keys:", sorted(load("03-posthoc-known-answer.json").keys()))
    print("bottle2 host parameters:", load("bottle2-3000/result.json")["payload"]["parameters"],
          "; intact host parameters:", load("intact-3000/result.json")["payload"]["parameters"])


if __name__ == "__main__":
    main()
