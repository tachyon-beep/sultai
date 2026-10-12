"""CPU controls for amendment v4.2: the extended P3 ladder, the chained identity and linked pilot 3 custody."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from sultai_screen import pilot as pilot_module
from sultai_screen.contracts import (
    AMENDMENT_PATH,
    AMENDMENT_SHA,
    AMENDMENT_V41_PATH,
    AMENDMENT_V41_SHA,
    P3_LADDER,
    P3_RETIRED,
    P3_RETIRED_GAINS,
    PROTOCOL_VERSION,
    InstrumentHold,
    PilotHold,
    file_sha,
)
from sultai_screen.fit import losses
from sultai_screen.host import Features, Host, freeze
from sultai_screen.impair import make_impairment
from sultai_screen.pilot import Supervisor, magnitude_decision
from sultai_screen.screen import MATERIALITY_LADDERS
from test_protocol_identity import PUBLISHED

REPO = Path(__file__).resolve().parents[1]


def record(seed: int, p3_new: dict[str, float], retired: tuple[float, ...] | None = None) -> dict[str, object]:
    """A materiality payload: retired P3 rungs default to attempt 3's published gains for this seed."""
    values = P3_RETIRED_GAINS[str(seed)] if retired is None else retired
    p3 = {str(k): v for k, v in zip(P3_RETIRED, values, strict=True)} | p3_new
    return {
        "seed": seed,
        "rungs": {"P1": {"0.1": 0.3, "0.3": 0.2, "0.03": 0.5}, "P2": {"0.1": 2.0, "0.03": 2.0}, "P3": p3},
    }


class P3LadderTests(unittest.TestCase):
    def test_v42_ladder_and_chained_identity_are_pinned(self) -> None:
        self.assertEqual(P3_RETIRED, (0.25, 0.5, 1.0))
        self.assertEqual(P3_LADDER, (*P3_RETIRED, 2.0, 4.0))
        self.assertEqual(MATERIALITY_LADDERS["P3"], P3_LADDER)
        self.assertEqual(PROTOCOL_VERSION, "gate3-v4.2")
        self.assertEqual(file_sha(REPO / AMENDMENT_V41_PATH), AMENDMENT_V41_SHA)
        self.assertIn(AMENDMENT_V41_SHA, (REPO / AMENDMENT_PATH).read_text())
        self.assertEqual(file_sha(REPO / AMENDMENT_PATH), AMENDMENT_SHA)

    def test_retired_gains_are_attempt_three_published_values(self) -> None:
        for seed in range(3000, 3006):
            published = json.loads(
                (REPO / f"docs/results/gate3-pilot-2026-10-13-linked-2/materiality-{seed}/result.json").read_text()
            )["payload"]["rungs"]["P3"]
            self.assertEqual(P3_RETIRED_GAINS[str(seed)], tuple(published[str(k)] for k in P3_RETIRED))
            self.assertLess(P3_RETIRED_GAINS[str(seed)][-1], 0.05 if seed == 3002 else 1.0)
        self.assertTrue(any(P3_RETIRED_GAINS[str(seed)][-1] < 0.05 for seed in range(3000, 3006)))

    def test_first_new_rung_clearing_every_anchor_is_chosen_and_exhaustion_holds(self) -> None:
        clears_at_two = tuple(
            record(seed, {"2.0": 0.12 + 0.01 * i, "4.0": 0.5}) for i, seed in enumerate(range(3000, 3006))
        )
        chosen = magnitude_decision(clears_at_two)
        self.assertEqual(chosen["kappa"], 2.0)
        self.assertEqual((chosen["alpha_low"], chosen["alpha_high"], chosen["s_min"]), (0.1, 0.3, 0.1))
        clears_at_four = (*clears_at_two[:5], record(3005, {"2.0": 0.049, "4.0": 0.5}))
        self.assertEqual(magnitude_decision(clears_at_four)["kappa"], 4.0)
        exhausted = (*clears_at_two[:5], record(3005, {"2.0": 0.049, "4.0": 0.0499}))
        with self.assertRaisesRegex(InstrumentHold, "P3 exhausted"):
            magnitude_decision(exhausted)

    def test_retired_rungs_must_replay_attempt_three_exactly_and_are_never_chosen(self) -> None:
        good = tuple(record(seed, {"2.0": 0.3, "4.0": 0.6}) for seed in range(3000, 3006))
        # A retired rung that now clears the bar on every anchor would be a choice under the old
        # rule; under v4.2 any difference from attempt 3, clearing or not, is a replay hold.
        drifted_clear = (*good[:2], record(3002, {"2.0": 0.3, "4.0": 0.6}, retired=(0.06, 0.07, 0.08)), *good[3:])
        with self.assertRaisesRegex(PilotHold, "retired P3 rung"):
            magnitude_decision(drifted_clear)
        last = P3_RETIRED_GAINS["3004"]
        drifted_tiny = (
            *good[:4],
            record(3004, {"2.0": 0.3, "4.0": 0.6}, retired=(last[0], last[1], last[2] + 1e-9)),
            good[5],
        )
        with self.assertRaisesRegex(PilotHold, "retired P3 rung"):
            magnitude_decision(drifted_tiny)
        unknown_seed = (*good[:5], record(3099, {"2.0": 0.3, "4.0": 0.6}, retired=(0.0, 0.0, 0.0)))
        with self.assertRaises(KeyError):
            magnitude_decision(unknown_seed)
        duplicate_seed = (*good[:5], good[0])
        with self.assertRaisesRegex(PilotHold, "development anchors"):
            magnitude_decision(duplicate_seed)

    def test_new_rungs_keep_the_exact_inverse_within_endpoint_tolerance(self) -> None:
        host = Host(4)
        freeze(host)
        z = torch.rand((8, 16, 8, 8), generator=torch.Generator().manual_seed(3002))
        labels = torch.arange(8) % 10
        intact = losses(host, Features(z, z, labels), None)
        for kappa in (2.0, 4.0):
            operator = make_impairment("P3", kappa, 3002, z)
            shifted = operator.apply(z)
            self.assertGreater(float((shifted - z).abs().max()), 0.0)
            restored = Features(z, operator.undo(shifted), labels)
            self.assertLess(float((losses(host, restored, None) - intact).abs().max()), 1e-4)


PUBLISHED_3 = {
    **PUBLISHED,
    ".weft/gate3-pilot-linked-pilot-2.json": "docs/results/gate3-pilot-2026-10-13-linked-2/attempt-custody.json",
    "runs/gate3-pilot-2026-10-13-linked-2/pilot-result.json": (
        "docs/results/gate3-pilot-2026-10-13-linked-2/pilot-result.json"
    ),
    "runs/gate3-pilot-2026-10-13-linked-2/ledger-020.json": "docs/results/gate3-pilot-2026-10-13-linked-2/ledger-020.json",
}


class LinkedPilotThreeTests(unittest.TestCase):
    def test_linked_pilot_three_binds_four_custodies_and_the_cumulative_charge(self) -> None:
        module = pilot_module
        self.assertEqual({relative for relative, _ in module.PRIOR_RECEIPTS_3}, set(PUBLISHED_3))
        self.assertEqual(module.CUMULATIVE_CHARGE_3, 2934.3435500559863)
        approved = {
            "linked_pilot_3_approved": True,
            "prior_charge_gpu_process_seconds": module.CUMULATIVE_CHARGE_3,
            "evidence": [{"path": module.DISPOSITION_3}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for relative, expected in module.PRIOR_RECEIPTS_3:
                target = base / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((REPO / PUBLISHED_3[relative]).read_bytes())
                self.assertEqual(file_sha(target), expected)
            originals = {relative: (base / relative).read_bytes() for relative, _ in module.PRIOR_RECEIPTS_3}
            disposition = base / module.DISPOSITION_3
            with patch("sultai_screen.pilot.REPO", base):
                with self.assertRaisesRegex(PilotHold, "disposition document"):
                    module.linked_pilot_three(approved)
                disposition.parent.mkdir(parents=True)
                disposition.write_text("John's disposition fixture\n")
                linkage = module.linked_pilot_three(approved)
                self.assertEqual(linkage["opening_charge_gpu_process_seconds"], 2934.3435500559863)
                self.assertEqual(linkage["attempt_number"], 4)
                self.assertIs(linkage["further_attempt_authorized"], False)
                self.assertEqual(len(linkage["prior_receipts"]), 9)
                receipt = base / module.PILOT_3_RECEIPT
                module.claim_attempt(receipt, base / "attempt-four", linkage)
                with self.assertRaisesRegex(PilotHold, "already exists"):
                    module.claim_attempt(receipt, base / "attempt-five", linkage)
                for key, bad in (
                    ("linked_pilot_3_approved", False),
                    ("prior_charge_gpu_process_seconds", module.CUMULATIVE_CHARGE),
                    ("evidence", [{"path": module.DISPOSITION_2}]),
                ):
                    with self.assertRaises(PilotHold):
                        module.linked_pilot_three(dict(approved, **{key: bad}))
                for relative, _ in module.PRIOR_RECEIPTS_3:
                    path = base / relative
                    path.write_bytes(originals[relative] + b" ")
                    with self.assertRaisesRegex(PilotHold, "missing or changed"):
                        module.linked_pilot_three(approved)
                    path.unlink()
                    with self.assertRaisesRegex(PilotHold, "missing or changed"):
                        module.linked_pilot_three(approved)
                    path.write_bytes(originals[relative])
                supervisor = Supervisor(base, base, module.CUMULATIVE_CHARGE_3)
                self.assertEqual(supervisor.charged, module.CUMULATIVE_CHARGE_3)

    def test_pilot_entry_refuses_third_linked_attempt_without_disposition_or_with_two_linkages(self) -> None:
        from sultai_screen.pilot import pilot

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            certificate = base / "certificate.json"
            certificate.write_text('{"linked_pilot_3_approved": false, "evidence": []}\n')
            with (
                patch("sultai_screen.pilot.check_certificate", return_value={"commit": "fixture"}),
                patch("sultai_screen.pilot.environment_identity", return_value={}),
                patch("sultai_screen.pilot.claim_attempt") as claim,
                patch("sultai_screen.pilot.check_handover") as handover,
                patch("sultai_screen.pilot.Supervisor") as supervisor,
            ):
                with self.assertRaisesRegex(PilotHold, "disposition"):
                    pilot(base / "new-output", base, certificate, third=True)
                for flags in ({"retry": True, "third": True}, {"second": True, "third": True}):
                    with self.assertRaisesRegex(PilotHold, "one linkage"):
                        pilot(base / "new-output", base, certificate, **flags)
                claim.assert_not_called()
                handover.assert_not_called()
                supervisor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
