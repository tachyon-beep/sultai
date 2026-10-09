"""Run the fixed, CPU-only synthetic checks: python -m sultai.smoke."""

import json

from .repair import PARAMETERS, Adapter, Episode, feature_rank, fit, mse, select, synthetic_episode
from .report_types import SmokeEpisodeReport, SmokeReport

# Predeclared settings: never tune these using smoke selection or test outcomes.
RIDGES = (1e-8, 1e-3, 0.1)


def run_episode(episode: Episode) -> SmokeEpisodeReport:
    candidates = tuple(fit(episode.conditioning, ridge) for ridge in RIDGES)
    chosen = select(candidates, episode.selection)
    # The candidate bank and winner are frozen before this single test phase.
    # Evaluate all three predeclared reporting arms in that phase: no repair,
    # the first fixed candidate, and best-of-K. None feeds back into fitting.
    return {
        "examples": {
            "conditioning": len(episode.conditioning),
            "selection": len(episode.selection),
            "test": len(episode.test),
        },
        "conditioning_feature_rank": feature_rank(episode.conditioning),
        "feature_columns": 17,
        "learned_adapter_parameters": PARAMETERS,
        "retained_adapter_parameters": PARAMETERS,
        "candidate_bank_parameters_before_selection": len(candidates) * PARAMETERS,
        "host_learned_parameters": 0,
        "retained_optimizer_state_values": 0,
        "alpha_control": 1.0,
        "method": "conventional_paired_ridge_optimization",
        "ridge_candidates": list(RIDGES),
        "fit_example_uses": len(candidates) * len(episode.conditioning),
        "ridge_solves": len(candidates),
        "candidate_index": chosen.candidate_index,
        "selection_losses": list(chosen.losses),
        "selection_queries": chosen.selection_queries,
        "selection_scalar_predictions": chosen.selection_queries * 16,
        "test_queries": 3 * len(episode.test),
        "test_scalar_predictions": 3 * len(episode.test) * 16,
        "test_mse": {
            "no_repair": mse(Adapter.zero(), episode.test),
            "single_predeclared_candidate": mse(candidates[0], episode.test),
            "best_of_k": mse(chosen.adapter, episode.test),
        },
    }


def run() -> SmokeReport:
    return {
        "instrument": "synthetic_local_repair_smoke",
        "evidence_status": "software instrument checks; oracle targets; no goal validation",
        "positive_planted": run_episode(synthetic_episode(7)),
        "healthy_no_repair_control": run_episode(synthetic_episode(11, healthy=True)),
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
