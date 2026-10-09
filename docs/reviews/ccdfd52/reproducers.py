"""Independent boundary probes; --observe records evidence, default runs regression expectations."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / 'snapshot'
sys.path.insert(0, str(SNAPSHOT / 'src'))
from sultai import formation, hybrid, lifecycle
from sultai.repair import Adapter, mse

SAVED = json.loads((SNAPSHOT / 'docs/results/hybrid-cpu-2026-10-09.json').read_text())

class StubGenerator:
    def form(self, examples):
        raise AssertionError('must not form in boundary-report tests')


def runner_with(formation_report, lifecycle_report):
    with patch.object(hybrid, 'run_formation', return_value=(formation_report, StubGenerator())), \
         patch.object(hybrid, 'run_lifecycle', return_value=lifecycle_report):
        return hybrid.run()


class RequiredStateRegression(unittest.TestCase):
    def test_affine_empty_state_rejected_before_admission(self):
        with self.assertRaises(ValueError):
            formation.admit(formation.AffineControl((), ()), formation.make_episode(3000).selection)

    def test_frozen_generator_missing_rows_rejected(self):
        with self.assertRaises(ValueError):
            formation.FrozenGenerator('paired', ()).form(formation.make_episode(3000).conditioning)

    def test_frozen_generator_unused_nonfinite_rows_rejected(self):
        with self.assertRaises(ValueError):
            formation.FrozenGenerator('paired', ((0.0,) * 4,) * 11 + ((float('nan'),) * 4,)).form(
                formation.make_episode(3000).conditioning)

    def test_required_runner_gate_cannot_be_removed(self):
        f, l = copy.deepcopy(SAVED['formation']), copy.deepcopy(SAVED['lifecycle'])
        del l['acceptance']['formed_post_removal_target_met']
        with self.assertRaises(ValueError):
            runner_with(f, l)

    def test_empty_healthy_evidence_rejected(self):
        f = SAVED['formation']
        with self.assertRaises(ValueError):
            formation.formation_acceptance(f['development'], f['test'], {'summary': {}})

    def test_malformed_cost_count_rejected(self):
        for value in (True, 1.5, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                formation.costs(selection_queries=value)

    def test_missing_or_unknown_ledger_key_rejected(self):
        for ledger in ({}, {'selection_queres': 99}, {'selection_queries': -99}):
            with self.subTest(ledger=ledger), self.assertRaises(ValueError):
                formation.add_costs(ledger)

    def test_missing_summary_not_rendered_as_success(self):
        result = copy.deepcopy(SAVED)
        del result['formation']['summary']
        with patch.object(hybrid, 'run', return_value=result), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises((KeyError, ValueError)):
                hybrid.main(['--summary'])


def observe():
    evidence = {}
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    evidence['commit'] = manifest['commit']
    evidence['manifest_files_verified'] = len(manifest['files'])
    evidence['manifest_mismatches'] = [x['path'] for x in manifest['files']
        if hashlib.sha256((SNAPSHOT / x['path']).read_bytes()).hexdigest() != x['sha256']]
    e = formation.make_episode(3000)
    a = formation.AffineControl((), ())
    d = formation.admit(a, e.selection)
    evidence['empty_affine'] = {'output_channels': len(a.apply(e.selection[0].h)),
        'raw_selection_mse': d.raw_selection_mse, 'baseline_selection_mse': d.no_op_selection_mse,
        'acted': d.acted, 'raw_test_mse': mse(a, e.test)}
    evidence['empty_generator_is_zero_adapter'] = formation.FrozenGenerator('paired', ()).form(e.conditioning) == Adapter.zero()
    evidence['unused_nan_generator_row_is_zero_adapter'] = formation.FrozenGenerator('paired', ((0.0,) * 4,) * 11 + ((float('nan'),) * 4,)).form(e.conditioning) == Adapter.zero()
    f, l = copy.deepcopy(SAVED['formation']), copy.deepcopy(SAVED['lifecycle'])
    del l['acceptance']['formed_post_removal_target_met']
    result = runner_with(f, l)
    evidence['missing_lifecycle_gate'] = {'passed': result['passed'], 'gate_count': len(result['acceptance'])}
    evidence['empty_healthy_gates'] = formation.formation_acceptance(f['development'], f['test'], {'summary': {}})
    evidence['ledger_typo_becomes_zero'] = formation.add_costs({'selection_queres': 99})['selection_queries']
    evidence['negative_ledger_accepted'] = formation.add_costs({'selection_queries': -99})['selection_queries']
    evidence['cost_bool_accepted'] = formation.costs(selection_queries=True)['selection_queries']
    evidence['cost_fraction_accepted'] = formation.costs(selection_queries=1.5)['selection_queries']
    evidence['cost_nan_accepted'] = str(formation.costs(selection_queries=float('nan'))['selection_queries'])
    result = copy.deepcopy(SAVED)
    del result['formation']['summary']
    stream = io.StringIO()
    with patch.object(hybrid, 'run', return_value=result), contextlib.redirect_stdout(stream):
        code = hybrid.main(['--summary'])
    evidence['missing_summary'] = {'exit_code': code, 'summary': json.loads(stream.getvalue())['formation_summary']}
    live = hybrid.run()
    evidence['live_fixed_run'] = {'passed': live['passed'], 'gate_count': len(live['acceptance']),
        'source_identity_matches_saved': live['source_identity'] == SAVED['source_identity'],
        'json_normalized_report_matches_saved': json.loads(json.dumps(live, allow_nan=False)) == SAVED,
        'serialized_report_sha256': hashlib.sha256((json.dumps(live, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()).hexdigest(),
        'runtime': live['runtime'], 'saved_runtime': SAVED['runtime'],
        'formation_summary': live['formation']['summary'], 'lifecycle_summary': live['lifecycle']['summary']}
    existing = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(SNAPSHOT / 'tests'), '-v'],
        cwd=SNAPSHOT, capture_output=True, text=True, timeout=120)
    evidence['existing_tests'] = {'exit_code': existing.returncode, 'output': existing.stdout + existing.stderr}
    stream = io.StringIO()
    outcome = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RequiredStateRegression))
    evidence['new_regression_expectations'] = {'tests_run': outcome.testsRun, 'failure_count': len(outcome.failures),
        'error_count': len(outcome.errors), 'output': stream.getvalue()}
    evidence['manifest_mismatches_after'] = [x['path'] for x in manifest['files']
        if hashlib.sha256((SNAPSHOT / x['path']).read_bytes()).hexdigest() != x['sha256']]
    (ROOT / 'semantic-review-results.json').write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in evidence.items() if k not in ('existing_tests', 'new_regression_expectations')}, indent=2))
    print('Existing tests:', existing.returncode, '; new expectation failures:', len(outcome.failures))

if __name__ == '__main__':
    if '--observe' in sys.argv:
        observe()
    else:
        unittest.main()
