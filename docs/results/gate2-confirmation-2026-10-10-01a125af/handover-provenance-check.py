import hashlib, json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads((p / 'development.json').read_text())
c = json.loads((p / 'confirmation.json').read_text())
assert hashlib.sha256((p / 'development.json').read_bytes()).hexdigest() == '0fab1abb37b5d6a15c4082f154978e6c8b86f62503e6ed71cc6232f5dc9de97e'
assert (p / 'confirmation.json').read_bytes() == (p / 'replay.json').read_bytes()
assert d['phase'] == 'development' and c['phase'] == 'confirmation'
assert c['identity']['source_commit'] == '64439d80bdd7897aa94f8aa819ab23c17fcaffba'
assert c['identity']['plan_sha256'] == d['identity']['plan_sha256']
assert c['source_identity'] == d['source_identity']
assert c['software_valid'] and not c['training_ready']
dp = {x['name']: x for x in d['formation']['provenance']}
cp = {x['name']: x for x in c['formation']['provenance']}
assert dp['train'] == cp['train']  # shared frozen offline training is deliberate
held = ('evaluation', 'clean_healthy', 'noisy_healthy')
dev_ids = {x for n in held for x in dp[n]['lineages']}
confirm_ids = {x for n in held for x in cp[n]['lineages']}
assert not dev_ids.intersection(confirm_ids)
for name in held:
    assert len(cp[name]['lineages']) == 8
    assert cp[name]['sample_count'] == 1216
    assert cp[name]['sample_ids_sha256'] != dp[name]['sample_ids_sha256']
    assert cp[name]['pairs_sha256'] != dp[name]['pairs_sha256']
assert [s['root'] for s in c['trajectory']['streams']] == [9000, 9010, 9020, 9030]
assert not {s['root'] for s in c['trajectory']['streams']}.intersection(s['root'] for s in d['trajectory']['streams'])
print('Replay and report provenance checks passed; scientific verdicts follow:')
print(json.dumps({'gates': c['gates'], 'specificity': c['diagnostic_specificity'], 'contrasts': c['trajectory']['contrast_summary']}, indent=2))
