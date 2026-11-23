import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from arm_design import queue_runner as q
from arm_design.analysis import ordered_map
from arm_design.config import ROOT, load_config
from arm_design.dashboard import create_app

DYN = ROOT/'robots/dynamics_sim_arm/urdf/dynamics_sim_arm.sim.yaml'
OPTIONS = dict(workers=2, seed=3, min_duration=.6, max_duration=1., force_n=100, moment_nm=10, trial_duration=.05, all_extremes=False, name='t')


def item(kind, payload=0., amount=16, repeats=1, on=True):
    return dict(kind=kind, payload_kg=payload, amount=amount, repeats=repeats, on=on)


def wait(condition, seconds=90):
    end = time.time()+seconds
    while time.time() < end:
        if condition():
            return
        time.sleep(.2)
    raise AssertionError('timed out')


def test_ordered_map_keeps_order_streams_past_a_slow_job_and_cancels():
    started, stop = [], [False]
    def job(k):
        started.append(k)
        time.sleep(.3 if k == 0 else .01)
        return k
    with ThreadPoolExecutor(2) as pool:
        results = ordered_map(pool, job, range(20), 2, lambda: stop[0])
        assert next(results) == 0
        assert len(started) > 4  # A batch barrier would have stopped at two.
        stop[0] = True
        rest = list(results)
    assert rest == sorted(rest) and len(started) < 20


@pytest.mark.parametrize('change', [dict(amount=0), dict(amount=1.5), dict(payload_kg=-1), dict(payload_kg=None), dict(repeats=0)])
def test_bad_rows_are_rejected_before_anything_is_written(change):
    with pytest.raises(ValueError):
        q.plan(DYN, [dict(item('sweep'), **change)], OPTIONS)
    with pytest.raises(ValueError, match='at least one'):
        q.plan(DYN, [item('sweep', on=False)], OPTIONS)


def test_plan_expands_seeds_and_passes_every_setting():
    p = q.plan(DYN, [item('sweep', 5, 100, 2), item('bounds', 5, 50), item('timestep', 0, 30), item('restrained', 5, 3000)], OPTIONS)
    sweep, second, bounds, check, hand = [s['args'] for s in p['steps']]
    assert sweep[sweep.index('--samples')+1] == '100' and '--skip-timestep-validation' in sweep and '--skip-timestep-validation' not in check
    assert [a[a.index('--seed')+1] for a in (sweep, second)] == ['3', '4']
    assert bounds[bounds.index('--max-contact-moment-nm')+1] == '10.0'
    assert hand[hand.index('--poses')+1] == '2' and hand[hand.index('--max-trials')+1] == '3000'
    o = p['options']
    few, many = (q.step_seconds(dict(p['steps'][0], amount=n), o, p['robot'], {})[0] for n in (100, 1000))
    assert many > 5*few
    key = q.rate_key(p['steps'][0], 2, p['robot'])
    assert q.step_seconds(p['steps'][0], o, p['robot'], dict(learned={key: 2.})) == (205., True)


def test_queue_runs_in_order_survives_a_failure_and_collects_fea_inputs(tmp_path, monkeypatch):
    prepared = q.plan(DYN, [item('bounds', 5, 20), item('sweep', 1, 16, 2), item('timestep', 0, 8)], dict(OPTIONS, max_duration=2.))
    # A broken test must not stop the rest.
    prepared['steps'].insert(1, dict(prepared['steps'][0], label='broken', args=['bounds', '--samples', '-1', '--max-contact-force-n', '1', '--workers', '1']))
    for s in prepared['steps']:
        if s['kind'] == 'bounds':
            s['args'] = [str(5) if a == str(q.REFINE) else a for a in s['args']]
    source = DYN.read_bytes()
    (tmp_path/q.RATES).write_text(json.dumps(dict(step_s=5e-4)))
    folder = q.save_queue(tmp_path, prepared)
    q.start(folder).wait(300)
    state = q.queue_state(folder)
    assert [s['status'] for s in state['steps']] == ['finished', 'failed', 'finished', 'finished', 'finished'], (folder/'queue.log').read_text()
    assert state['status'] == 'finished' and not state['error'] and DYN.read_bytes() == source
    assert state['steps'][1]['note'] and 'Overall peaks' in state['steps'][4]['note']
    run = next((folder/state['steps'][2]['folder']).glob('campaign-*/run.json'))
    assert json.loads(run.read_text())['settings']['analysis_payload']['mass_kg'] == 1
    s = json.loads((folder/'summary.json').read_text())
    assert not s['problems'] and any('Aggressive' in k for k in s['spreads'])
    # Every headline peak equals the largest value in the tests' own exports.
    row = next(r for r in s['rows'] if r['connection'] == 'lever <- l6')
    peaks = [float(p['maximum']) for step in state['steps'] if step['status'] == 'finished' for _, e in q.exports(folder/step['folder'])
             for p in csv.DictReader((e/'peaks.csv').open()) if p['interface'] == 'lever <- l6' and p['metric'] == 'force_N']
    assert len(peaks) == 6 and row['force_N_maximum'] == max(peaks)
    cases = list(csv.DictReader((folder/'fea-inputs/l6/governing-cases.csv').open()))
    assert cases and any('max force_N' in c['selected_for'] for c in cases)
    assert all(float(c['force_residual_N']) < 1e-6 for c in cases)
    assert len(list((folder/'fea-inputs/l6').glob('*.csv'))) == 7
    # Load Lab sees the runs and renders this queue.
    from arm_design.dashboard import discover
    assert run.parent in discover(tmp_path)
    from arm_design.dashboard_queue import summary_view, status_table
    assert len(summary_view(folder)) >= 4
    left, per = q.remaining(state, {})
    assert left == 0 and status_table(state, per, left)


def test_stop_now_keeps_results_and_resume_restarts_the_test(tmp_path):
    (tmp_path/q.RATES).write_text(json.dumps(dict(step_s=5e-4)))
    folder = q.save_queue(tmp_path, q.plan(DYN, [item('sweep', 0, 4000), item('bounds', 0, 10)], OPTIONS))
    proc = q.start(folder)
    wait(lambda: (q.queue_state(folder)['steps'][0].get('done') or 0) >= 3)
    assert q.queue_state(folder)['status'] == 'running'
    q.request_stop(folder)
    proc.wait(60)
    state = q.queue_state(folder)
    assert state['status'] == 'stopped' and [s['status'] for s in state['steps']] == ['stopped', 'pending']
    kept = next((folder/state['steps'][0]['folder']).glob('campaign-*/run.json'))
    assert json.loads(kept.read_text())['trials'] >= 3
    assert 'partial' in json.loads((folder/'summary.json').read_text())['tests'][0]
    # Pause before resuming into it: the first test runs again, the second never starts.
    data = json.loads((folder/'queue.json').read_text())
    data['steps'][0].update(amount=8, args=['8' if a == '4000' else a for a in data['steps'][0]['args']])
    (folder/'queue.json').write_text(json.dumps(data))
    proc = q.start(folder)
    wait(lambda: q.queue_state(folder)['steps'][0]['status'] == 'running')
    q.request_stop(folder, after_current=True)
    proc.wait(60)
    state = q.queue_state(folder)
    assert [s['status'] for s in state['steps']] == ['finished', 'pending'] and state['status'] == 'stopped'
    assert state['steps'][0]['folder'].endswith('retry1') and kept.exists()


def test_dead_runner_is_reported_and_resumable(tmp_path):
    folder = q.save_queue(tmp_path, q.plan(DYN, [item('bounds', 0, 10)], OPTIONS))
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    data = json.loads((folder/'queue.json').read_text())
    data.update(status='running', pid=dead.pid)
    data['steps'][0].update(status='running', started=time.time(), folder='x')
    (folder/'queue.json').write_text(json.dumps(data))
    assert q.queue_state(folder)['status'] == 'interrupted'


def test_tab_layout_estimate_and_mutual_exclusion(tmp_path):
    (tmp_path/q.RATES).write_text(json.dumps(dict(step_s=5e-4)))
    app = create_app(tmp_path)
    def callback(text):
        return next(v['callback'].__wrapped__ for k, v in app.callback_map.items() if text in k and 'callback' in v)
    ids = [dict(type='queue-field', row=i, field=f) for i in range(len(q.PRESET)) for f in ('on', 'payload_kg', 'amount', 'repeats')]
    values = [v for x in q.default_items() for v in (['on'] if x['on'] else [], x['payload_kg'], x['amount'], x['repeats'])]
    options = [str(DYN), '', 4, 0, .1, 1, 100, 10, .1, ['on']]
    times, total = callback('queue-total')(values, *options, 'queue', None, ids)
    assert times[0].startswith('≈') and times[-1] == '—' and 'finishes about' in total
    values[2] = None
    assert 'number' in callback('queue-total')(values, *options, 'queue', None, ids)[1]
    # A running queue keeps the Runner from starting a competing test.
    folder = q.save_queue(tmp_path, q.plan(DYN, [item('bounds', 0, 10)], OPTIONS))
    data = json.loads((folder/'queue.json').read_text())
    data.update(status='running', pid=os.getpid())
    (folder/'queue.json').write_text(json.dumps(data))
    message = callback('launch-status')(1, 'campaign', str(DYN), 5, 2, 0, 1, 'random', .5, 1, .5, .05, 1, '', [], '', 0, 0, 0, 0, .05, .05, .05, '', {})[0]
    assert 'queue is running' in message and not list(tmp_path.glob('job-*'))


def test_bounds_payload_flag_adds_the_runner_payload(tmp_path):
    from arm_design.payload import box_payload
    c = load_config(DYN)
    assert box_payload(c, 5)['com_offset_m'] == [0., .15, 0.]
    subprocess.run([sys.executable, '-m', 'arm_design.cli', 'bounds', '--config', str(DYN), '--payload', '5', '--max-contact-force-n', '100',
                    '--samples', '10', '--iterations', '0', '--workers', '1', '--out', str(tmp_path)], check=True, capture_output=True)
    import yaml
    saved = yaml.safe_load(next(tmp_path.glob('bounds-*/gravity/config.yaml')).read_text())
    assert saved['analysis_payload']['mass_kg'] == 5
