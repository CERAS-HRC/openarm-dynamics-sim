"""Unattended test queue: one detached process runs checked tests in order through the normal CLI, then collects FEA inputs.

A queue folder holds queue.json (atomic state), request.json and one folder per test attempt. Results are the
tests' ordinary run folders, so Load Lab's Viewer, fea-cases and replay work on them unchanged.
"""
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np
import yaml

from .analysis import atomic_json
from .config import load_config, validate
from .fea import METRICS
from .payload import box_payload

KINDS = {
    'sweep': dict(label='Aggressive sweeps', unit='trials', amount=10000),
    'bounds': dict(label='Static + contact (bounds)', unit='poses', amount=2000),
    'timestep': dict(label='Finer-timestep check', unit='trials', amount=1000),
    'restrained': dict(label='Restrained hand', unit='trials', amount=2000),
}
# The "Full FEA set": (kind, payload kg, checked).
PRESET = [('sweep', 0., True), ('sweep', 5., True), ('sweep', 2., True), ('bounds', 0., True), ('bounds', 5., True),
          ('timestep', 0., True), ('timestep', 5., True), ('restrained', 5., False)]
REFINE = 300            # bounds iterations per target
RECORDS = 70            # typical record-setting trials rerun by the timestep check
BYTES_PER_TRIAL = 95e3
RATES = 'queue-rates.json'
DONE = ('finished', 'failed', 'stopped')


def default_items():
    return [dict(kind=k, payload_kg=p, amount=KINDS[k]['amount'], repeats=1, on=on) for k, p, on in PRESET]


def number(value, label, low, high, whole=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be a number.') from None
    if not np.isfinite(value) or not low <= value <= high or whole and value != int(value):
        raise ValueError(f'{label} must be {"a whole number " if whole else ""}between {low:g} and {high:g}.')
    return int(value) if whole else value


def plan(model, items, options):
    """Validated steps for the checked items. Nothing is written or started."""
    base = load_config(model)
    o = dict(workers=number(options['workers'], 'Workers', 1, 64, True), seed=number(options['seed'], 'Seed', 0, 2**31, True),
             min_duration=number(options['min_duration'], 'Shortest sweep', .1, 120), max_duration=number(options['max_duration'], 'Longest sweep', .1, 120),
             force_n=number(options['force_n'], 'Contact force', .001, 1e6), moment_nm=number(options['moment_nm'], 'Contact moment', .001, 1e6),
             trial_duration=number(options['trial_duration'], 'Restrained trial time', .05, 120),
             all_extremes=bool(options.get('all_extremes', True)), name=str(options.get('name') or '').strip()[:60])
    if o['min_duration'] > o['max_duration']:
        raise ValueError('Shortest sweep must not exceed the longest sweep.')
    steps = []
    for item in items:
        if not item['on']:
            continue
        kind, label = item['kind'], KINDS[item['kind']]['label']
        amount = number(item['amount'], f'{label}: amount', 1, 10**7, True)
        payload = number(item['payload_kg'], f'{label}: payload', 0, 100)
        cfg = deepcopy(base)
        cfg['analysis_payload'] = box_payload(cfg, payload)
        validate(cfg)
        for k in range(number(item['repeats'], f'{label}: seeds', 1, 50, True)):
            seed = o['seed']+k
            args = ['--workers', str(o['workers']), '--seed', str(seed)]
            if kind in ('sweep', 'timestep'):
                args = ['campaign', '--aggressive', '--samples', str(amount), '--min-duration', str(o['min_duration']),
                        '--max-duration', str(o['max_duration']), *args, *(['--skip-timestep-validation'] if kind == 'sweep' else [])]
            elif kind == 'bounds':
                args = ['bounds', '--samples', str(amount), '--iterations', str(REFINE), '--max-contact-force-n', str(o['force_n']),
                        '--max-contact-moment-nm', str(o['moment_nm']), *args]
            else:
                args = ['restrained', '--poses', str(max(1, int(np.ceil(amount/3**len(cfg['joints']))))), '--max-trials', str(amount),
                        '--trial-duration', str(o['trial_duration']), '--torque-ramp', str(min(.05, o['trial_duration']/2)), *args]
            steps.append(dict(kind=kind, label=f'{label} · {payload:g} kg · seed {seed}', payload_kg=payload, amount=amount, seed=seed,
                              args=args, status='pending', attempt=0, folder=None, started=None, elapsed_s=None, note=''))
    if not steps:
        raise ValueError('Check at least one test.')
    return dict(model=str(model), robot=base['name'], options=o, steps=steps)


def capacity(workers):
    """Busy performance-core equivalents; efficiency cores measured about 0.4 of one, scheduling keeps about 95% busy."""
    def core(level):
        return int(subprocess.check_output(['sysctl', '-n', f'hw.perflevel{level}.physicalcpu'], stderr=subprocess.DEVNULL))
    try:
        fast, slow = core(0), core(1)
    except (OSError, ValueError, subprocess.CalledProcessError):
        fast, slow = os.cpu_count() or 1, 0
    return .95*(min(workers, fast)+.4*max(0, min(workers-fast, slow)))


def benchmark(model):
    """CPU seconds per physics step on this machine (best of three short aggressive sweeps)."""
    from .campaign import run_sweep
    from .collision import CollisionChecker
    from .model import build_arm
    cfg = load_config(model)
    cfg['sweep'].update(settle_s=.1, hold_s=.25, load_acceptance='motor_loads_v2')
    cfg['drive']['trajectory_feedforward'] = True
    cfg['simulation']['step_s'] = min(cfg['simulation']['step_s'], .0005)
    arm = build_arm(cfg)
    checker = CollisionChecker(arm)
    q = [float(np.clip(0, *j['limits_deg'])) for j in cfg['joints']]
    best = np.inf
    for _ in range(3):
        started = time.process_time()
        run_sweep(arm, q, q, .5, checker=checker, strict=True)
        best = min(best, (time.process_time()-started)/(.95/cfg['simulation']['step_s']))
    return best


def rate_key(step, workers, robot):
    return f"{robot}|{step['kind']}|{step['payload_kg']:g}|{workers}"


def step_seconds(step, options, robot, rates):
    """(seconds, measured): learned wall time per unit when this exact test ran here before, else the benchmark."""
    learned = rates.get('learned', {}).get(rate_key(step, options['workers'], robot))
    if learned:
        return 5+learned*step['amount'], True
    cpu, n = rates.get('step_s', 5.5e-4), capacity(options['workers'])
    trial = 2000*cpu
    if step['kind'] == 'sweep':
        work = step['amount']*trial
    elif step['kind'] == 'timestep':
        work = (step['amount']+RECORDS*12)*trial
    elif step['kind'] == 'bounds':
        work = 7*cpu*(step['amount']+126*REFINE/7)
    else:
        work = step['amount']*options['trial_duration']/.001*cpu*1.2
    return 8+work/n, False


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load_rates(root):
    return read(Path(root)/RATES, {})


def save_queue(root, prepared):
    folder = Path(root)/('queue-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:4])
    folder.mkdir(parents=True)
    atomic_json(folder/'request.json', prepared)
    atomic_json(folder/'queue.json', dict(prepared, status='starting', created=time.time()))
    return folder


def start(folder):
    """Detached, so closing the browser or the dashboard does not stop it."""
    folder = Path(folder)
    for flag in ('CANCEL', 'PAUSE'):
        (folder/flag).unlink(missing_ok=True)
    state = read(folder/'queue.json')
    for step in state['steps']:
        if step['status'] in ('running', 'stopped'):
            step.update(status='pending', note='Restarted from the beginning; the earlier partial results are kept.')
    state.update(status='starting', started=time.time(), error='')
    atomic_json(folder/'queue.json', state)
    with (folder/'queue.log').open('a') as log:
        return subprocess.Popen([sys.executable, '-m', 'arm_design.queue_runner', str(folder)], stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def request_stop(folder, after_current=False):
    folder = Path(folder)
    (folder/('PAUSE' if after_current else 'CANCEL')).touch()
    state = read(folder/'queue.json', {})
    if not after_current:
        for step in state.get('steps', []):
            if step['status'] == 'running' and step['folder']:
                # Existing runs stop between trials; runs that have not opened yet check the parent.
                (folder/step['folder']/'CANCEL').touch()


def queue_state(folder):
    """State with live progress; a dead runner is reported, never shown as running."""
    folder = Path(folder)
    state = read(folder/'queue.json')
    if not state:
        return None
    if state['status'] == 'running' and state.get('pid'):
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            state.update(status='interrupted', error='The queue process ended (restart, power loss or crash). Press Resume.')
        except PermissionError:
            pass
    if state['status'] == 'starting' and time.time()-state.get('started', state['created']) > 60:
        state.update(status='interrupted', error='The queue did not start; see queue.log.')
    for step in state['steps']:
        if step['status'] == 'running':
            step['elapsed_s'] = time.time()-step['started']
            run = next(iter(sorted((folder/step['folder']).glob('*/run.json'))), None)
            info = read(run, {}) if run else {}
            step['done'], step['counted'] = info.get('trials', 0), info.get('counted', 0)
            check = info.get('validation_progress')
            step['phase'] = info.get('phase', '') + (f" {check['checked']}/{check['total']} at {1e3*check['timestep_s']:g} ms" if check else '')
    state['stopping'] = 'now' if (folder/'CANCEL').exists() else 'after' if (folder/'PAUSE').exists() else ''
    return state


def remaining(state, rates):
    """(seconds left, per-step estimates); a running test blends its own measured pace."""
    total, per = 0., []
    for step in state['steps']:
        seconds, measured = step_seconds(step, state['options'], state['robot'], rates)
        if step['status'] in DONE:
            per.append(step['elapsed_s'])
            continue
        if step['status'] == 'running':
            done, spent = step.get('done', 0), step['elapsed_s']
            if step['kind'] in ('sweep', 'restrained') and done >= max(20, .01*step['amount']):
                seconds = spent*step['amount']/done
            seconds = max(seconds, spent)
            total += seconds-spent
        else:
            total += seconds
        per.append(seconds)
    return total, per


def exports(step_folder):
    """(case label, export folder) for every load-case export of one attempt."""
    found = []
    for run in sorted(Path(step_folder).glob('*/run.json')):
        found += [('', p.parent) for p in sorted(run.parent.glob('fea-*/peaks.csv'))]
    found += [(p.parent.name, p.parent) for p in sorted(Path(step_folder).glob('bounds-*/*/peaks.csv'))]
    return found


def timestep_note(step_folder):
    path = next(iter(Path(step_folder).glob('campaign-*/timestep-validation.json')), None)
    v = read(path, {}) if path else {}
    if len(v.get('levels', [])) != 2 or not v.get('completed'):
        return 'Finer-timestep check incomplete.'
    valid = [{r['trial']: r for r in level['trials'] if r['counted']} for level in v['levels']]
    both = sorted(set(valid[0]) & set(valid[1]))
    if not both:
        return 'No record-setting trial stayed valid at finer timesteps.'
    peak = [np.max([np.array(level[t]['maximum'])[:, 6:] for t in both], axis=0) for level in valid]
    change = np.abs(peak[1]-peak[0])/np.maximum(peak[1], 1e-6)
    change = change[peak[1] > 1e-3].max()  # Free pins carry no moment; ignore zero loads.
    lost = len(v['source_trials'])-len(both)
    return f'Overall peaks change at most {100*change:.1f}% between the two finer timesteps; {lost} of {len(v["source_trials"])} record trials were not valid at both.'


def summarize(folder):
    """Worst load per connection across finished tests, seed agreement, and FEA inputs. Values never mix instants."""
    from .fea_cases import part_cases, write_csv
    folder = Path(folder)
    state = read(folder/'queue.json')
    rows, sources, agreement = {}, [], {}
    for n, step in enumerate(state['steps']):
        if step['status'] != 'finished' and not (step['status'] == 'stopped' and step['folder']):
            continue
        for case, export in exports(folder/step['folder']):
            label = step['label'] + (f' · {case}' if case else '') + (' · partial' if step['status'] == 'stopped' else '')
            sources.append((label, export))
            with (export/'peaks.csv').open() as f:
                for p in csv.DictReader(f):
                    row = rows.setdefault((p['body'], p['interface']), {})
                    for side, better in (('minimum', np.less), ('maximum', np.greater)):
                        if side == 'minimum' and p['metric'] in METRICS[6:]:
                            continue
                        value, key = float(p[side]), (p['metric'], side)
                        if key not in row or better(value, row[key]['value']):
                            row[key] = dict(value=value, source=label, export=len(sources)-1, sample=int(p[side[:3]+'_sample']))
                        if p['metric'] in METRICS[6:] and step['status'] == 'finished':
                            group = agreement.setdefault((step['kind'], step['payload_kg'], case, p['interface'], p['metric']), {})
                            group[step['seed']] = value
    table = [dict(part=part, connection=name, **{f'{m}_{s}': row[(m, s)]['value'] for m, s in row},
                  force_test=row[('force_N', 'maximum')]['source'], moment_test=row[('moment_Nm', 'maximum')]['source'])
             for (part, name), row in rows.items()]
    # Largest seed-to-seed spread per test; connections that stay tiny are ignored.
    spreads = {}
    for (kind, payload, case, name, metric), values in agreement.items():
        v = np.array(list(values.values()))
        if len(v) > 1 and v.max() > (1 if metric == 'force_N' else .05):
            key = f"{KINDS[kind]['label']} · {payload:g} kg"+(f' · {case}' if case else '')
            spread = float((v.max()-v.min())/v.max())
            if spread > spreads.get(key, dict(spread=-1))['spread']:
                spreads[key] = dict(spread=spread, connection=name, metric=metric, seeds=len(v))
    inputs = folder/'fea-inputs'
    shutil.rmtree(inputs, ignore_errors=True)
    problems = []
    for part in dict.fromkeys(p for p, _ in rows):
        (inputs/part).mkdir(parents=True)
        governing = {}
        for (p, name), row in rows.items():
            if p == part:
                for (metric, side), x in row.items():
                    if abs(x['value']) < 1e-9 or not state['options']['all_extremes'] and metric not in METRICS[6:]:
                        continue
                    governing.setdefault((x['export'], x['sample']), []).append(f"{side[:3]} {metric} {x['value']:.4g} at {name}")
        chosen = []
        for (e, sample), reasons in sorted(governing.items()):
            label, export = sources[e]
            for case in part_cases(export, part, selected_samples=[sample]):
                chosen.append(dict(case, phase=f"{label}: {case['phase']}", sample=len(chosen), reasons=reasons))
                if max(case['residual_n'], case['moment_residual_nm']) > 1e-6:
                    problems.append(f'{part}: force balance residual in {label}')
        write_csv(inputs/part/'governing-cases.csv', chosen)
        for e, (label, export) in enumerate(sources):
            cases = part_cases(export, part, all_extremes=state['options']['all_extremes'])
            safe = ''.join(c if c.isalnum() else '-' for c in label.replace(' · ', '_'))
            write_csv(inputs/part/f'{e+1:02}-{safe}.csv', cases)
    with (folder/'summary.csv').open('w', newline='') as f:
        fields = ['part', 'connection', 'force_N_maximum', 'force_test', 'moment_Nm_maximum', 'moment_test',
                  *(f'{m}_{s}' for m in METRICS[:6] for s in ('minimum', 'maximum'))]
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(table)
    atomic_json(folder/'summary.json', dict(rows=table, spreads=spreads, problems=problems, tests=[s for s, _ in sources]))


def run(folder):
    folder = Path(folder).resolve()
    awake = None
    if sys.platform == 'darwin' and shutil.which('caffeinate'):
        awake = subprocess.Popen(['caffeinate', '-ims', '-w', str(os.getpid())])
    state = read(folder/'queue.json')
    state.update(status='running', pid=os.getpid(), error='')
    publish = lambda: atomic_json(folder/'queue.json', state)
    publish()
    rates = load_rates(folder.parent)
    if 'step_s' not in rates:
        rates['step_s'] = benchmark(state['model'])
        atomic_json(folder.parent/RATES, rates)
    try:
        for n, step in enumerate(state['steps']):
            if step['status'] in DONE:
                continue
            if (folder/'CANCEL').exists() or (folder/'PAUSE').exists():
                break
            step['attempt'] += 1
            name = f"{n+1:02}-{step['kind']}-{step['payload_kg']:g}kg-seed{step['seed']}" + (f"-retry{step['attempt']-1}" if step['attempt'] > 1 else '')
            out = folder/name
            out.mkdir(exist_ok=True)
            cfg = load_config(state['model'])
            cfg['analysis_payload'] = box_payload(cfg, step['payload_kg'])
            cfg['analysis_run_name'] = (state['options']['name']+' ' if state['options']['name'] else '')+step['label']
            (out/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
            step.update(status='running', folder=name, started=time.time(), note='')
            publish()
            if (folder/'CANCEL').exists():
                (out/'CANCEL').touch()
            with (out/'log.txt').open('w') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'arm_design.cli', *step['args'], '--config', str(out/'config.yaml'), '--out', str(out)],
                                        stdout=log, stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if (out/'CANCEL').exists() and step['kind'] == 'bounds':
                        proc.terminate()  # Seconds long, with no partial results to keep.
                    time.sleep(1)
            step['elapsed_s'] = time.time()-step['started']
            cancelled = (out/'CANCEL').exists()
            # The process-level result, not the presence of files, decides success.
            step['status'] = 'stopped' if cancelled else 'finished' if proc.returncode == 0 else 'failed'
            if step['status'] == 'failed':
                tail = (out/'log.txt').read_text().strip().splitlines()
                step['note'] = (tail[-1] if tail else f'Exit code {proc.returncode}')[:300]
            elif step['kind'] == 'timestep':
                step['note'] = timestep_note(out)
            if step['status'] == 'finished':
                info = read(next(iter(out.glob('*/run.json')), out/'none'), {})
                step['done'], step['counted'] = info.get('trials', step['amount']), info.get('counted')
                rates = load_rates(folder.parent)
                rates.setdefault('learned', {})[rate_key(step, state['options']['workers'], state['robot'])] = step['elapsed_s']/step['amount']
                if step['amount'] >= 1000:  # Short runs are mostly start-up time.
                    atomic_json(folder.parent/RATES, rates)
            publish()
            try:
                summarize(folder)
            except Exception as exc:
                state['error'] = f'Summary failed: {exc}'
                publish()
        left = [s for s in state['steps'] if s['status'] not in DONE]
        stopped = any(s['status'] == 'stopped' for s in state['steps'])
        state['status'] = 'stopped' if left or stopped else 'finished'
    except BaseException as exc:
        state.update(status='interrupted', error=str(exc))
        raise
    finally:
        state['ended'] = time.time()
        publish()
        if awake:
            awake.terminate()


if __name__ == '__main__':
    run(sys.argv[1])
