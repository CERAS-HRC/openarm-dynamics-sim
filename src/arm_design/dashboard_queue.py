"""Load Lab's Test queue tab: check tests, see the time, run them unattended."""
import os
from pathlib import Path
import shutil
import time
from threading import RLock

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from . import queue_runner as q

OPTIONS = ['model', 'name', 'workers', 'seed', 'min_duration', 'max_duration', 'force_n', 'moment_nm', 'trial_duration', 'all_extremes']
FIELDS = ['on', 'payload_kg', 'amount', 'repeats']


def duration(seconds):
    if seconds is None:
        return '—'
    if seconds < 90:
        return f'{seconds:.0f} s'
    minutes = int(round(seconds/60))
    return f'{minutes} min' if minutes < 90 else f'{minutes//60} h {minutes % 60:02} min'


def clock(seconds):
    t = time.localtime(time.time()+seconds)
    return time.strftime('%-I:%M %p' if os.name != 'nt' else '%I:%M %p', t) + ('' if t.tm_yday == time.localtime().tm_yday else ' tomorrow' if seconds < 172800 else time.strftime(' on %b %d', t))


def layout(models):
    cores = os.cpu_count() or 4
    field = lambda i, name, **kw: dcc.Input(id=dict(type='queue-field', row=i, field=name), type='number', persistence='queue-v1', **kw)
    rows = []
    for i, item in enumerate(q.default_items()):
        kind = q.KINDS[item['kind']]
        rows.append(html.Tr([
            html.Td(dcc.Checklist(id=dict(type='queue-field', row=i, field='on'), options=[dict(label=kind['label'], value='on')],
                                  value=['on'] if item['on'] else [], persistence='queue-v1')),
            html.Td(field(i, 'payload_kg', min=0, step=.1, value=item['payload_kg'])),
            html.Td(html.Div([field(i, 'amount', min=1, step=1, value=item['amount']), html.Span(kind['unit'])], className='queue-amount')),
            html.Td(field(i, 'repeats', min=1, step=1, value=item['repeats'])),
            html.Td(id=dict(type='queue-time', row=i), className='queue-time')]))
    opt = lambda name, label, value, **kw: html.Div([html.Label(label), dcc.Input(id='queue-'+name.replace('_', '-'), type='number', value=value, persistence='queue-v1', **kw)])
    return html.Div([
        html.Main([
            html.Div([html.H2('Test queue'), html.P('Check the tests, set how much of each, press Run. They run one after another and survive closing the browser.', className='note')]),
            html.Fieldset([
                html.Div([html.Div([html.Label('Robot'), dcc.Dropdown(id='queue-model', clearable=False, value=str(models[0]), persistence='queue-v1',
                            options=[dict(label=p.parent.parent.name if 'robots' in p.parts else 'Built-in arm', value=str(p)) for p in models])]),
                          html.Div([html.Label('Queue name (optional)'), dcc.Input(id='queue-name', placeholder='e.g. Full FEA set', persistence='queue-v1')]),
                          opt('workers', f'Worker processes (this machine: {cores} cores)', max(1, cores-2), min=1, max=64, step=1)], className='queue-options'),
                html.Div(html.Table([html.Thead(html.Tr([html.Th(x) for x in ('Test', 'Payload (kg)', 'Amount', 'Seeds (repeats)', 'Estimated time')])),
                            html.Tbody(rows)], className='queue-table'), className='link-summary'),
                html.Details([html.Summary('Advanced settings'),
                    html.Div([opt('min_duration', 'Shortest sweep (s)', .1, min=.1), opt('max_duration', 'Longest sweep (s)', 1, min=.1),
                              opt('force_n', 'Contact force limit (N)', 100, min=.001), opt('moment_nm', 'Contact moment limit (Nm)', 10, min=.001),
                              opt('trial_duration', 'Restrained trial time (s)', .1, min=.05), opt('seed', 'First seed', 0, min=0, step=1)], className='queue-options'),
                    dcc.Checklist(id='queue-all-extremes', options=[dict(label='FEA cases for every Fx…Mz extreme (off: peak force and moment only)', value='on')], value=['on'], persistence='queue-v1')]),
            ], id='queue-settings'),
            html.Div(id='queue-total', className='status'),
            html.Div([html.Button('Run selected tests', id='queue-run', className='primary'),
                      html.Button('Stop after current test', id='queue-pause', disabled=True), html.Button('Stop now', id='queue-stop', disabled=True),
                      html.Button('Resume', id='queue-resume', disabled=True)], className='runner-actions'),
            html.Div(id='queue-message', className='note'),
            html.P('Keeps a Mac awake while running; leave a laptop plugged in with its lid open. Stop keeps completed trials; a stopped test restarts from the beginning on Resume.', className='note'),
            html.Label('Queue'), dcc.Dropdown(id='queue-select', placeholder='No queues yet'),
            html.Div(id='queue-status'), html.Div(id='queue-summary'),
            dcc.Store(id='queue-version'), dcc.Store(id='queue-started'),
        ]),
    ], id='queue-panel', className='queue-workspace', style=dict(display='none'))


def queues(root):
    return sorted((p.parent for p in Path(root).glob('queue-*/queue.json')), reverse=True)


def status_table(state, per, left):
    icons = dict(pending='Waiting', running='Running', finished='Done', failed='Failed', stopped='Stopped')
    rows = []
    for step, seconds in zip(state['steps'], per):
        done, total = step.get('done') or (step['amount'] if step['status'] == 'finished' else 0), step['amount']
        progress = '—' if step['kind'] == 'bounds' else f'{done:,} / {total:,}'
        if step.get('counted') is not None:
            progress += f" · {step['counted']:,} accepted"
        if step['status'] == 'running' and step.get('phase'):
            progress += ' · '+step['phase']
        timing = duration(seconds) if step['status'] in q.DONE else f"{duration(step['elapsed_s'])} of ≈ {duration(seconds)}" if step['status'] == 'running' else f'≈ {duration(seconds)}'
        rows.append(html.Tr([html.Td(icons[step['status']], className='queue-'+step['status']), html.Td(step['label']), html.Td(progress),
                             html.Td(timing), html.Td(step.get('note', ''))]))
    return html.Div(html.Table([html.Thead(html.Tr([html.Th(x) for x in ('Status', 'Test', 'Trials', 'Time', 'Notes')])), html.Tbody(rows)],
                               className='queue-table'), className='link-summary')


def summary_view(folder):
    s = q.read(Path(folder)/'summary.json')
    if not s or not s['rows']:
        return ''
    cells = lambda r: [r['part'], r['connection'], f"{r['force_N_maximum']:.4g}", r['force_test'], *((f"{r['moment_Nm_maximum']:.4g}", r['moment_test']) if r['moment_Nm_maximum'] > 1e-9 else ('—', 'Free pin'))]
    out = [html.H2('Worst loads so far'),
           html.P(['Each value is one instant in the named test; never combine rows into one load case. Enter the cases from ',
                   html.Code(str(Path(folder)/'fea-inputs')), ': ', html.Code('<part>/governing-cases.csv'),
                   ' holds the cases that set any overall extreme, the numbered files hold every test.'], className='note'),
           html.Div(html.Table([html.Thead(html.Tr([html.Th(x) for x in ('Part', 'Connection', 'Peak force (N)', 'Set by', 'Peak moment (Nm)', 'Set by')])),
                                html.Tbody([html.Tr([html.Td(x) for x in cells(r)]) for r in s['rows']])], className='queue-table'), className='link-summary')]
    if s['spreads']:
        out += [html.H2('Do the seeds agree?'), html.Ul([html.Li(f"{k}: largest difference {100*v['spread']:.0f}% across {v['seeds']} seeds ({v['connection']}, {v['metric'].split('_')[0]})"
                + ('' if v['spread'] <= .1 else ' — run more trials')) for k, v in s['spreads'].items()], className='note')]
    if s['problems']:
        out.append(html.P('Check: '+'; '.join(s['problems']), className='replay-warning'))
    return out


def register(app, root, models, runner_active):
    """runner_active() is true while the Runner tab has a job; the two never share the machine."""
    root, lock, processes = Path(root), RLock(), {}

    def safe(value):
        p = Path(value).resolve()
        if p.parent != root or not (p/'queue.json').exists():
            raise ValueError('Unknown queue.')
        return p

    def live(folder):
        state = q.queue_state(folder)
        proc = processes.get(str(folder))
        # A dead child lingers as a zombie, which still answers signals.
        if state and proc is not None and proc.poll() is not None and state['status'] in ('running', 'starting'):
            state = dict(q.read(Path(folder)/'queue.json'))
            if state['status'] in ('running', 'starting'):
                state.update(status='interrupted', error='The queue process ended unexpectedly; see queue.log. Press Resume.', stopping='')
        return state

    def busy():
        return next((p for p in queues(root)[:20] if (live(p) or {}).get('status') in ('running', 'starting')), None)

    def collect(ids, values, options):
        items = [dict(x, on=False) for x in q.default_items()]
        for ident, value in zip(ids, values):
            items[ident['row']][ident['field']] = bool(value) if ident['field'] == 'on' else value
        o = dict(zip(OPTIONS, options))
        o['all_extremes'] = bool(o['all_extremes'])
        if Path(o['model']) not in models:
            raise ValueError('Choose a robot.')
        return o.pop('model'), items, o

    inputs = [Input('queue-'+k.replace('_', '-'), 'value') for k in OPTIONS]
    states = [State('queue-'+k.replace('_', '-'), 'value') for k in OPTIONS]

    @app.callback(Output(dict(type='queue-time', row=ALL), 'children'), Output('queue-total', 'children'),
                  Input(dict(type='queue-field', row=ALL, field=ALL), 'value'), *inputs, Input('workspace-tab', 'value'), Input('queue-version', 'data'),
                  State(dict(type='queue-field', row=ALL, field=ALL), 'id'))
    def estimate(values, *rest):
        options, tab, ids = rest[:len(OPTIONS)], rest[len(OPTIONS)], rest[-1]
        if tab != 'queue':
            raise PreventUpdate
        rates = q.load_rates(root)
        blank = ['—']*len(q.PRESET)
        try:
            model, items, o = collect(ids, values, options)
            if 'step_s' not in rates:
                with lock:
                    rates = q.load_rates(root)
                    if 'step_s' not in rates:
                        rates['step_s'] = q.benchmark(model)  # About a second, once per results folder.
                        q.atomic_json(root/q.RATES, rates)
            # One item at a time, so one bad row does not hide the others' times.
            times, total, trials, all_measured = list(blank), 0., 0, True
            for i, item in enumerate(items):
                if not item['on']:
                    continue
                prepared = q.plan(model, [item], o)
                seconds = 0.
                for step in prepared['steps']:
                    s, measured = q.step_seconds(step, prepared['options'], prepared['robot'], rates)
                    seconds += s
                    all_measured &= measured
                    trials += step['amount'] if step['kind'] != 'bounds' else 0
                times[i] = '≈ '+duration(seconds)
                total += seconds
            if not any(x['on'] for x in items):
                return times, 'Check at least one test.'
            disk, free = trials*q.BYTES_PER_TRIAL, shutil.disk_usage(root).free
            text = f"Estimated total ≈ {duration(total)} · finishes about {clock(total)} if started now · about {max(.1, disk/1e9):.1f} GB"
            text += ' · from past runs on this machine' if all_measured else ' · estimate from a quick speed test, allow ±30%'
            if disk > .9*free:
                text += f' · NOT ENOUGH DISK SPACE ({free/1e9:.0f} GB free)'
            return times, text
        except (ValueError, TypeError, OSError, KeyError) as exc:
            return blank, str(exc)

    @app.callback(Output('queue-message', 'children'), Output('queue-started', 'data'),
                  Input('queue-run', 'n_clicks'), Input('queue-pause', 'n_clicks'), Input('queue-stop', 'n_clicks'), Input('queue-resume', 'n_clicks'),
                  State('queue-select', 'value'), State(dict(type='queue-field', row=ALL, field=ALL), 'value'),
                  State(dict(type='queue-field', row=ALL, field=ALL), 'id'), *states, prevent_initial_call=True)
    def act(a, b, c, d, selected, values, ids, *options):
        try:
            with lock:
                current = busy()
                if ctx.triggered_id in ('queue-run', 'queue-resume'):
                    if current:
                        return 'A queue is already running.', no_update
                    if runner_active():
                        return 'A Runner test is active. Stop it or wait, then start the queue.', no_update
                    if ctx.triggered_id == 'queue-run':
                        folder = q.save_queue(root, q.plan(*collect(ids, values, options)))
                    else:
                        folder = safe(selected)
                    processes[str(folder)] = q.start(folder)
                    return ('Queue started.' if ctx.triggered_id == 'queue-run' else 'Queue resumed.'), str(folder)
                if not current:
                    return 'No queue is running.', no_update
                after = ctx.triggered_id == 'queue-pause'
                q.request_stop(current, after)
                return ('The queue will stop when the current test finishes.' if after else 'Stopping: active trials finish and completed results are kept.'), no_update
        except (ValueError, TypeError, OSError, KeyError) as exc:
            return str(exc), no_update

    @app.callback(Output('queue-select', 'options'), Output('queue-select', 'value'), Output('queue-status', 'children'), Output('queue-summary', 'children'),
                  Output('queue-run', 'disabled'), Output('queue-pause', 'disabled'), Output('queue-stop', 'disabled'), Output('queue-resume', 'disabled'),
                  Output('queue-settings', 'disabled'), Output('queue-version', 'data'),
                  Input('refresh', 'n_intervals'), Input('workspace-tab', 'value'), Input('queue-started', 'data'), Input('queue-select', 'value'),
                  State('queue-select', 'options'), State('queue-version', 'data'))
    def show(_, tab, started, selected, previous_options, previous):
        if tab != 'queue':
            raise PreventUpdate
        found = queues(root)
        if ctx.triggered_id == 'queue-started' and started:
            selected = started
        if selected not in [str(p) for p in found]:
            selected = str(found[0]) if found else None
        options = []
        for p in found[:100]:
            s = q.read(p/'queue.json', {})
            options.append(dict(label=f"{s.get('options', {}).get('name') or 'Queue'} · {p.name[6:21]} · {len(s.get('steps', []))} tests", value=str(p)))
        current = busy()
        state = live(safe(selected)) if selected else None
        if not state:
            return options, None, '', '', bool(current), True, True, True, bool(current), None
        rates = q.load_rates(root)
        left, per = q.remaining(state, rates)
        running = state['status'] in ('running', 'starting')
        done = sum(s['status'] in q.DONE for s in state['steps'])
        text = f"{state['status'].upper()} · {done} of {len(state['steps'])} tests done"
        if running:
            text += f" · about {duration(left)} left · finishes about {clock(left)}"
        if state.get('stopping') and running:
            text += ' · stopping now' if state['stopping'] == 'now' else ' · stopping after the current test'
        if state.get('error'):
            text += ' · '+state['error']
        summary = Path(selected)/'summary.json'
        version = [selected, summary.stat().st_mtime if summary.exists() else 0, state['status']]
        view = no_update if version == previous else summary_view(selected)
        resumable = not current and state['status'] in ('stopped', 'interrupted')
        return (no_update if options == previous_options else options, selected, [html.Div(text, className='status'), status_table(state, per, left)], view,
                bool(current), not current, not current, not resumable, bool(current), version if version != previous else no_update)

    return busy
