import importlib.util
import json
import shutil
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml

from arm_design.campaign import campaign
from arm_design.collision import CollisionChecker
from arm_design.config import ROOT, load_config
from arm_design.fea import FEARecording
from arm_design.fea_cases import part_cases
from arm_design.linkage import Linkage
from arm_design.model import build_arm
from arm_design.trial_playback import display_pose, model_geometry

DYN = ROOT/'robots/dynamics_sim_arm/urdf/dynamics_sim_arm.sim.yaml'


def row(rec, label):
    return next(i for i, x in enumerate(rec.interfaces) if x['label'] == label)


def test_prepare_script_reproduces_the_checked_in_model(tmp_path):
    robot = tmp_path/'dynamics_sim_arm'
    shutil.copytree(DYN.parents[1], robot)
    urdf = robot/'urdf/dynamics_sim_arm.urdf'
    shutil.copyfile(DYN.parent/'dynamics_sim_arm.onshape.urdf.raw', urdf)
    spec = importlib.util.spec_from_file_location('prep', ROOT/'scripts/prepare_dynamics_sim_arm.py')
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    prep.prepare(urdf)
    assert urdf.read_text() == (DYN.parent/'dynamics_sim_arm.urdf').read_text()
    assert yaml.safe_load(urdf.with_suffix('.sim.yaml').read_text()) == yaml.safe_load(DYN.read_text())
    with pytest.raises(ValueError, match='already prepared'):
        prep.prepare(urdf)


def test_model_parts_and_closure_over_the_range():
    arm = build_arm(load_config(DYN))
    m = arm.model
    rec = FEARecording(arm, arm.data())
    parts = {p['name']: p for p in rec.parts()}
    assert parts['l5']['bodies'] == ['l5', 'rs00_2', 'lever_to_l5', 'lever', 'l5_2']
    assert parts['l5']['visual_bodies'] == ['l5', 'rs00_2', 'l5_2']
    assert np.isclose(sum(p['mass_kg'] for p in parts.values())+sum(m.body(b).mass[0] for b in ('rs03', 'rs03_1', 'rs06', 'rs06_1', 'rs00', 'rs00_1')), m.body_mass.sum())
    link, d = rec.linkages[0], mujoco.MjData(m)
    for q in np.deg2rad(np.linspace(-28.7, 28.7, 41)):
        d.qpos[:] = [.3, -.7, .4, .5, -.6, q, 1.]
        mujoco.mj_kinematics(m, d)
        s = link.solve(d)
        assert abs(np.linalg.norm(s['P']-s['B'])-link.L) < 1e-6
        assert abs(d.xaxis[link.joint]@s['u']) < 1e-6
        before = d.geom_xpos.copy()
        link.place(m, d)
        moved = np.flatnonzero(np.linalg.norm(d.geom_xpos-before, axis=1) > 1e-9)
        assert set(m.geom_bodyid[moved]) <= {link.crank, link.lever}


def test_loads_balance_and_match_the_hand_calculation():
    arm = build_arm(load_config(DYN))
    m, d = arm.model, arm.data()
    rec = FEARecording(arm, d)
    rng = np.random.default_rng(1)
    for _ in range(10):
        d.qpos[:] = rng.uniform(-.45, .45, m.nq)
        d.qvel[:] = rng.uniform(-2, 2, m.nv)
        d.ctrl[:] = d.qpos+rng.uniform(-.2, .2, m.nv)
        mujoco.mj_forward(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        loads, points = rec.interface_loads(m, d)
        link = rec.linkages[0]
        s = link.solve(d)
        torque = d.qfrc_actuator[link.joint]+d.qfrc_passive[link.joint]+d.qfrc_constraint[link.joint]
        assert np.isclose(np.linalg.norm(loads[row(rec, 'l6 <- lever'), :3]), abs(torque/s['arm']), rtol=1e-6)
        # A free bearing carries no moment about the swivel axis.
        i = row(rec, 'l6 <- l5')
        axis = d.xmat[link.rocker].reshape(3, 3).T@d.xaxis[link.joint]
        assert abs(axis@loads[i, 3:]) < 1e-8
        for part in ('lever', 'lever_to_l5'):
            k = [n for n, x in enumerate(rec.interfaces) if x['part'] == part]
            assert np.allclose(sum(loads[n, :3] for n in k), 0, atol=1e-8)
            assert np.allclose(sum(loads[n, 3:]+np.cross(points[n], loads[n, :3]) for n in k), 0, atol=1e-8)
        # Two-force member: the load stays along the lever's own, fixed pin line.
        k = [row(rec, 'lever <- lever_to_l5'), row(rec, 'lever <- l6')]
        line = points[k[1]]-points[k[0]]
        assert np.linalg.norm(np.cross(line, loads[k[0], :3])) < 1e-6*max(1, np.linalg.norm(loads[k[0], :3]))
        assert np.allclose(points[k[1]], link.lever_pin, atol=1e-6)
        assert np.allclose(points[row(rec, 'lever_to_l5 <- lever')], link.crank_pin, atol=1e-6)


@pytest.mark.parametrize('swivel_deg,j4', [(-25., .6), (0., .6), (25., -0.6)])
def test_lever_force_matches_a_true_closed_loop(swivel_deg, j4):
    """Independent reference: hinge the crank and lever, close the loop, hold still by driving only the crank."""
    cfg = load_config(DYN)
    cfg['analysis_payload'] = dict(mass_kg=5., com_offset_m=[-.25, .15, .25], box_size_m=[.05]*3)
    cfg['simulation']['step_s'] = .0002
    arm = build_arm(cfg)
    link = FEARecording(arm, arm.data()).linkages[0]
    q0 = np.array([.4, -1., .3, j4, .5, np.deg2rad(swivel_deg), .2])
    d = arm.data()
    d.qpos[:] = q0
    mujoco.mj_kinematics(arm.model, d)
    s = link.solve(d)
    root = ET.fromstring(arm.xml)
    root.remove(root.find('actuator'))
    body = {b.get('name'): b for b in root.iter('body')}
    text = lambda v: ' '.join(f'{x:.12g}' for x in v)
    R = lambda b: d.xmat[b].reshape(3, 3)
    for name, joint, axis, pos in (('lever_to_l5', 'crank', link.axis, np.zeros(3)),
            ('lever', 'lever', R(link.lever).T@R(link.crank)@link.axis, R(link.lever).T@(link.neutral_pin(d)-d.xpos[link.lever]))):
        ET.SubElement(body[name], 'joint', name=joint, type='hinge', axis=text(axis), pos=text(pos), damping='1', armature='.05')
    for joint in root.iter('joint'):
        joint.set('armature', '.1')  # Settles the stiff reference; no effect at rest.
    ET.SubElement(body['lever'], 'site', name='pa', pos=text(link.lever_pin))
    ET.SubElement(body['l6'], 'site', name='pb', pos=text(link.rocker_pin))
    eq = ET.SubElement(root, 'equality')
    ET.SubElement(eq, 'connect', site1='pa', site2='pb', solref='.01 1', solimp='.999 .9999 .0001')
    act = ET.SubElement(root, 'actuator')
    hold = dict(zip([j['name'] for j in cfg['joints']], q0))
    hold['crank'] = s['theta']
    del hold['j6_swivel']
    for j in hold:
        ET.SubElement(act, 'position', joint=j, kp='2000', kv='200')
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    full = mujoco.MjData(m)
    for j, v in dict(hold, j6_swivel=q0[5], lever=s['phi']-s['theta']).items():
        full.joint(j).qpos = v
    full.ctrl[:] = list(hold.values())
    for _ in range(20000):
        mujoco.mj_step(m, full)
    mujoco.mj_forward(m, full)
    mujoco.mj_rnePostConstraint(m, full)
    assert np.abs(full.qvel).max() < 1e-2
    reference = np.linalg.norm(full.cfrc_int[m.body('lever').id][3:])
    for j in cfg['joints']:
        d.joint(j['name']).qpos = full.joint(j['name']).qpos
    mujoco.mj_forward(arm.model, d)
    d.qacc[:] = 0
    d.qvel[:] = 0
    mujoco.mj_rnePostConstraint(arm.model, d)
    rec = FEARecording(arm, d)
    loads, _ = rec.interface_loads(arm.model, d)
    force = np.linalg.norm(loads[row(rec, 'l6 <- lever'), :3])
    assert force > 5
    # Differences: the neglected lever/crank weight and the reference's slightly soft pin.
    assert abs(force-reference) < .1
    # Upstream of the whole assembly the two models agree too.
    assert np.isclose(np.linalg.norm(loads[row(rec, 'l5 <- rs00'), :3]),
                      np.linalg.norm(full.cfrc_int[m.body('l5').id][3:]), atol=.5)


def test_j6_range_is_clear_but_real_wrist_collisions_are_found():
    checker = CollisionChecker(build_arm(load_config(DYN)))
    assert not checker.ignored
    for s in np.linspace(-28.7, 28.7, 15):
        assert not checker.contacts(np.deg2rad([0, 0, 0, 0, 0, s, 0]))
    assert ('base_2', 'rs00_2') in checker.contacts(np.deg2rad([0, 0, 0, 0, 0, 25, 90]))


def test_linkage_configuration_errors():
    cfg = load_config(DYN)
    arm = build_arm(cfg)
    bad = dict(cfg['linkages'][0], rocker_pin_m=[.1, 0, 0])
    with pytest.raises(ValueError, match='open by'):
        Linkage(arm.model, bad)
    cfg['fea']['exclude_bodies'] = ['rs*']
    with pytest.raises(ValueError, match='linkage needs'):
        FEARecording(build_arm(cfg), arm.data())


def test_campaign_fea_cases_dashboard_and_playback(tmp_path):
    from arm_design import dashboard
    from arm_design.fea_view import mesh_figure
    from arm_design.trial_playback import playback
    from arm_design.trial_recording import regenerate
    cfg = load_config(DYN)
    cfg['sweep'].update(settle_s=.2, hold_s=.1)
    folder, rec = campaign(cfg, tmp_path, 3, 0, (1., 1.5), 1, 2, log=lambda *a, **k: None)
    run = folder.parent
    meta = json.loads((folder/'metadata.json').read_text())
    for part in meta['parts']:
        for case in part_cases(folder, part['name'], all_extremes=True):
            assert case['residual_n'] < 1e-6 and case['moment_residual_nm'] < 1e-6
    case = part_cases(folder, 'l6')[0]
    assert case['fixed'] == 'rs00_1' and {r['other'] for r in case['applied']} == {'l5', 'lever'}
    case = part_cases(folder, 'l5')[0]
    assert case['fixed'] == 'rs00' and {r['other'] for r in case['applied']} == {'l6', 'lever_to_l5'}
    # Every loaded connection has its own complete FEA table.
    table, options, _ = dashboard.fea_limits(run, 'l5', 'random', 'accepted')
    assert len(options) == 24 and len(dashboard.fea_limits(run, 'l3', 'random', 'accepted')[1]) == 12
    for part in ('l5', 'l6', 'lever', 'lever_to_l5'):
        figure = mesh_figure(meta, None, part, run)
        assert any(t.type == 'mesh3d' for t in figure.data)
    refs = meta['interfaces']
    trial = next(i for i, s in enumerate(rec.sweeps) if s['counted'])
    i = next(i for i, x in enumerate(refs) if x['label'] == 'lever <- l6')
    data = playback(run, trial, 'lever', i, 'force_N', 'maximum')
    assert data['members'] == ['lever'] and len(data['mates']) == 2
    # Browser pose coordinates include the two display-only lever hinges.
    model = model_geometry(str(run/'model.mjb'), (run/'model.mjb').stat().st_mtime_ns)
    assert len(model['qpos0']) == len(model['joints']) == len(display_pose(run, [0]*7)) == 9
    assert display_pose(run, [0]*7)[7:] == pytest.approx([0, 0], abs=1e-6)
    assert regenerate(run, trial)
