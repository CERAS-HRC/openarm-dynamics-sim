"""Re-apply everything Onshape cannot export for robots/dynamics_sim_arm. Run once after EVERY re-export:

    .venv/bin/python scripts/prepare_dynamics_sim_arm.py

Edits the raw URDF in place (the raw file is kept as *.onshape.urdf.raw) and updates joint limits, J6 ratings and
pin points in the .sim.yaml. Saved poses, gains and other settings are kept.
"""
from pathlib import Path
import shutil
import struct
import sys
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
MARK = ' Prepared by scripts/prepare_dynamics_sim_arm.py; do not hand-edit, re-run it after re-exporting. '
MOTORS = {'03': (.88, 60., 20., 100., 195.), '06': (.621, 36., 11., 100., 480.), '00': (.31, 14., 5., 100., 315.)}  # kg, peak/rated Nm, rated/no-load rpm
HOUSING = {'rs03': '03', 'rs06': '06', 'rs00': '00'}
DRIVE = dict(j1='03', j2='03', j3='06', j4='06', j5='00', j7='00')
SWIVEL, MOTOR, BEARING = 'j6_swivel', 'j6_motor', 'j6_crank_bearing'
CRANK, LEVER, ROCKER = 'lever_to_l5', 'lever', 'lever_to_l6'
MARKERS = {'j6_pin_lever', 'j6_pin_l6'}
BORES = ('l5', 'l5_2')       # parts holding the two swivel bearings
BORE_RADIUS_M = .010


def pose(origin):
    M = np.eye(4)
    if origin is not None:
        M[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        M[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return M


def put(origin, M):
    origin.set('xyz', ' '.join(f'{x:.12g}' for x in M[:3, 3]))
    origin.set('rpy', ' '.join(f'{x:.12g}' for x in Rotation.from_matrix(M[:3, :3]).as_euler('xyz')))


def prepare(path):
    path = Path(path)
    if 'prepare_dynamics_sim_arm' in path.read_text():
        raise ValueError(f'{path.name} is already prepared; re-export it from Onshape first.')
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(path, parser)
    robot = tree.getroot()
    joints = {j.get('name'): j for j in robot.findall('joint')}
    links = {l.get('name'): l for l in robot.findall('link')}
    missing = {SWIVEL, MOTOR, BEARING}-set(joints) | ({CRANK, LEVER, ROCKER, *BORES} | MARKERS)-set(links)
    if missing or any('loop_closure' in n for n in joints):
        raise ValueError(f'Unexpected export (missing {sorted(missing)} or a loop closure); check grounding and mates in Onshape.')
    parent = {j.find('child').get('link'): j for j in robot.findall('joint')}

    def world(link):
        return world(parent[link].find('parent').get('link'))@pose(parent[link].find('origin')) if link in parent else np.eye(4)

    def mesh(link):
        node = links[link].find('visual')
        raw = (path.parent.parent/'meshes'/node.find('.//mesh').get('filename').split('/')[-1]).read_bytes()
        n = struct.unpack('<I', raw[80:84])[0]
        v = np.frombuffer(raw[84:], np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), n)['v'].astype(float)
        M = world(link)@pose(node.find('origin'))
        return v.reshape(-1, 3)@M[:3, :3].T+M[:3, 3]

    local = lambda M, p: M[:3, :3].T@(p-M[:3, 3])
    W = {n: world(n) for n in links}
    swivel = W['l6'][:3, 3]
    axis = W['l6'][:3, :3]@np.fromstring(joints[SWIVEL].find('axis').get('xyz'), sep=' ')
    axis /= np.linalg.norm(axis)
    for j in (MOTOR, BEARING):
        child = joints[j].find('child').get('link')
        a = W[child][:3, :3]@np.fromstring(joints[j].find('axis').get('xyz'), sep=' ')
        if abs(abs(a@axis)-1) > 1e-6:
            raise ValueError(f'{j} is not parallel to {SWIVEL}')
    along = lambda p: float(axis@(p-swivel))
    planar = lambda v: v-np.multiply.outer(v@axis, axis)

    # Both pins at the lever's mid-thickness; markers supply the pin axes only.
    mid = np.mean([f(np.array([along(p) for p in mesh(LEVER)])) for f in (np.min, np.max)])
    on_mid = lambda p: p+axis*(mid-along(p))
    center = lambda n: (W[n]@np.r_[np.fromstring(links[n].find('inertial/origin').get('xyz'), sep=' '), 1])[:3]
    A, B = W[CRANK][:3, 3], on_mid(W[LEVER][:3, 3])
    Lp, P = on_mid(center('j6_pin_lever')), on_mid(center('j6_pin_l6'))
    r, L = np.linalg.norm(planar(B-A)), np.linalg.norm(planar(Lp-B))
    e1 = planar(B-A)/r
    e2 = np.cross(axis, e1)
    x, y = e1@(P-A), e2@(P-A)
    D = np.hypot(x, y)
    beta = np.arccos((r*r+D*D-L*L)/(2*r*D))
    theta = min((np.angle(np.exp(1j*(np.arctan2(y, x)+s*beta))) for s in (1, -1)), key=abs)
    print(f'Linkage mm: crank {r*1e3:.2f}, lever {L*1e3:.2f}, rocker {np.linalg.norm(planar(P-swivel))*1e3:.2f}, '
          f'ground {np.linalg.norm(planar(A-swivel))*1e3:.2f}; crank closed by {np.rad2deg(theta):+.2f} deg')
    if abs(theta) > np.deg2rad(60):
        raise ValueError('The linkage was exported far from its assembled position.')

    # Rotate crank and lever into the closed pose, then weld them (weight only).
    turn = lambda angle: Rotation.from_rotvec(axis*angle).as_matrix()
    crank = W[CRANK].copy()
    crank[:3, :3] = turn(theta)@crank[:3, :3]
    B2 = A+turn(theta)@(B-A)
    v0, v1 = turn(theta)@planar(Lp-B), planar(P-B2)
    phi = np.arctan2(axis@np.cross(v0, v1), v0@v1)
    lever = W[LEVER].copy()
    lever[:3, :3] = turn(theta+phi)@lever[:3, :3]
    lever[:3, 3] = A+turn(theta)@(W[LEVER][:3, 3]-A)
    grand = W[joints[MOTOR].find('parent').get('link')]
    put(joints[MOTOR].find('origin'), np.linalg.inv(grand)@crank)
    put(joints[BEARING].find('origin'), np.linalg.inv(crank)@lever)
    for n in (MOTOR, BEARING):
        joints[n].set('type', 'fixed')
    pins = dict(crank_pin_m=local(W[CRANK], B), lever_pin_m=local(W[LEVER], Lp))

    # One hinge midway between the two bearing bores.
    bores = []
    for n in BORES:
        v = mesh(n)
        near = v[abs(np.linalg.norm(planar(v-swivel), axis=1)-BORE_RADIUS_M) < 5e-5]
        bores.append(np.mean([min(map(along, near)), max(map(along, near))]))
    shift = float(np.mean(bores))
    print(f'Swivel bearings {bores[0]*1e3:+.1f} / {bores[1]*1e3:+.1f} mm along the exported hinge; hinge moved {shift*1e3:+.1f} mm')
    moved = W['l6'].copy()
    moved[:3, 3] += axis*shift
    put(joints[SWIVEL].find('origin'), np.linalg.inv(W[joints[SWIVEL].find('parent').get('link')])@moved)
    offset = np.linalg.inv(moved)@W['l6']
    for node in [links['l6'].find('inertial/origin'), *[x.find('origin') for x in links['l6'].findall('visual')],
                 *[j.find('origin') for j in joints.values() if j.find('parent').get('link') == 'l6']]:
        put(node, offset@pose(node))
    pins['rocker_pin_m'] = local(moved, P)

    # J6 limits: motor ratings through the extreme lever ratios over the swivel range.
    lim = joints[SWIVEL].find('limit')
    ratios = []
    for q in np.linspace(float(lim.get('lower')), float(lim.get('upper')), 241):
        p = swivel+turn(q)@(P-swivel)
        x, y = e1@(p-A), e2@(p-A)
        D = np.hypot(x, y)
        c = (r*r+D*D-L*L)/(2*r*D)
        if abs(c) >= 1:
            raise ValueError('The linkage cannot close over the J6 range.')
        t = min((np.angle(np.exp(1j*(np.arctan2(y, x)+s*np.arccos(c)))) for s in (1, -1)), key=abs)
        b = A+turn(t)@(B-A)
        u = planar(p-b)/np.linalg.norm(planar(p-b))
        ratios.append(abs((axis@np.cross(b-A, u))/(axis@np.cross(p-swivel, u))))   # swivel speed / motor speed
    lo, hi = float(min(ratios)), float(max(ratios))
    m = MOTORS['00']
    j6 = dict(mass_kg=m[0], rated_torque_nm=round(m[2]/hi, 3), peak_torque_nm=round(m[1]/lo, 3),
              rated_speed_rpm=round(m[3]*hi, 2), no_load_speed_rpm=round(m[4]*hi, 2), gear_ratio=10.,
              lever_ratio_range=[round(1/hi, 4), round(1/lo, 4)],
              source='RobStride 00 through the J6 lever: peak torque and speeds use the most favourable lever ratio')
    print(f'Swivel/motor speed ratio {lo:.3f}..{hi:.3f}: J6 peak {j6["peak_torque_nm"]} Nm, no-load {j6["no_load_speed_rpm"]} rpm')

    for name, j in joints.items():
        if j.get('type') == 'fixed':
            continue
        peak, rpm = (j6['peak_torque_nm'], j6['no_load_speed_rpm']) if name == SWIVEL else (MOTORS[DRIVE[name[:2]]][1], MOTORS[DRIVE[name[:2]]][4])
        j.find('limit').set('effort', f'{peak:g}')
        j.find('limit').set('velocity', f'{rpm*np.pi/30:.6f}')
    for name, link in links.items():
        if name[:4] in HOUSING:
            mass, node = MOTORS[HOUSING[name[:4]]][0], link.find('inertial')
            scale = mass/float(node.find('mass').get('value'))
            node.find('mass').set('value', f'{mass:g}')
            for k, v in list(node.find('inertia').attrib.items()):
                node.find('inertia').set(k, f'{float(v)*scale:.12g}')
    for name in MARKERS:
        robot.remove(links[name])
        robot.remove(parent[name])
    robot.insert(0, ET.Comment(MARK))

    settings = path.with_suffix('.sim.yaml')
    header = ''.join(l+'\n' for l in settings.read_text().splitlines() if l.startswith('#')) if settings.exists() else HEADER
    c = yaml.safe_load(settings.read_text()) if settings.exists() else template()
    old = {j['name']: j for j in c['joints']}
    c['joints'] = []
    for name in sorted(n for n, j in joints.items() if j.get('type') != 'fixed'):
        limit = joints[name].find('limit')
        j = old.get(name) or dict(kp_nm_rad=120 if name[:2] < 'j5' else 60, kd_nm_s_rad=12 if name[:2] < 'j5' else 5)
        j = dict(name=name, type='revolute', motor='00-J6' if name == SWIVEL else DRIVE[name[:2]],
                 limits_deg=[round(float(np.rad2deg(float(limit.get(k)))), 6) for k in ('lower', 'upper')],
                 kp_nm_rad=j['kp_nm_rad'], kd_nm_s_rad=j['kd_nm_s_rad'])
        c['joints'].append(j)
    c['motors']['00-J6'] = j6
    c['linkages'] = [dict(joint=SWIVEL, crank=CRANK, lever=LEVER, support='rs00_1',
                          **{k: [round(float(x), 9) for x in v] for k, v in pins.items()})]
    c['sweep'].update({k: [float(np.clip(q, *j['limits_deg'])) for q, j in zip(c['sweep'][k], c['joints'])] for k in ('a_deg', 'b_deg')})
    text = header+yaml.safe_dump(c, sort_keys=False)   # Serialize first: a failure must leave the raw export untouched.
    shutil.copyfile(path, path.with_suffix('.onshape.urdf.raw'))
    tree.write(path, encoding='unicode')
    settings.write_text(text)
    sys.path.insert(0, str(ROOT/'src'))
    from arm_design.config import load_config
    from arm_design.model import build_arm
    from arm_design.linkage import linkages
    arm = build_arm(load_config(settings))
    linkages(arm.model, arm.cfg)
    print(f'Prepared {path.name} and {settings.name}: {arm.model.nv} joints, {arm.model.body_mass.sum():.4f} kg')


HEADER = '''# dynamics_sim_arm (Onshape): J1/J2: 03, J3/J4: 06, J5/J6/J7: 00. Joint angles are degrees.
# The URDF, joint limits, 00-J6 ratings and linkages pins are written by scripts/prepare_dynamics_sim_arm.py.
# J6 is driven at the swivel (no moving lever); 00-J6 is the motor through the most favourable lever ratio, so it leans high.
# linkages: the swivel load is replaced by bearing + lever pin loads (see docs/MODEL.md).
# fea.exclude_bodies lists every motor EXCEPT rs00_2: the J6 motor, l5 and l5_2 are one rigid part named l5.
# collision.decompose_bodies: concave forearm parts collide as small convex cells instead of one filled hull.
# Tip is the drumstick's striking end. The floor height is assumed, not measured.
'''


def template():
    c = yaml.safe_load((ROOT/'robots/right_arm_assembly/urdf/right_arm_assembly.sim.yaml').read_text())
    c.update(name='Dynamics sim arm / RobStride / J6 lever')
    c['urdf'].update(path='dynamics_sim_arm.urdf')
    c['simulation']['floor_z_m'] = -.686
    c['fea'] = dict(exclude_bodies=['rs03*', 'rs06*', 'rs00', 'rs00_1'])
    c['collision'] = dict(decompose_bodies=['l5', 'l5_2'], cell_mm=10.)
    c['sweep'].update(a_deg=[0.]*7, b_deg=[45., -45., 30., 30., 30., 15., 30.])
    return c


if __name__ == '__main__':
    prepare(sys.argv[1] if len(sys.argv) > 1 else ROOT/'robots/dynamics_sim_arm/urdf/dynamics_sim_arm.urdf')
