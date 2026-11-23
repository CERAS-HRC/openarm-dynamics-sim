"""Self/floor collision checks on a contact-enabled copy of the arm; the physics never sees them."""
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from .linkage import linkages


def decompose(root, model, name, cell):
    """Replace a concave body's mesh hulls with small convex cells of its surface (slightly oversized, never hollow-filling)."""
    body = next(b for b in root.iter('body') if b.get('name') == name)
    bid = model.body(name).id
    d = mujoco.MjData(model)
    mujoco.mj_kinematics(model, d)
    R, p = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    points = []
    for g in np.flatnonzero((model.geom_bodyid == bid) & (model.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH))):
        k = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[k]:][:model.mesh_vertnum[k]]@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g]
        tris = ((v-p)@R)[model.mesh_face[model.mesh_faceadr[k]:][:model.mesh_facenum[k]]]
        while True:  # Bisect longest edges until triangles are small compared with a cell.
            edge = np.linalg.norm(tris-np.roll(tris, -1, axis=1), axis=2)
            big = edge.max(axis=1) > cell/2
            if not big.any():
                break
            t, e = tris[big], edge[big].argmax(axis=1)
            n = np.arange(len(t))
            a, b, c = t[n, e], t[n, (e+1) % 3], t[n, (e+2) % 3]
            middle = (a+b)/2
            tris = np.concatenate((tris[~big], np.stack((a, middle, c), 1), np.stack((middle, b, c), 1)))
        points.append(tris)
    tris = np.concatenate(points)
    keys = np.floor(tris.mean(axis=1)/cell).astype(int)
    _, owner = np.unique(keys, axis=0, return_inverse=True)
    owner = owner.ravel()
    asset = root.find('asset')
    for geom in body.findall('geom'):
        if geom.get('mesh'):
            geom.attrib.update(contype='0', conaffinity='0')
    for i in range(owner.max()+1):
        v = np.unique(np.round(tris[owner == i].reshape(-1, 3), 7), axis=0)
        # Flat patches need a little thickness to have a convex hull.
        u, s, axes = np.linalg.svd(v-v.mean(axis=0), full_matrices=False)
        if len(v) < 4 or s[-1] < 1e-4:
            normal = np.cross(axes[0], axes[1])
            v = np.concatenate((v-5e-5*normal, v+5e-5*normal))
        v = v[ConvexHull(v).vertices]
        ET.SubElement(asset, 'mesh', name=f'cell_{name}_{i}', vertex=' '.join(f'{x:.8g}' for x in v.ravel()), inertia='shell')
        ET.SubElement(body, 'geom', type='mesh', mesh=f'cell_{name}_{i}', group='5', mass='0', contype='1', conaffinity='1')


class CollisionChecker:
    """Meshes collide as convex hulls, so the check errs toward rejecting borderline poses."""
    def __init__(self, arm, max_overlap_mm=0.):
        root = ET.fromstring(arm.xml)
        for geom in root.iter('geom'):
            geom.attrib.update(contype='1', conaffinity='1')
        cells = arm.cfg.get('collision', {})
        for name in cells.get('decompose_bodies', []):
            decompose(root, arm.model, name, cells.get('cell_mm', 10.)/1000)
        # Hulls and body collision filters are built at compile time, so recompile.
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.linkages = linkages(self.model, arm.cfg)
        self.data = mujoco.MjData(self.model)
        # Floating-point contacts at exactly touching hulls are not overlaps.
        self.tolerance = max(max_overlap_mm/1000, 1e-6)
        m = self.model
        self.names = ['floor' if b == 0 else m.body(b).name or f'body_{b}' for b in range(m.nbody)]
        limits = np.deg2rad([j['limits_deg'] for j in arm.cfg['joints']])
        # Pairs touching in the reference pose are by design (e.g. a housing on
        # the link it drives), like an allowed-collision matrix.
        self.ignored = set()
        self.ignored = set(self.contacts(np.clip(0, limits[:, 0], limits[:, 1])))

    def contacts(self, q):
        """{(body, body): deepest overlap in mm} for overlaps beyond the tolerance."""
        m, d = self.model, self.data
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for link in self.linkages:
            link.place(m, d)
        mujoco.mj_collision(m, d)
        found = {}
        c = d.contact
        for g1, g2, dist in zip(c.geom1, c.geom2, c.dist):
            if dist >= -self.tolerance:
                continue
            pair = tuple(sorted((self.names[m.geom_bodyid[g1]], self.names[m.geom_bodyid[g2]])))
            if pair not in self.ignored:
                found[pair] = max(found.get(pair, 0.), -dist*1000)
        return found

    def path_contacts(self, a, b, step_deg=2.):
        """Contacts along the planned A -> B path, every step_deg of the largest joint move."""
        # smooth_move only retimes a straight joint-space line, so sample its length evenly.
        a, b = np.deg2rad(a), np.deg2rad(b)
        n = max(2, int(np.ceil(np.rad2deg(np.max(np.abs(b-a), initial=0))/step_deg))+1)
        for s in np.linspace(0, 1, n):
            found = self.contacts(a + (b-a)*s)
            if found:
                return found
        return {}


def describe(found):
    return ', '.join(f'{a}-{b} {mm:.1f} mm' for (a, b), mm in sorted(found.items(), key=lambda x: -x[1]))
