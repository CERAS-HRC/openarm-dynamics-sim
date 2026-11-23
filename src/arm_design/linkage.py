"""Lever-driven hinge: the simulator drives the hinge directly; this resolves the lever's two-force load.

Config `linkages` entries name the hinge and the crank/lever bodies (rigid weight on the hinge's parent).
Pin points are in each body's own frame, at the lever's mid-thickness. Lever inertia and friction are ignored.
"""
import mujoco
import numpy as np

KEYS = ('crank_pin_m', 'lever_pin_m', 'rocker_pin_m')


def validate_linkages(c):
    from .config import vector
    for link in c.get('linkages', []):
        for k in KEYS:
            vector(link[k], 3, k)
        if link['joint'] not in [j['name'] for j in c['joints']]:
            raise ValueError(f"Linkage joint {link['joint']!r} is not a controlled joint")


def rotation(axis, angle):
    R = np.zeros(9)
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis, angle)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class Linkage:
    """All inputs come from the compiled model, whose neutral pose is the closed linkage."""
    def __init__(self, m, cfg):
        self.cfg = cfg
        self.joint, self.crank, self.lever = m.joint(cfg['joint']).id, m.body(cfg['crank']).id, m.body(cfg['lever']).id
        self.rocker = int(m.jnt_bodyid[self.joint])
        self.crank_pin, self.lever_pin, self.rocker_pin = (np.array(cfg[k], float) for k in KEYS)
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        # Constants in the crank body's unmoved frame.
        R = d.xmat[self.crank].reshape(3, 3)
        self.axis = R.T@d.xaxis[self.joint]
        planar = lambda v: v-self.axis*(self.axis@v)
        b = planar(self.crank_pin)
        self.r = np.linalg.norm(b)
        self.e1 = b/self.r
        self.e2 = np.cross(self.axis, self.e1)
        tip = R.T@(self.point(d, self.lever, self.lever_pin)-d.xpos[self.crank])
        pin = R.T@(self.point(d, self.rocker, self.rocker_pin)-d.xpos[self.crank])
        if np.linalg.norm(tip-pin) > 1e-4:
            raise ValueError(f"Linkage {cfg['joint']} is open by {1000*np.linalg.norm(tip-pin):.2f} mm at zero")
        self.L = np.linalg.norm(planar(tip-self.crank_pin))
        x, y = self.e1@pin, self.e2@pin
        self.branch = 1. if abs(np.angle(np.exp(1j*(np.arctan2(y, x)+self._beta(x, y))))) < \
            abs(np.angle(np.exp(1j*(np.arctan2(y, x)-self._beta(x, y))))) else -1.
        self.lever0 = self._lever_angle(x, y, 0.)
        # The crank drives the rocker through one sign over the whole range, so no dead points.
        lo, hi = m.jnt_range[self.joint]
        for q in np.linspace(lo, hi, 61):
            d.qpos[m.jnt_qposadr[self.joint]] = q
            mujoco.mj_kinematics(m, d)
            if min(self.solve(d)['sines']) < .1:
                raise ValueError(f"Linkage {cfg['joint']} locks within its joint range")

    @staticmethod
    def point(d, body, local):
        return d.xpos[body]+d.xmat[body].reshape(3, 3)@local

    def _beta(self, x, y):
        D = np.hypot(x, y)
        c = (self.r**2+D*D-self.L**2)/(2*self.r*D)
        if abs(c) > 1:
            raise ValueError('Lever linkage cannot close at this joint angle')
        return np.arccos(c)

    def _lever_angle(self, x, y, theta):
        return np.arctan2(y-self.r*np.sin(theta), x-self.r*np.cos(theta))

    def solve(self, d):
        """Crank/lever rotations from neutral, world pins and the lever direction (crank -> rocker)."""
        R, origin = d.xmat[self.crank].reshape(3, 3), d.xpos[self.crank]
        P = self.point(d, self.rocker, self.rocker_pin)
        p = R.T@(P-origin)
        x, y = self.e1@p, self.e2@p
        theta = np.angle(np.exp(1j*(np.arctan2(y, x)+self.branch*self._beta(x, y))))
        phi = np.angle(np.exp(1j*(self._lever_angle(x, y, theta)-self.lever0)))
        Rc = R@rotation(self.axis, theta)
        B = origin+Rc@self.crank_pin
        u = (P-B)/np.linalg.norm(P-B)
        axis = d.xaxis[self.joint]
        moment_arm = axis@np.cross(P-d.xanchor[self.joint], u)
        rocker = P-d.xanchor[self.joint]
        rocker = np.linalg.norm(rocker-axis*(axis@rocker))
        crank_arm = axis@np.cross(B-origin, u)
        return dict(theta=theta, phi=phi, origin=origin, B=B, P=P, u=u, arm=moment_arm, crank_arm=crank_arm,
                    sines=(abs(moment_arm)/rocker, abs(crank_arm)/self.r),
                    crank_rotation=Rc, lever_rotation=d.xmat[self.lever].reshape(3, 3)@rotation(
                        d.xmat[self.lever].reshape(3, 3).T@axis, phi))

    def loads(self, d, force, moment):
        """World loads replacing the hinge wrench (on the rocker about the anchor): pin joints carry no axis moment."""
        s = self.solve(d)
        anchor, axis = d.xanchor[self.joint], d.xaxis[self.joint]
        # Compression positive: force of the lever on the rocker along u.
        push = (axis@moment)/s['arm']*s['u']
        bearing = force-push, moment-np.cross(s['P']-anchor, push)
        return s, push, bearing, np.cross(s['B']-s['origin'], push)

    def place(self, m, d):
        """Move the crank/lever visuals to their true pose (they are rigid weight in the physics)."""
        s = self.solve(d)
        pivot = {self.crank: (s['origin'], s['crank_rotation'], d.xpos[self.crank]),
                 self.lever: (s['B'], s['lever_rotation'], self.neutral_pin(d))}
        for g in np.flatnonzero(np.isin(m.geom_bodyid, (self.crank, self.lever))):
            b = int(m.geom_bodyid[g])
            new, R, old = pivot[b]
            turn = R@d.xmat[b].reshape(3, 3).T
            d.geom_xpos[g] = new+turn@(d.geom_xpos[g]-old)
            d.geom_xmat[g] = (turn@d.geom_xmat[g].reshape(3, 3)).ravel()

    def neutral_pin(self, d):
        return self.point(d, self.crank, self.crank_pin)


def linkages(m, cfg):
    return [Linkage(m, x) for x in cfg.get('linkages', [])]
