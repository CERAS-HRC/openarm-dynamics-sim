# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MuJoCo-based dynamics sim for 7-DOF OpenArm-style arms with RobStride actuators. Rigid-body loads only: stress, deflection and fatigue are done in an external FEA tool from the exports. Package `arm_design` under `src/`; entry point `arm-sim` (`arm_design.cli:main`). Two front ends over the same physics: a native GLFW/ImGui viewer (jog, A/B sweeps, FEA recorder) and **Load Lab**, a local Dash web dashboard (run campaigns, browse load distributions, play back trials).

Four robots: `configs/arm.yaml` (generated arm, the `--config` default), `robots/moving_asm/urdf/moving_asm.urdf` (CAD arm, parts `part_1..7`), `robots/right_arm_assembly/urdf/right_arm_assembly.urdf` (older drumming arm, parts `l1..l6` and `base` = base+screwon+drumstick), `robots/dynamics_sim_arm/urdf/dynamics_sim_arm.urdf` (current drumming arm with a lever-driven J6; parts `l1..l6`, `base_2` = tool, `lever_to_l5` = crank, `lever`; the dashboard's default robot).

## Commands

```bash
./scripts/setup.sh                      # create .venv (python3.13, override with PYTHON_BIN) + editable install with dev extras, pinned by requirements-lock.txt
./scripts/start.sh [--urdf path.urdf]   # GUI viewer (= python -m arm_design.cli view)
.venv/bin/arm-sim dashboard [--results results] [--port 8050]   # Load Lab at http://127.0.0.1:8050

.venv/bin/python -m pytest              # all tests
.venv/bin/python -m pytest tests/test_urdf.py::test_name   # single test

# Headless (no window)
.venv/bin/arm-sim sweep [--urdf ...] [--direction b-to-a] [--duration S] --out results/x   # exits 2 if tracking fails
.venv/bin/arm-sim export [--urdf ...] --out results/x   # writes generated MJCF arm.xml
.venv/bin/arm-sim campaign [--urdf ...] --samples N --iterations K [--workers W] [--include-failed] [--live] [--full-history]   # random A/B sweeps + refinement
.venv/bin/arm-sim campaign [--urdf ...] --aggressive --samples N --search-steps 8 --min-duration .1 --max-duration 2 [--skip-timestep-validation]   # duration search, both directions
.venv/bin/arm-sim restrained [--urdf ...] --poses N [--poses-file f.json] [--max-trials N] [--workers W]   # tip welded to world, all 3^n torque combinations
.venv/bin/arm-sim bounds [--urdf ...] --max-contact-force-n F [--max-contact-moment-nm M]   # static gravity + blocked-tip (+ clamped-tip with M) bounding cases (seconds)
.venv/bin/arm-sim fea-cases --results <campaign-*, restrained-*, fea-* or bounds-*> --part part_3   # per-part FEA setup: fixed face, remote loads, body loads
.venv/bin/arm-sim watch [--results <campaign-progress.csv, convergence.csv or folder>]   # live/finished campaign convergence plot
.venv/bin/arm-sim replay --results <campaign-*/fea-*> --trial I   # window: rerun one campaign sweep
.venv/bin/python -m arm_design.trial_recording <run-folder> <trial>   # regenerate + verify one trial's full recording
```

`--config` and `--urdf` are mutually exclusive; `--package-root` (repeatable) requires `--urdf`. `--out` defaults to `results/` (gitignored, can be many GB). Passing a `.sim.yaml` to `--config` works too. `--workers` defaults to the CPU count for `campaign`/`bounds` but 1 for `restrained`.

```bash
.venv/bin/python scripts/prepare_dynamics_sim_arm.py   # after EVERY Onshape re-export of dynamics_sim_arm; refuses to run twice
```

## Architecture

**Config is the single source of truth.** Everything is driven by a YAML dict (`configs/arm.yaml` for the built-in arm, or `<robot>.sim.yaml` for imports). `config.validate` checks it; `config.save_controls` writes poses, gains and geometry back to it from the UI. Joint angles in YAML are degrees; units are in field names (`_mm`, `_kg`, `_deg`, `_nm`). Optional `analysis_payload` (`payload.py`) adds a box body with explicit inertia at the tip; `drive.trajectory_feedforward` and `sweep.load_acceptance` are set by aggressive campaigns, not in the checked-in files.

**Two model builders, one `Arm` dataclass** (`cfg, model, xml, structural_masses`). `model.build_arm(c)` dispatches on `c['model_type']`:
- Default: generates MJCF from scratch: analytical hollow-tube + bracket inertia (`model.structure`), motor housings as cylinders on the parent side of each joint, tool/payload boxes on J7. Axes and origin rotations come from the pinned OpenArm v1 URDF in `references/`.
- `urdf`: `urdf.py` parses the URDF, resolves mesh paths (relative, absolute, `file://`, `package://` via package roots / `ROS_PACKAGE_PATH`), and compiles through MuJoCo's native URDF compiler. `prepare_urdf_config` generates the `.sim.yaml` beside the URDF on first launch; `build_urdf_arm` refuses to build if joint names/types/limits drift from the saved YAML or peak torque exceeds URDF effort. Only revolute/continuous/fixed joints are supported.

Downstream code is model-agnostic and adapts to any joint count.

**MuJoCo owns all physics.** `simulation.py` only wraps `mj_step`: joint PD + velocity feedforward + optional gravity feedforward (or, with `trajectory_feedforward`, inverse-dynamics feedforward), always *inside* each actuator's force clamp. `torque_limit` applies the derating (0.8 on the checked-in arms) and optional speed envelope. `step` records everything pre-integration, so pose, torque and reactions share one timestamp. `Sweep` runs approach → settle → quintic A↔B move → hold; `run_headless` is the CLI path.

**Viewer.** `experiment.Experiment` holds all window-independent state/actions (load/reload, goto, sweeps, saving, FEA recording, campaign replay) so buttons and shortcuts share logic. `viewer.py` is one GLFW window with MuJoCo rendering + an ImGui (`imgui_bundle`) panel; `torque_charts.py` draws per-joint charts. `Experiment.load` builds the new arm before swapping, so invalid YAML leaves the running sim intact. `reporting.save_result` writes each sweep's timestamped folder (CSV, YAML snapshot, MJCF, PNG, `report.html`).

**Load recording (`fea.py`).** `FEARecording` samples every physics step via `mj_rnePostConstraint` and reports loads per structural **part**: bodies matching the `fea.exclude_bodies` globs (actuators, `rs*` on the CAD arms; default none) get no rows, and structural bodies joined without a joint merge into one part in its root body's frame (metadata `parts` has combined mass/COM/inertia). `rec.interfaces` is the row list: per part its proximal mate followed by its distal mates. Exports `fea-*` directories (peaks, load cases, optional full history, `model.mjb`, metadata, schema 2). `fea_cases.py` turns any such export into per-part setups and reports force and moment balance residuals.

**Load generators**, all producing the same export format:
- `campaign.py`: random A → B sweeps, optional (1+1)-ES refinement of every row's force/moment magnitude, in spawn-based workers. `--aggressive` instead does `duration_search` (geometric bisection) per random pose pair and direction, with a ≤0.5 ms timestep and the stricter `motor_loads_v2` acceptance, then reruns record setters at 1/2 and 1/4 timestep (`timestep-validation.json`). `CampaignRecording` subclasses `FEARecording` to export the extrema. `collision.CollisionChecker` (recompiled contact-enabled copy; convex hulls) rejects self-colliding planned paths and stops sweeps that collide.
- `restrained.py`: 6D tip-to-world weld, direct motor torques in {−peak, 0, +peak}^n ramped from rest, no PD or gravity compensation; internal arm motion allowed.
- `bounds.py`: static search (worst gravity pose; tip blocked with motors at peak torque; with a moment cap, tip clamped). Exports into `bounds-*/gravity|blocked|clamped/`. Blocked cases are `(joint, k, sign)`, clamped ones `('clamped', pf..., pm...)`; `Statics.contact` returns `(body, point, force, moment)` for either.

**Run store (`analysis.py`).** `campaign` and `restrained` write `campaign-*/` or `restrained-*/` run folders through `RunStore`: atomic `run.json`, `trials.sqlite`, `trials/<id>/` (`distribution.npz` with every physics sample in shared signed log bins, `trace.npz` preview), model/config snapshots, and the global `fea-*` export inside the run folder. Storage is compact by default (`compact_v1`: ≤120 preview frames per accepted trial, exact snapshots only for current global extrema); `--full-history` keeps everything. `trial_recording.py` re-simulates one trial into `trials/<id>/detail/` and publishes it only if acceptance, counts and statistics match the original. A `CANCEL` file in the run folder (or its parent) stops a run after active trials. `Peaks` lives here and keeps only the samples behind current extrema.

**Load Lab (`dashboard.py` + `assets/`).** **Runner** tab launches jobs; **Viewer** tab browses any run under the results root (discovery stops descending at a run folder). A launch is validated by `dashboard_jobs.prepare` (the source YAML is never modified), snapshotted to `job-*/` (`request.json`, `config.yaml`, `job.json`, `worker.log`) and executed as a detached `python -m arm_design.dashboard_jobs` process that calls the normal CLI; closing the browser or server does not stop it. One active job per mode (free motion / restrained). Helpers: `load_histograms.py` (signed bins, tail percentiles, SVG previews), `fea_view.py` (part mesh, mates, axes), `trial_playback.py` (forward kinematics over recorded poses, never a new simulation), `live_preview.py` (one worker publishes sampled poses while a browser renews a 3 s lease; it must never affect physics or fail a trial). JS is plain assets plus a vendored Three.js 0.170.0; no build step. `00_preview_utils.js` must sort first: Dash loads assets alphabetically.

**Test queue (`queue_runner.py`, `dashboard_queue.py`).** `plan` validates checked items into CLI steps; `start` launches one detached `python -m arm_design.queue_runner <queue-folder>` that runs each step as a subprocess with its own `config.yaml`, rewrites `queue.json` atomically, and calls `summarize` after every step (worst loads, seed spread, `fea-inputs/`). `CANCEL`/`PAUSE` files stop it; `queue-rates.json` holds the benchmark and learned rates. Only the CLI produces results; never compute loads in the queue.

`watch.py` is the older matplotlib convergence plot; `campaign.load_results` + `Experiment.replay_sweep` power the viewer's **Campaign replay** panel and the `replay` command.

## Gotchas

- **Linux rendering:** the imgui-bundle Linux wheel vendors a private libglvnd, so its native renderer can't see GLFW's context. `viewer.py` uses the PyOpenGL `ProgrammablePipelineRenderer` on Linux and forces PyOpenGL to GLX (it picks EGL under WSLg). Don't revert this.
- `robots/**/*.sim.yaml` is gitignored except the curated `moving_asm`, `right_arm_assembly` and `dynamics_sim_arm` ones (rated vs. peak torque, speed envelope, gains): don't regenerate them. A new robot's curated file needs its own `.gitignore` exception.
- Link masses in `configs/arm.yaml` intentionally match `part_1..part_7` in `moving_asm.urdf`; `tests/test_mujoco.py` asserts this.
- Actuator gear is 1: torques are output-shaft Nm; RobStride gear ratios are metadata only.
- `.sim.yaml` files store `urdf.path` relative to themselves (`config.portable`) and keep their leading comments (`config.header`) on every save, so committed settings work from any checkout. `load_config` resolves the path back to absolute in memory.
- Contacts are off by default (`simulation.contacts`) so coarse geometry doesn't support the arm.
- The CAD arm's meshes import as visual-only (`contype 0`), so `simulation.contacts: true` does not make its links collide. MuJoCo builds body collision filters and mesh convex hulls at compile time: to check collisions, recompile with `contype`/`conaffinity` set (as `CollisionChecker` does); editing `model.geom_contype` afterwards silently does nothing.
- `moving_asm` joint limits are the real hardware limits: J1/J3/J5 continuous, J2 −20..200°, J4 ±135°, J6 −90..50°, J7 −56..70°. J7's hardware limit is −70°, but `part_7`'s hull overlaps `part_5` below −56° with the other joints at zero. About 4% of random poses still self-collide (wrist combinations, `part_7` folding back to the shoulder); only the collision check removes those. `right_arm_assembly` is the same chain with a different tool: J7 ±70° (axis reversed), tracking tolerance 5° instead of 2°, `base` is a terminal part with a single mate.
- Don't re-export `moving_asm.urdf` from Onshape just to change limits: it carries hand edits (RS00/RS06 catalog housing masses with scaled inertia, effort = RobStride peak torque, velocity = no-load speed). `right_arm_assembly.urdf` likewise (its export had no `<limit>`s, and the tool links carry measured masses). Edit the `<joint>`/`<limit>` elements directly and update the matching `type`/`limits_deg` in the robot's `.sim.yaml`, or `build_urdf_arm` rejects the mismatch. Continuous joints have no physical stop in MuJoCo; revolute ones do.
- Campaign counts only sweeps that track within `tracking_tolerance_deg`: untrackable commands end in joint-stop impacts whose loads depend on MuJoCo constraint softness (a 0.22 s sweep once gave a bogus 1650 N). Keep that default. Aggressive runs additionally require the tracking tolerance at every sweep/hold sample, a complete sweep and hold, and final joint speeds ≤20°/s; they reject `--include-failed`, `--allow-self-collision` and `--iterations` > 0, and starting sag is only diagnostic.
- The uncapped `bounds` blocked case has no ceiling: near aligned poses (toggle-clamp geometry) small torques resist unbounded forces, so results grow with sampling. Always pass `--max-contact-force-n`; blocked loads scale almost linearly with it (≈ cap + weight in force, ≈ cap × reach in moment).
- `bounds` clamped needs both caps (`--max-contact-moment-nm` without `--max-contact-force-n` is an error) and its results scale with both; take the moment cap from the wrist motors or the gripper's slip limit, not a guess. It mainly raises loads in the distal links. It is at the tip only and uses its own RNG stream, so enabling it leaves gravity/blocked results for a seed unchanged; without the flag outputs match the two-case run.
- Design roles: `bounds` blocked governs static strength; `bounds` gravity is the everyday baseline and the fatigue mean load; the campaign shows dynamics don't govern (CAD arm: about 1.5–3× gravity, below a 100 N tip contact) and gives a conservative fatigue range and motor/speed limits. That comparison predates the aggressive mode and payload runs. Don't chase campaign convergence; rerun it when payload, geometry, gains or motors change. No gripper mechanism is modelled on either CAD arm; the dashboard can add an `analysis_payload` (its form defaults to **5 kg**; 0 keeps the modelled mass). CLI runs take `--payload kg` (Runner's default box); anything else needs `analysis_payload` in a config copy. Light and heavy payloads govern different parts, so run both.
- FEA rows are parts, not bodies: `interface['body_id']` is the part's root body, every row has `kind: mate` and an `other_body_id`, and per-sample kinematic arrays (`com_acceleration_world`, ...) are still per MuJoCo body (index `body_id-1`). `fea-cases --part` takes part names only (`base`, not `screwon`/`drumstick`; never an `rs*`). Older saved aggregate rows are hidden by the dashboard, replay, live plots and FEA setup export; preserve original row indices when reading their arrays.
- A new URDF import gets `fea: {exclude_bodies: []}`; set the actuator globs by hand. `moving_asm` and `right_arm_assembly` use `['rs*']`; `dynamics_sim_arm` must list every motor except `rs00_2` (J6), which merges into `l5`.
- In exports, `peaks.csv` `min_sample`/`max_sample` refer to each sample's `sample_index` field, not its position in `load_cases.json` (tied samples may be unreferenced). Look samples up by `sample_index`.
- The full test suite takes about 6 minutes (campaign/dashboard tests run real sweeps and workers). A stray `MUJOCO_LOG.TXT` (instability warnings from deliberately violent test sweeps) can appear in the repo root; delete it, never commit it.
- Use `.venv/bin/python -m pytest` rather than `.venv/bin/pytest`: venv scripts hard-code an absolute path and break if the repo folder moves or is renamed. Rerunning `setup.sh` refreshes the `arm-sim` script and editable path, not every script; delete `.venv` first for a clean fix.
- `replay` and the replay panel need the inner `fea-*` folder (it holds `campaign.json`), not the `campaign-*` run folder; `fea-cases` accepts either. `run.json` stores `export` as an absolute path, so moving a run breaks `fea-cases` on the run folder: point it at the inner `fea-*`.
- `replay_sweep` reruns from rest at A at time zero, so it reproduces recorded samples exactly and can pause on the peak's step. It uses `Sweep`'s default start-tracking check, which aggressive campaigns disable: an aggressive trial with large starting sag can be counted yet refuse to replay. The dashboard's playback has no such limit.
- The restrained clamp's movable world site must have `site_sameframe=0` (`set_pose`); otherwise the compiler's identity shortcut ignores its runtime position. Don't freeze qpos to mimic a restrained equilibrium. The weld is numerical compliance, not gripper stiffness or a slip limit; rejection reasons are clamp drift, collision, joint-limit contact and instability.
- **Linkages (`linkage.py`, `linkages:` in YAML).** The hinge is driven directly; `FEARecording` replaces its two rows with bearing rows and adds motor/rocker-pin/crank/lever rows (`interface['linkage']`, `role`). Crank and lever are `massless` parts whose loads are in their true moving frames (NOT `body_rotation_world`); their weight stays in the driving part, whose `visual_bodies` omits them. The model's zero pose must be the closed linkage. `place()` must run after `mj_kinematics`/`mj_forward` wherever geoms are drawn or collided; the browser gets two extra display-only hinges (`model_geometry` + `display_pose`). `l6` lists its motor mate first (`support`) because bearings alone cannot hold it. Pin markers give the pin axis only; the script puts pins at lever mid-thickness.
- Never hand-edit `dynamics_sim_arm.urdf`; change the script (or Onshape) and re-run on the raw export. `tests/test_linkage.py` requires the checked-in URDF/YAML to equal the script's output.
- `analysis.ordered_map` feeds all worker pools: results stay in job order (trial ids are reproducible for a seed and any worker count) without batch barriers. Don't go back to `pool.map` over batches; it left about a third of the cores idle.
- `collision.decompose_bodies` affects only `CollisionChecker`. Without it the filled `l5` fork hull falsely blocks J6 above +22 deg.
- The dashboard FEA table shows one table per loaded mate (all but the first); slider ids use `metric|interface` after the first.
- `dynamics_sim_arm` restrained trials are mostly rejected (about 10% accepted at 0.5 s in a 60-trial check): the narrow J6 range hits its limit. Shorten `--trial-duration` and read rejection counts.
- Preserve complete external moments in `fea_cases`, including pure moments and merged-member resultant moment arms; tests verify Newton–Euler balance.
- Keep the random and refinement stages separate in statistics: `campaign` needs at least one random sample, and dashboard campaigns run no refinement or finer-timestep reruns. Histogram frequencies are test coverage, never a duty cycle; independent extrema are not one load case.
- Changing `analysis.edges()` or the `distribution` fields breaks aggregation of saved runs and regeneration checks. Saved `motor_loads_v1` runs must keep their own acceptance policy, and runs without `run.json` (legacy exports) must still open.
- Regeneration needs the run's original MuJoCo version (`mujoco==3.16.0` is pinned) and compatible controller code: changing `simulation.control` or `run_sweep` can make old trials non-regenerable.
- Modeling assumptions are documented in `docs/MODEL.md`; data provenance (OpenArm pinned revision, RobStride specs) in `docs/SOURCES.md`. Update them when changing model assumptions or motor data.

## Style

Code is deliberately compact: short names, dense one-liners in places, few comments (module docstrings state each file's responsibility). Match that rather than expanding it.
