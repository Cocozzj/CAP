# TAMP (PDDLStream) Baseline

Classical Task and Motion Planning via [PDDLStream](https://github.com/caelan/pddlstream)
— the standard PDDL-based planner with continuous motion sampling.

## Why this baseline

In the new 5-baseline matrix, this is the **classical / non-learning** corner.
Compared to our hand-coded ``tamp_rule`` (deprecated), PDDLStream gives:

- Real symbolic search over multi-step plans
- Combined task + motion planning (handles sub-goals + IK + collision)
- Stream-based continuous parameter sampling
- Recognized in the robotics literature

Reviewer angle: "Even with a sophisticated symbolic planner, the learned method
generalizes better on novel verbs, deformable objects, and long horizons."

## Setup

```bash
# 1. Install PDDLStream (Python wrapper around Fast-Downward)
pip install pddlstream

# 2. Install Fast-Downward (symbolic planner backend)
git clone https://github.com/aibasel/downward.git ~/downward
cd ~/downward && python build.py
export FD_PATH=~/downward/builds/release/bin/

# 3. (optional) Install pybullet for collision checking
pip install pybullet

# 4. Verify
python -c "from pddlstream.algorithms.meta import solve; print('OK')"
```

## Files (TODO — most still skeleton, see Implementation Notes)

```
tamp_pddl/
├── __init__.py
├── README.md                  this file
├── domain/
│   ├── domain.pddl            PDDL domain (predicates, actions)
│   └── stream.pddl            stream definitions (motion sampler)
├── motion.py                  motion primitive library
├── interface.py               wraps our manifest → PDDLStream world model
└── run_tamp.py                main entry: iterate split → write outputs
```

## Implementation status

- [x] Skeleton + README
- [ ] `domain.pddl` — define `(open ?obj)`, `(close ?obj)`, `(push ?obj ?from ?to)` actions
- [ ] `stream.pddl` — define motion sample streams + grasp generators
- [ ] `motion.py` — joint-trajectory primitives (open: sweep range, close: reverse, push: linear)
- [ ] `interface.py` — manifest entry → PDDLStream `(world, init, goal)` problem
- [ ] `run_tamp.py` — solve plan, execute primitives, write `pred_4dgs.npz`

## Comparison protocol

| Input | TAMP (PDDLStream) | Ours |
|---|---|---|
| Text | mapped to predicate goals | direct text input |
| Scene | URDF + GT articulation | init_gs.ply only |
| Plan | symbolic search over predicates | learned planner (CVAE+AR) |
| Motion | stream-sampled IK trajectories | learned executor + physics |

PDDLStream gets GT URDF/articulation (its UNFAIR advantage). We measure:

| Metric | Expected |
|---|---|
| IID success | competitive (simple tasks PDDL solves cleanly) |
| Unseen verb | drops (no rule for "push" if not in domain) |
| Long-horizon | competitive (search handles multi-step) |
| Deformable | **fails** (PDDL has no soft-body model) |
| Cross-material | **fails** (PDDL doesn't reason about materials) |

## Estimated work

PDDLStream setup + integration: **1-2 weeks** of focused engineering.

Primary risks:
- Fast-Downward platform compatibility (Linux works; macOS sometimes needs build tweaks)
- Mapping PartNet-Mobility URDFs → PDDL streams (needs Stream functions per joint type)
- IK solver convergence on articulated joints

If too time-consuming, fall back to the deprecated ``tamp_rule`` (hand-coded
linear interpolation between GT pose endpoints) — it's a much weaker baseline
but ships in 1 day.
