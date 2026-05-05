;; PDDL domain for TAMP (PDDLStream) baseline on PartNet-Mobility.
;;
;; Models 7 atomic verbs from the manifest's task vocabulary:
;;     open, close, push, pull, rotate, fold, squeeze, pour
;; Compositional tasks (comp:close_open, comp:open_close, ...) are decomposed
;; into sequences of these atomic verbs by ``interface.py``.
;;
;; Numeric joint angles, target poses, and force magnitudes are sampled by
;; streams in ``stream.pddl``.

(define (domain partnet-mobility-tamp)
  (:requirements :strips :typing :equality :negative-preconditions)

  (:types
    object joint pose angle force - object_t
  )

  (:predicates
    ;; Object structure
    (Articulated     ?obj - object)
    (HasJoint        ?obj - object  ?j - joint)
    (Revolute        ?j   - joint)              ;; rotation joint
    (Prismatic       ?j   - joint)              ;; translation joint

    ;; Joint state
    (JointAngle      ?j - joint  ?a - angle)
    (JointMin        ?j - joint  ?a - angle)
    (JointMax        ?j - joint  ?a - angle)

    ;; Goal predicates per verb
    (Open            ?obj - object)
    (Closed          ?obj - object)
    (Pushed          ?obj - object)
    (Pulled          ?obj - object)
    (Rotated         ?obj - object)
    (Folded          ?obj - object)
    (Squeezed        ?obj - object)
    (Poured          ?obj - object)

    ;; Helpers (set by streams)
    (At              ?obj - object  ?p - pose)
    (FromTo          ?p1 - pose ?p2 - pose)
    (CanApply        ?f  - force ?obj - object)
  )

  ;; ────────────────────────────────────────────────────────────────────
  ;; (open ?obj) — sweep its primary joint to range_max
  ;; ────────────────────────────────────────────────────────────────────
  (:action open
    :parameters (?obj - object  ?j - joint  ?a-cur - angle  ?a-max - angle)
    :precondition (and
      (HasJoint ?obj ?j)
      (JointAngle ?j ?a-cur)
      (JointMax   ?j ?a-max)
    )
    :effect (and
      (not (JointAngle ?j ?a-cur))
      (JointAngle ?j ?a-max)
      (Open ?obj)
      (not (Closed ?obj))
    )
  )

  ;; ────────────────────────────────────────────────────────────────────
  ;; (close ?obj) — sweep its primary joint to range_min
  ;; ────────────────────────────────────────────────────────────────────
  (:action close
    :parameters (?obj - object  ?j - joint  ?a-cur - angle  ?a-min - angle)
    :precondition (and
      (HasJoint ?obj ?j)
      (JointAngle ?j ?a-cur)
      (JointMin   ?j ?a-min)
    )
    :effect (and
      (not (JointAngle ?j ?a-cur))
      (JointAngle ?j ?a-min)
      (Closed ?obj)
      (not (Open ?obj))
    )
  )

  ;; ────────────────────────────────────────────────────────────────────
  ;; (push ?obj ?from ?to) — translate object via external force
  ;; ────────────────────────────────────────────────────────────────────
  (:action push
    :parameters (?obj - object  ?p1 - pose  ?p2 - pose  ?f - force)
    :precondition (and
      (At        ?obj ?p1)
      (FromTo    ?p1  ?p2)
      (CanApply  ?f   ?obj)
    )
    :effect (and
      (not (At ?obj ?p1))
      (At      ?obj ?p2)
      (Pushed  ?obj)
    )
  )

  ;; ────────────────────────────────────────────────────────────────────
  ;; (pull ?obj ?from ?to) — translate object backward
  ;; ────────────────────────────────────────────────────────────────────
  (:action pull
    :parameters (?obj - object  ?p1 - pose  ?p2 - pose  ?f - force)
    :precondition (and
      (At        ?obj ?p1)
      (FromTo    ?p1  ?p2)
      (CanApply  ?f   ?obj)
    )
    :effect (and
      (not (At ?obj ?p1))
      (At      ?obj ?p2)
      (Pulled  ?obj)
    )
  )

  ;; ────────────────────────────────────────────────────────────────────
  ;; (rotate ?obj) — sweep revolute joint by a target angle
  ;; ────────────────────────────────────────────────────────────────────
  (:action rotate
    :parameters (?obj - object  ?j - joint  ?a-cur - angle  ?a-tgt - angle)
    :precondition (and
      (HasJoint ?obj ?j)
      (Revolute ?j)
      (JointAngle ?j ?a-cur)
    )
    :effect (and
      (not (JointAngle ?j ?a-cur))
      (JointAngle ?j ?a-tgt)
      (Rotated ?obj)
    )
  )

  ;; ────────────────────────────────────────────────────────────────────
  ;; Soft-body verbs (fold / squeeze / pour) — fail in basic TAMP since
  ;; PDDL doesn't model deformation; we keep the predicates so the
  ;; aggregator can mark them as "task_not_in_pddl_domain".
  ;; ────────────────────────────────────────────────────────────────────
  (:action fold
    :parameters (?obj - object)
    :precondition (Articulated ?obj)
    :effect (Folded ?obj)
  )

  (:action squeeze
    :parameters (?obj - object)
    :precondition (Articulated ?obj)
    :effect (Squeezed ?obj)
  )

  (:action pour
    :parameters (?obj - object)
    :precondition (Articulated ?obj)
    :effect (Poured ?obj)
  )

)
