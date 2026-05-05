;; PDDL domain for the TAMP (PDDLStream) baseline.
;;
;; Models PartNet-Mobility manipulation tasks as symbolic predicates.  The
;; continuous parameters (joint angles, target poses, contact forces) are
;; sampled by streams defined in stream.pddl.
;;
;; This is a STARTING POINT — extend per task-class as needed.

(define (domain partnet-mobility-tamp)
  (:requirements :strips :typing :equality)

  (:types
    object joint pose angle - object_t
    surface - object_t
  )

  (:predicates
    (Articulated     ?obj - object)               ; the object has joints
    (HasJoint        ?obj - object  ?j - joint)   ; joint belongs to object
    (JointAngle      ?j - joint     ?a - angle)   ; current joint angle
    (JointMin        ?j - joint     ?a - angle)   ; range_min
    (JointMax        ?j - joint     ?a - angle)   ; range_max
    (Open            ?obj - object)               ; goal predicate
    (Closed          ?obj - object)               ; goal predicate
    (Pushable        ?obj - object)
    (At              ?obj - object  ?p - pose)
  )

  ;; (open ?obj) — set joint to range_max
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

  ;; (close ?obj) — set joint to range_min
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

  ;; (push ?obj ?from ?to) — translate object from p1 to p2
  (:action push
    :parameters (?obj - object  ?p1 - pose  ?p2 - pose)
    :precondition (and
      (Pushable ?obj)
      (At       ?obj ?p1)
    )
    :effect (and
      (not (At ?obj ?p1))
      (At  ?obj ?p2)
    )
  )

)
