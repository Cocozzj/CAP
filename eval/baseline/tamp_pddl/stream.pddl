;; PDDLStream stream definitions for the TAMP baseline.
;;
;; Streams sample CONTINUOUS parameters (joint angles, target poses, force
;; magnitudes) that satisfy domain preconditions.  The PDDLStream solver
;; chains symbolic actions with these continuous samplers to produce a
;; full plan whose grounding has all numeric arguments concrete.
;;
;; Each (:stream ...) names an inputs/outputs signature; the actual sampler
;; is a Python function bound at solve-time in run_tamp.py.

(define (stream partnet-mobility-tamp-streams)

  ;; sample-target-pose ─ given current pose, sample a "pushed" target
  ;;   inputs:  (?obj - object  ?p1 - pose)
  ;;   outputs: (?p2 - pose)
  (:stream sample-target-pose
    :inputs (?obj ?p1)
    :domain (At ?obj ?p1)
    :outputs (?p2)
    :certified (FromTo ?p1 ?p2)
  )

  ;; sample-force ─ sample a force magnitude that can move ?obj
  ;;   inputs:  (?obj - object)
  ;;   outputs: (?f - force)
  (:stream sample-force
    :inputs (?obj)
    :domain (Articulated ?obj)
    :outputs (?f)
    :certified (CanApply ?f ?obj)
  )

  ;; sample-rotate-target ─ pick a target joint angle for "rotate" verb
  ;;   inputs:  (?j - joint  ?a-cur - angle)
  ;;   outputs: (?a-tgt - angle)
  (:stream sample-rotate-target
    :inputs (?j ?a-cur)
    :domain (and (Revolute ?j) (JointAngle ?j ?a-cur))
    :outputs (?a-tgt)
    :certified (and)
  )
)
