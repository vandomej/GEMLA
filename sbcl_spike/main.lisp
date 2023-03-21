;; Define a type that contains a population size and a population cutoff
(defclass simulation-node () ((population-size :initarg :population-size :accessor population-size)
                              (population-cutoff :initarg :population-cutoff :accessor population-cutoff)
                              (population :initform () :accessor population)))

;; Define a method that initializes population-size number of children in a population each with a random value
(defmethod initialize-instance :after ((node simulation-node) &key)
  (setf (population node) (make-list (population-size node) :initial-element (random 100))))

(let ((node (make-instance 'simulation-node :population-size 100 :population-cutoff 10)))
  (print (population-size node))
  (population node))