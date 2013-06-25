(ns lorax.sgd

  (:import (org.jblas DoubleMatrix)))

(defn sgd
  "Stochastic gradient descent with L2 regularization.
X matrix whose rows are training samples
Y vector with class label for x's training samples
grad-loss fn(w, x, y)
"
  ([^DoubleMatrix X ^DoubleMatrix Y grad-loss]
     (let [n (.rows X)
           w (DoubleMatrix/zeros n)]
       (sgd X Y w grad-loss)))
  ([^DoubleMatrix X ^DoubleMatrix Y ^DoubleMatrix w grad-loss]

     (let [n (.rows X)
           epochs 50
           λ 1e-3 ;;regularization parameter
           factor (Math/sqrt (/ 2 λ))]

       (reduce (fn [^DoubleMatrix w [t x y]]
                 (let [new-w (-> w
                                 ;;Not sure if this is part of the regularization or just some standard decrease-the-learning-rate shit
                                 (.mul (- 1.0 (/ t)))
                                 (.sub (.div ^DoubleMatrix (grad-loss w x y)
                                             (* λ t))))]

                   ;;renormalize?
                   (.mul new-w (min 1.0 (/ factor (.norm2 new-w))))))

               ;;initial w
               (DoubleMatrix/zeros (.columns X))

               (->> (interleave (cycle (.rowsAsList X))
                                (map #(.get ^DoubleMatrix % 0 0) (cycle (.rowsAsList Y))))
                    (partition 2)
                    (take (* epochs n))
                    ;;shuffle
                    (map-indexed (fn [t [x y]] [(inc t) x y])))))))

(defn grad-hinge-loss
  "Gradient of the hinge loss for a linear model with weights w, instance x, and binary label y."
  [^DoubleMatrix w ^DoubleMatrix x ^double y]
  (if (> 1 (* y (.dot w x)))
    (.mul x (- y))
    (DoubleMatrix. (double-array [0.0]))))