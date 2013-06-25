(ns lorax.t-sgd
  (:require [lorax.sgd :refer :all]
            [midje.sweet :refer :all])
  (:import (org.jblas DoubleMatrix)))

;;hinge-loss suitable test dataset
(let [n 100
      X (DoubleMatrix. (into-array (for [i (range n)]
                                     (double-array (if (> i (/ n 2))
                                                     [1 (rand)]
                                                     [0 (rand)])))))
      Y (DoubleMatrix.
         (into-array Double/TYPE (for [i (range n)]
                                   (if (> i (/ n 2))
                                     1 0))))]

  (fact
   (first (.-data (sgd X Y grad-hinge-loss))) => (roughly 1)))



;;Lets try to understand SGD by fitting a line to some points
(comment
  (let [n 100]
    (def xs
      (DoubleMatrix. (into-array (for [i (range n)]
                                   (double-array [1 (+ i (- (rand) 0.5))])))))
    (def ys
      (for [i (range n)]
        (+ (* 2 i) (- (rand) 0.5))))

    (let [alpha 1e-5
          loss (fn [^DoubleMatrix w ^DoubleMatrix x y]
                 (Math/pow (- (.dot w x) y) 2.0))
          grad-loss (fn [^DoubleMatrix w ^DoubleMatrix x y]
                      (.mul x (* 2 (- (.dot w x) y))))]

      (reduce
       (fn [w [x y]]
         (.sub w (.mul (grad-loss w x y) alpha)))
       (DoubleMatrix/zeros (.columns xs))
       (shuffle (take 10000 (cycle (partition 2 (interleave (.rowsAsList xs)
                                                            ys))))))))
  )
