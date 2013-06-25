(ns lorax.shamir
  (:require [lorax.util :refer :all]
            [lorax.sgd :refer [sgd grad-hinge-loss]]
            [clojure.math.combinatorics :as combo])
  (:import (org.jblas DoubleMatrix Geometry Singular MatrixFunctions)))

(set! *unchecked-math* true)
(set! *warn-on-reflection* true)

(defn build-basis-1
  "Given matrix E, returns a weight matrix W such that E*W is orthonormal, corresponding to the largest width singular values.
   Corresponds to shamir's `orthonormalize` fn."
  ([^DoubleMatrix E] (build-basis-1 E (max (.rows E) (.columns E))))
  ([^DoubleMatrix E width]
     (let [;;matching variable names in paper's implementation
           [_ ^DoubleMatrix D ^DoubleMatrix W] (Singular/fullSVD E)
           ;;Trim
           ^DoubleMatrix W (.getColumns W (ia (range 0 (min width (.columns W)))))
           ^DoubleMatrix B (.mmul E W)
           sqrtm (Math/sqrt (.rows E))]

       (-> (.mul W sqrtm)
           (.div (-> (MatrixFunctions/pow B 2.0)
                     (.columnSums)
                     (MatrixFunctions/sqrt)
                     (.repmat (.rows W) 1)))))))

(defn build-basis-t
  [Fts width b ^DoubleMatrix V]

  (assert (zero? (mod width b)) "Batch size must evenly divide width.")

  (let [^DoubleMatrix F1 (first Fts)
        ^DoubleMatrix F (concat-horizontally Fts)
        Ft-1-col-start (reduce + (map #(.columns ^DoubleMatrix %) (butlast Fts)))
        Et-source-indexes (vec (combo/cartesian-product (range (.columns F1))
                                                        (range Ft-1-col-start
                                                               (.columns F))))

        ^DoubleMatrix Et (concat-columns (for [[i j] Et-source-indexes]
                                             (.mul (.getColumn F i)
                                                   (.getColumn F j))))

        sqrtm (Math/sqrt (.rows F))
        ;;OF = left singular values of previous basis F
        ;;TODO: find a nice way to save OF for the t+1 invocation of build-basis.
        ^DoubleMatrix OF (first (Singular/sparseSVD F))]

    (loop [Ft [] nodes []
           ^DoubleMatrix OFOFt (.mmul OF (.transpose OF))
           ^DoubleMatrix V (.sub V (.mmul OFOFt V))
           batches-completed 0]

      (if (= width (* batches-completed b))
        ;;We're done!
        [(concat-columns (for [[w i j] nodes]
                             (.mul (.mul (.getColumn F i) (.getColumn F j))
                                   w)))

         nodes]
        ;;else, run through another batch
        (let [^DoubleMatrix C (Geometry/normalizeColumns (.sub Et (.mmul OFOFt Et)))
              ^DoubleMatrix OV (first (Singular/sparseSVD V))
              top-indexes (->> (.mmul (.transpose OV) C)
                               (.columnsAsList)
                               (map-indexed vector)
                               (sort-by #(.norm2 ^DoubleMatrix (second %)))
                               (take-last b)
                               (map first))
              nodes-for-block (for [idx top-indexes]
                                  (let [weight (/ sqrtm (.norm2 (.getColumn Et idx)))
                                        [i j] (nth Et-source-indexes idx)]
                                    [weight i j]))
              block (.getColumns Et (ia top-indexes))
              ^DoubleMatrix OC (first (Singular/sparseSVD block))]

          (recur (conj Ft block)
                 (concat nodes nodes-for-block)

                 ;;New OF * OFt is just the old one with the new column summed on
                 (.add OFOFt (.mmul OC (.transpose OC)))
                 (.sub V (.mmul (.mmul OC (.transpose OC)) V))
                 (inc batches-completed)))))))

(defn predict
  [model ^DoubleMatrix x]
  ;;W1: The first layer weight matrix
  ;;nodes: vector of [w i j] triples where w is a weight, and i & j are indicies of previous nodes that should be multiplied together
  (let [{:keys [^DoubleMatrix W1 ^DoubleMatrix output-weights
                nodes]} model
                x1 (DoubleMatrix/concatHorizontally (DoubleMatrix/ones 1) x)
                ;;TODO: worry about getting the factors used to normalize second moments of F1?
                first-layer (mapv #(.dot ^DoubleMatrix % x1) (.columnsAsList W1))]

    ;;TODO: make this faster by preallocating Java Array or DoubleMatrix with length widths*layers
    (let [evaled-nodes (->> (reduce (fn [evaled-nodes [w i j]]
                                      (conj evaled-nodes
                                            (* w (nth evaled-nodes i) (nth evaled-nodes j))))
                                    first-layer
                                    nodes)
                            da
                            (DoubleMatrix. ))]

      (.dot output-weights evaled-nodes))))



(defn train
  [X Y {:keys [max-delta width b]
        :or {max-delta 5
             width 10
             b 1}}]

  (let [E1 (DoubleMatrix/concatHorizontally (DoubleMatrix/ones (.rows X)) X)
        W1 (build-basis-1 E1 width)
        F1 (.mmul E1 W1)]

    (loop [nodes []
           Fts [F1]
           delta 1]

      ;;Print training error for previous dimension
      (let [output-weights (sgd (concat-horizontally Fts) Y grad-hinge-loss)]
        (prn (- 1 (accuracy #(Math/signum (predict {:W1 W1 :nodes nodes :output-weights output-weights} %))
                            X Y))))

      (if (= delta max-delta)
        ;;we're done
        {:W1 W1
         :output-weights (sgd (concat-horizontally Fts) Y grad-hinge-loss)
         :nodes nodes}
        ;;else, bump into the next dimension
        (let [[Ft new-nodes] (build-basis-t Fts width b Y)]
          (recur (concat nodes new-nodes)
                 (conj Fts Ft)
                 (inc delta)))))))



;;;;;Run!
(comment
  (import 'com.jmatio.io.MatFileReader)
  ;;Paper sample data
  (let [d (.getContent (MatFileReader. "vendor/basis_learner/data.mat"))]
    (def X (DoubleMatrix. (.getArray (get d "X"))))
    (def Y (DoubleMatrix. (.getArray (get d "Y")))))

  (cross-validate (fn [X Y]
                    (let [model (train X Y {:max-delta 3 :width 20})]
                      (fn [y] (Math/signum (predict model y)))))
                  X Y 5)




  (require 'lorax.mnist)
  (def X (.getRows lorax.mnist/test-images (ia (range 1000))))
  (def Y (.getRows lorax.mnist/test-labels (ia (range 1000))))


  (def MaxDelta 5)
  (def width 10)
  (def b 1)

  (def E1 (DoubleMatrix/concatHorizontally (DoubleMatrix/ones (.rows X)) X))
  (def W1 (build-basis-1 E1 width))
  (def F1 (.mmul E1 W1))





  (let [[Ft _] (build-basis-t [F1] 10 b Y)]
    (.dot (.getColumn Ft 0)
          (.getColumn Ft 0)))



















  (cross-validate train X Y 5)


  ;;Share results with Matlab
  (import 'com.jmatio.io.MatFileWriter
          'com.jmatio.types.MLDouble)

  (defn matlab-save! [^DoubleMatrix X]
    (MatFileWriter. "grr"
                    [(MLDouble. "Z" (.-data X) (.rows X))]))




  (time
   (def Et
     (let [I (.columns F1)
           J (.columns Ft-1)
           res (DoubleMatrix. (.rows Ft-1) (* I J))]
       (dotimes [i I]
         (dotimes [j J]
           (.putColumn res (+ (* i I) j) (.mul (.getColumn F1 i) (.getColumn Ft-1 j)))))
       res)))

  (time
   (def Et
     (reduce #(DoubleMatrix/concatHorizontally %1 %2)
             (for [ft-1 (.columnsAsList Ft-1)
                   f1 (.columnsAsList F1)]
                 (.mul ft-1 f1)))))




  (.norm2 (.mul (.getColumn F1 0) (.getColumn F1 0)))


  (.transpose (DoubleMatrix. (into-array [(da [1 2 3 4]) (da [3 4 5 6])])))
  (.mul (DoubleMatrix. (da [1 2 3 4]))
        (DoubleMatrix. (da [5 6 7 8])))



  )
