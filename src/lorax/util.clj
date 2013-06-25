(ns lorax.util
  (:require [clojure.set :as set])
  (:import org.jblas.DoubleMatrix))

;;;;;;;;;;;;;;;;;;;;;;;;
;;Numeric primitives

(defn ^"[D" da [s]
  (into-array Double/TYPE s))
(defn ^"[I" ia [s]
  (into-array Integer/TYPE s))

(defn roughly-zero?
  "What's a pico between friends?"
  [x]
  (< (Math/abs x) 1E-12))


;;;;;;;;;;;;;;;;;;;;;;;;
;;JBlas helpers

(defn concat-horizontally
  [matricies]
  ;;TODO: make this faster by preallocating a matrix and bashing in place...
  ;;(let [num-columns (reduce #(+ % (.columns %2)) matricies)])
  (reduce #(DoubleMatrix/concatHorizontally %1 %2)
          matricies))

(defn ^DoubleMatrix concat-columns
  [cols]
  (assert (every? #(= 1 (.columns ^DoubleMatrix %)) cols)
          "Only works with column vectors")
  (.transpose (DoubleMatrix. ^"[[D" (into-array (map #(.-data ^DoubleMatrix %) cols)))))

;;Avoid printing out huge matrices on the repl...
(defmethod print-method DoubleMatrix
  [mat w]
  (let [n (.rows mat)
        m (.columns mat)]
    (if (every? #(< % 20) [n m])
      ;;Printout matrix
      (dotimes [i n]
        (.write w (.toString (.getRow mat i)))
        (.write w "\n"))
      ;;Print dimensions
      (.write w (str "DoubleMatrix[" n "," m "]")))))


;;;;;;;;;;;;;;;;;;;;;;;;
;;Classification helpers

(defn accuracy
  "Returns accuracy of model fn evaluated on rows of X with labels Y."
  [model ^DoubleMatrix X ^DoubleMatrix Y]
  (double (/ (->> (map (fn [x y] (= y (model x)))
                       (.rowsAsList X) (.-data Y))
                  (filter identity)
                  count)
             (.rows Y))))

(defn cross-validate
  "Given train(X, Y) fn that returns IFn-implementing model, X, Y, and number of folds k, partitions rows of X and Y into k folds and returns seq of k accuracies."
  ([train X Y]
     (cross-validate train X Y 5))
  ([train X Y k]
     (let [m (.rows X)
           indexes (set (range m))]

       (assert (zero? (mod m k))
               "Number of folds k must evenly divide number of samples.")

       (doall (for [test-indexes (map set (partition (/ m k) indexes))]
                (let [train-indexes (set/difference indexes test-indexes)
                      X-test (.getRows X (ia test-indexes))
                      Y-test (.getRows Y (ia test-indexes))
                      model (train (.getRows X (ia train-indexes))
                                   (.getRows Y (ia train-indexes)))]
                  (- 1 (accuracy model X-test Y-test))))))))