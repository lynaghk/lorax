(ns lorax.svm
  "Predict minst digits with svm as a baseline"
  (:require [lorax.mnist :refer [test-images test-labels]])
  (:import (libsvm svm svm_parameter svm_node svm_problem)
           (org.jblas DoubleMatrix ranges.IntervalRange)))


;;Some of these fns taken from clj-svm
(def kernel-types
  {:linear svm_parameter/LINEAR
   :poly svm_parameter/POLY
   :pre-computed svm_parameter/PRECOMPUTED
   :rbf svm_parameter/RBF
   :sigmoid svm_parameter/SIGMOID})

(def svm-types
  {:c-svc svm_parameter/C_SVC
   :epsilon-svr svm_parameter/EPSILON_SVR
   :nu-svc svm_parameter/NU_SVC
   :nu-svr svm_parameter/NU_SVR
   :one-class svm_parameter/ONE_CLASS})

(def default-params
  {:C 1
   :cache-size 100
   :coef0 0
   :degree 3
   :eps 1e-3
   :gamma 0
   :kernel-type (:rbf kernel-types)
   :nr-weight 0
   :nu 0.5
   :p 0.1
   :probability 0
   :shrinking 1
   :svm-type (:c-svc svm-types)
   :weight (double-array 0)
   :weight-label (int-array 0)})

(defn params
  ([] (params {}))
  ([options]
     (let [params (svm_parameter.)]
       (doseq [[key val] (merge default-params options)]
         (clojure.lang.Reflector/setInstanceField params (.replaceAll (name key) "-" "_") val))
       params)))

(defn svm-node [idx v]
  (let [n (svm_node.)]
    (set! (.-index n) (int idx))
    (set! (.-value n) (double v))
    n))

(defn instance [^DoubleMatrix x]
  (->> (.-data x)
       ;;TODO: reducers instead of map+remove
       (map-indexed (fn [idx v]
                      (svm-node idx v)))
       ;;libsvm accepts sparse vectors, so we don't explicit zero values.
       (remove #(zero? (.-value %)))
       into-array))

(defn problem [^DoubleMatrix xs ^DoubleMatrix ys]
  (assert (= (.rows xs) (.-length ys))
          "Number of instances (rows of xs) should be the same as number of labels")

  (let [problem (svm_problem.)]
    (set! (.-l problem) (.-length ys))
    (set! (.-x problem) (into-array (map instance (.rowsAsList xs))))
    (set! (.-y problem) (.-data ys))
    problem))


(defn train
  [problem params]
  (svm/svm_check_parameter problem params)
  (svm/svm_train problem params))


(comment
  (def xs test-images)
  (def ys test-labels)

  (time
   (def prob
     (problem xs ys)))

  (def n 1000)

  (time
   (def prob
     (problem (.getRows xs (IntervalRange. 0 n))
              (.getRows ys (IntervalRange. 0 n)))))
  (time
   (def model
     (train prob (params
                  ;;Gamma should be 1 / number of features
                  {:gamma (/ 1 (* 28 28))}))))

  (def res
    (for [idx (range n)]
      (= (.get ys idx 0)
         (svm/svm_predict model (instance (.getRow xs idx))))))
  
  (println (format "Training accuracy: %.2f" (float (/ (count (filter identity res)) n))))

  ;;print out to double check with libsvm text files
  (doseq [[idx v] (map-indexed vector (.-data (.getRow xs 0)))]
    (when-not (zero? v)
      (println idx ":" v)))
  )