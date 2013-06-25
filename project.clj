(defproject com.keminglabs/lorax "0.1.0-SNAPSHOT"
  :description "Efficient training of deep neural networks"
  :license {:name "BSD" :url "http://www.opensource.org/licenses/BSD-3-Clause"}

  :dependencies [[org.clojure/clojure "1.5.1"]
                 [org.clojure/math.combinatorics "0.0.4"]
                 [org.jblas/jblas "1.2.3"]
                 [tw.edu.ntu.csie/libsvm "3.1"]]

  :profiles {:dev {:dependencies [[midje "1.5.1"]

                                  ;;this is not on Maven; need to
                                  ;;    git clone https://github.com/sinjax/JMatIO
                                  ;;and install
                                  [net.sourceforge.jmatio/jmatio "1.2"]]}})
