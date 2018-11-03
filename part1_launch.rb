#!/usr/bin/env ruby


#kmeans-sweepk-CH-ds1
#bin/cluster_util.py -s ds1 -a kmeans-sweepk-CH |& tee $@.log

["ds1","ds4"].each{|ds|
  ["kmeans","gmm"].each{|algo|
    ["#{algo}-sweepk","#{algo}-sweepk-CH"].each{|action|
      puts "./launcher.sh --log-file #{action}-#{ds}.log unbuffer python bin/part1.py -s #{ds} -a #{action}"
    }
  }
}
