#!/bin/bash

#source specific_tests/mnist/mnist_config.sh
clidir=/home/anne/Documents/featurecloud/test-environment/cli
pydir=/home/anne/Documents/featurecloud/apps/fc-federated-k-means/test
basedir=/home/anne/Documents/featurecloud/test-environment/controller/data
datafile=data.tsv
dataname=data
outputfolder=$basedir/cluster_test
app_test=cluster_test

for od in $(ls $basedir/tests/$app_test/ )
do
    # collect all output files in a string separated variable
  cd $basedir/tests/$app_test/$od
  echo $(pwd)
  # get number of clients
  declare -i clients=$(ls *.zip| wc -l)

  for i in $(seq 0 $(($clients -1)))
  do

    sudo unzip -d client_$i $(ls | grep  client_$i)
  done

  cl=($(ls -l . | egrep '^d' | rev | cut -f1 -d' ' | rev))
  echo 'single mode'

  python $pydir/check_accuracy.py --baseline $test_report --federated $tests -o $od"_"$d"_test.tsv" -e $basedir/tests/$app_test/$od/$cl/config.yaml -i $basedir/tests/$app_test/$od/$cl/run_log.txt \
   --header 0 --rownames 0

  #fi
  cd ..
done

#od=($(ls $basedir/tests/$app_test/ ))
#tests=$(printf "$basedir/tests/$app_test/%s/$cl/log.txt " "${od[@]}")
#ids=$(printf "%s " "${od[@]}")
#
#
##python $pydir/runstats.py -d $test_report -o "run_summaries.tsv" -f $tests -i $ids
## generate report
#python $pydir/generate_report.py -d $test_report/test_results -r $test_report/report.md
#pandoc $test_report/report.md -f markdown -t html -o $test_report/report.html --css $pydir/templates/pandoc.css
