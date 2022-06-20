#!/bin/bash

#source specific_tests/mnist/mnist_config.sh
#source specific_tests/random/random_config.sh
clidir=/home/anne/Documents/featurecloud/test-environment/cli
pydir=/home/anne/Documents/featurecloud/apps/fc-federated-k-means/test
basedir=/home/anne/Documents/featurecloud/test-environment/controller/data
datafile=data.tsv
dataname=data
outputfolder=$basedir/cluster_test
app_test=cluster_test
outdirs=()
split_dir=splits


echo $app_test
# loop over all configuration files
for configf in $(ls $basedir/$app_test/config)
do
  # start run
  # collect all directories in a string separated variable
  cd $basedir
  echo $(pwd)
  dirs=($(ls -d $app_test/$split_dir/$configf/*))
  dirs=$(printf "%s," "${dirs[@]}")
  # remove trailing comma
  dirs=$(echo $dirs | sed 's/,*$//g')
  cd $mydir

  # generate a random string to use as the output directory
  outputdir=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')
  outputdir=$app_test/$outputdir
  outdirs[${#outdirs[@]}]=$outputdir
  sudo mkdir -p $controller_data_test_result/$app_test


  #echo $dirs
  echo python $clidir/cli.py start --controller-host http://localhost:8000 --client-dirs $dirs --app-image k_means:latest --channel local --query-interval 1 \
    --download-results $outputdir --generic-dir $app_test/config/$configf
  python $clidir/cli.py start --controller-host http://localhost:8000 --client-dirs $dirs --app-image featurecloud.ai/k_means:latest --channel local --query-interval 3 \
    --download-results $outputdir --generic-dir $app_test/config/$configf

done


