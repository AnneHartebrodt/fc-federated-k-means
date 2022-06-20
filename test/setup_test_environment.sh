clidir=/home/anne/Documents/featurecloud/test-environment/cli
pydir=/home/anne/Documents/featurecloud/apps/fc-federated-k-means/test
basedir=/home/anne/Documents/featurecloud/test-environment/controller/data
datafile=localData.csv
dataname=localData
outputfolder=$basedir/cluster_test
confounderfile=localData.labels.csv

#basedir=$1
#clidir=$2
#pydir=$3
#outputfolder=$basedir/$4
#seed=$5
#sites=$6
#samples=$7


echo $basedir
echo $clidir
echo $pydir
echo $outputfolder
mkdir -p $outputfolder

features=10
batchcount=3
k=10
sites=4
seed=11



# generate the data
python $pydir/generate_test_data.py --directory $outputfolder/data --filename $dataname --centers 10 --points 50 --nfeatures $features --variances 1.0 2.0 --delim ';' --filesuffix '.csv'


dirname=splits
for i in 1.0 2.0;
do
python $pydir/compute_canonical_solution.py --directory $outputfolder --filename data/$i/$datafile --k_min 3 --k_max 4 --k_step 1 --seed 11 --header 0 --output $i --center True --variance True  --delim ';' #--log_transform False
python $pydir/generate_splits.py -d $outputfolder -o $dirname/$i -f data/$i/$datafile -n $sites -s $seed --header 0  --separator ';' --filename $datafile --rownames 0
python $pydir/generate_splits.py -d $outputfolder -o $dirname/$i -f data/$i/$confounderfile -n $sites -s $seed --header 0  --separator ';' --filename confoundingData.csv --rownames 0

python $pydir/generate_config_files.py -d $outputfolder -o config -f $datafile --count $i --center True --variance True --k_min 3 --k_max 10 --k_step 1 --delim ';' #--log_transform False
done

