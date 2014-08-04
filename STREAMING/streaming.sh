mapper="/home/hduser/STREAMING/wordcount_mapper.py"
reducer="/home/hduser/STREAMING/wordcount_reducer.py"
inputs="/user/hduser/gutenberg/*"
output="/user/hduser/gutenberg-output"$1
hadoopjar="/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.3.0.jar"
echo "hadoop jar $hadoopjar -file $mapper -mapper $mapper -file $reducer -reducer $reducer -input $inputs -output $output"

reducer="/home/hduser/STREAMING/minimal_reducer.py"

mapper="/home/hduser/STREAMING/basic_mapper.py"
combiner="/home/hduser/STREAMING/basic_combiner.py"
reducer="/home/hduser/STREAMING/basic_reducer.py"
inputs="/user/hduser/ngrams/n5gram_dataset.txt"
output="/user/hduser/ngrams-output"$1
hadoopjar="/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.3.0.jar"

echo "hrm `basename $output`"
echo "hadoop jar $hadoopjar -file $mapper -mapper $mapper -file $reducer -reducer $reducer -combiner $combiner -input $inputs -output $output"
echo "hls `basename $output`"
echo "h1 tail `basename $output`/part-00000 | more"
echo "h1 ls `basename $output`/*SUCCESS"
