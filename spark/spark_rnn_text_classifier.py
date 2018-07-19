
# This is a test job to run tensorflow on spark
#  The code runs but, I realized that it maybe does not make a lot of sence if you cannot use the paralleization of the cluster
#  Perhaps if you use the clsuter nodes as tensorflow devices or
#  maybe it is better to put tensorflow on a docker container and use docker-machine to setup a dedicted cluster
#  need to investigate further.


# run interactivey in a spark cluster as:
# spark-submit \
#   --master yarn \
#   --deploy-mode cluster \
#   --py-files wasb:///repos/dependencies.zip \
#   --files <json_conf_file_path>\<json_conf_file_name> \
#  wasb:///repos/spark_rnn_text_classifier.py \
#  <json_conf_file_name>


import argparse
parser = argparse.ArgumentParser(description='configure parsing.')
parser.add_argument('conf_file',    type=str, help='specify configuration file.')
parser.add_argument('--suffix',     type=str, default='',   help='specify suffix for output files.')
parser.add_argument('--output-dir', type=str, default='/data/output/', help='specify output dir path')
opts = parser.parse_args()

# import stuff
import pyspark
import json
import pandas as pd
import numpy as np
from pprint import pprint
from scipy.stats import ttest_ind as ttest

from pyspark.sql.types import StructField, StructType, IntegerType
from pyspark.sql.functions import monotonically_increasing_id
    
from rnn_text_classifier import RnnTextClassifierWrapper
from sequence_manipulation import truncate_pad_shufflle_sequence_data

# parse conf file
cnf = json.load(open(opts.conf_file,'r'))
np.random.seed(cnf['numpySeed'])

# spark session
sc = pyspark.SparkContext(appName=opts.conf_file.split('/')[-1] )

# # helping stuff
appId = sc.applicationId

output_root_dir = '/'.join([opts.output_dir,appId])

make_schema = lambda ln, nm: StructType([StructField('%s%s'%(nm,i),
                                                     IntegerType(),
                                                     nullable = True) for i in range(ln)])

make_dframe = lambda rdd, scm: rdd.toDF(schema=scm).withColumn('id',monotonically_increasing_id())

out_tmpl = lambda x: '{root}/out_{what}_{id}_{suf}'.format(**{'root': opts.output_dir,
                                                              'what': x,
                                                              'id': appId,
                                                              'suf': opts.suffix}) 

# start spark session
with pyspark.sql.session.SparkSession(sc) as spark:
    
    # read embeddings 
    embeddingsMatrix = spark.read.load(cnf['embeddingsMatrix'], format="csv", sep=",", inferSchema="true", header="false")

    # read input sequences
    inputSeqRdd    = sc.textFile(cnf['inputSequences']).map(lambda record: json.loads(record)['0'])
    inputSeqSchema = make_schema(len(inputSeqRdd.first()), 'tkn')
    inputSeqDframe = make_dframe(inputSeqRdd,inputSeqSchema)
    inputSeqDframe.createOrReplaceTempView('inputSequences')

    # read input labels
    inputLabelsRdd    = sc.textFile(cnf['inputLabels']).map(lambda x: [int(x)])
    inputLabelsSchema = make_schema(1,'label')
    inputLabelsDframe = make_dframe(inputLabelsRdd, inputLabelsSchema)
    inputLabelsDframe.createOrReplaceTempView('inputLabels')

    # preprocessing
    # TODO: Use spark-hive for all data manipulations in the next round of optimization, for now stick to pandas
    cnf['inputSequences'] = spark.sql('select * from inputSequences').toPandas().values
    cnf['inputLabels']    = spark.sql('select label0 from inputLabels').toPandas().values
    cnf['inputLabels']    = np.array(list(map(lambda x: x[0],cnf['inputLabels'])))
    train_args = truncate_pad_shufflle_sequence_data(**cnf)

    # build rnn
    rnn = RnnTextClassifierWrapper(**cnf)
    rnn.build()

    # # train rnn
    rnn.train(**train_args, **cnf)

    # print stats
    print('\nBest epoch: %s'%rnn.best_epoch())
    print('Best epoch stats:\n')
    pprint(rnn.best_epoch_stats())
    
    # write training stats
    sc.parallelize([rnn.stats]).saveAsTextFile(out_tmpl('train_stats'))
    sc.parallelize([rnn.best_epoch_stats()]).saveAsTextFile(out_tmpl('best_epoch_stats'))

    # write embeddings
    if cnf['trainEmbeddings']:
        sc.parallelize(pd.DataFrame(rnn.embeddings).to_json(orient='index')).saveAsTextFile('embeddings')

