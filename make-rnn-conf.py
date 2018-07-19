import json
from hashlib import sha256
from itertools import product

learning_rate = [0.001, 0.0005]
embedding_size = [3, 5]
batch_size = [15, 20]

rnn_conf = lambda **kw: {

    "inputSequences": "./inputs/data/indexed-clean-tokenized-sentences.npy",
    "cleanSentences": "./inputs/data/clean-tokenized-sentences.npy",
    "inputLabels": "./inputs/data/indexed-clean-tokenized-sentences-labels.npy",
    "embeddingsMatrix": "./inputs/data/embeddings-glove-6B300d.npy",
    "vocabularySize": 1726,
    "reduceWithPCA": False,
    
    "sequenceLength": 3,
    "embeddingLength": kw['embeddingSize'],
    "batchSize": kw['batchSize'],
    "learningRate": kw['learningRate'],
    "numEpochs": 10,
    "trainingExamplesFrac": 0.7,
    
    "numpySeed": 22,
    "numberOfLabels": 6,
    "trainEmbeddings": True,
    
    "rnnCellProxy": "tf.contrib.rnn.BasicLSTMCell",
    "rnnCellConf": {},
    "regularizationMethodProxy": "tf.contrib.rnn.DropoutWrapper",
    "regularizationMethodConf": {
        "output_keep_prob": 0.75
    },
    "rnnBuilderProxy": "tf.nn.dynamic_rnn",
    "rnnBuilderConf": {
        "dtype": "tf.float32"
    },
    "outputLayerProxy": "tf.layers.dense",
    "outputLayerConf": {},
    "lossFunctionProxy": "tf.nn.sparse_softmax_cross_entropy_with_logits",
    "lossFunctionConf": {},
    "minimizerProxy": "tf.train.AdamOptimizer",
    "minimizerConf": {
        "learning_rate": 0.001
    },
    "minimizeMethodConf": {},
    "sessionConf": {}

}


for cnf_vals in list(product(learning_rate,
                         embedding_size,
                         batch_size)):

    configuration = rnn_conf(learningRate = cnf_vals[0],
                             embeddingSize = cnf_vals[1],
                             batchSize = cnf_vals[2])
    
    configuration['confId'] = sha256(json.dumps(configuration).encode('utf-8')).hexdigest()

    json.dump(configuration,
              open('rnn_conf_%s.json'%configuration['confId'],'w')
              )
