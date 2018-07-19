
import tensorflow as tf
import numpy as np

from base_classifiers import RnnBaseClassifierWrapper

class RnnTextClassifierWrapper(RnnBaseClassifierWrapper):

    def __init__(self, **kwargs):

        # base class initializer
        super().__init__(**kwargs)

        # parse arguments
        class_arguments = ['embeddingsMatrix','embeddingLength','reduceWithPCA',
                           'trainEmbeddings','vocabularySize']
        try:
            for arg_name in class_arguments:
                self.__dict__['_%s'%arg_name] = kwargs[arg_name]
        except KeyError as exc:
            raise RuntimeError('Cannot find required kwarg: %s.'%exc.args[0]) from exc    

        # parse embedings matrix
        if not self._trainEmbeddings:
            if type(self._embeddingsMatrix) == str:
                try: # locate
                    print('Using pre-trained embedings: %s'%self._embeddingsMatrix)
                    self._embeddingsMatrix = np.load(self._embeddingsMatrix)
                except Exception as exc:
                    msg  = 'Cannot open file %s: %s. '%(self._embeddingsMatrix, exc.args[0])
                    raise RuntimeError(msg) from exc
            assert type(self._embeddingsMatrix) in [np.array,np.ndarray],\
                '"embeddingsMatrix" argument type must be any of ["str","numpy.array","np.ndarray"]'
            
            # reduce with pca
            if self._reduceWithPCA:
                self._reduce_embeddings_matrix()

            self._embeddingLength = self._embeddingsMatrix.shape[1]

        # tensorflow placeholders
        self._declare_placeholders()
        
        # embeddings matrix container
        self._embedings = None


    @property
    def embeddings(self):
        return self._embeddings


    def _reduce_embeddings_matrix(self):
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=self._embeddingLength)

            init_shape = self._embeddingsMatrix.shape
            self._embeddingsMatrix = pca.fit_transform(self._embeddingsMatrix)
            
        except Exception as exc:
            cls = exc.__class__.__name__
            msg = 'Error while reducing embeddings matrix: %s(%s)'%(cls,exc.args[0])
            raise RuntimeError(msg) from exc

        print('Reduced embedings matrix from %s to %s'%(init_shape,self._embeddingsMatrix.shape))
        print('Explained variance ratio: %s'%(pca.explained_variance_ratio_.sum()))


    def build(self):

        # embedings tensor
        if self._trainEmbeddings:

            embs_mat_init_vals = tf.random_uniform([self._vocabularySize,self._embeddingLength], -1., 1.)
            self._embedings_tensor_object = tf.Variable(embs_mat_init_vals, name = 'embeddings_tensor')
        else: # pre-trained embeedings
            
            self._embedings_tensor_object = tf.constant(self._embeddingsMatrix)

        # build network
        rnn_input = tf.nn.embedding_lookup(self._embedings_tensor_object, self._x)

        rnn_cell = self._rnnCellProxy(self._embeddingLength, **self._rnnCellConf)
        rnn_cell = self._regularizationMethodProxy(cell=rnn_cell, **self._regularizationMethodConf)

        output, (encoding, final_state_info) = self._rnnBuilderProxy(rnn_cell, rnn_input, **self._rnnBuilderConf)

        logits = self._outputLayerProxy(encoding, self._numberOfLabels, **self._outputLayerConf)

        cross_entropy = self._lossFunctionProxy(logits=logits, labels=self._y, **self._lossFunctionConf)

        loss, loss_std = tf.nn.moments(cross_entropy,[0])

        optimizer = self._minimizerProxy(**self._minimizerConf)

        self._train_step = optimizer.minimize(loss, **self._minimizeMethodConf)

        # prepare some metrics
        predictions    = tf.argmax(logits,1)
        true_positives = tf.equal(predictions, tf.cast(self._y,tf.int64))
        accuracy       = tf.reduce_mean(tf.cast(true_positives,tf.float32))

        self._metrics = [loss, loss_std, accuracy, predictions]

        self._prepare_additional_metrics()


    def _prepare_additional_metrics(self):

        hits = tf.cast(tf.equal(self._pred, self._cvec), tf.int32)
        miss = tf.cast(tf.not_equal(self._pred, self._cvec), tf.int32)

        class_hit_ref = tf.cast(tf.equal(self._otpt, self._cvec), tf.int32)
        class_mis_ref = tf.cast(tf.not_equal(self._otpt, self._cvec), tf.int32)

        self._f1score_computations = [ tf.reduce_sum(hits),                            # true_positives
                                       tf.reduce_sum(tf.multiply(hits,class_mis_ref)), # false_positives
                                       tf.reduce_sum(tf.multiply(miss,class_hit_ref)), # false_negatives
                                       tf.reduce_sum(class_hit_ref),                   # class count
                                       ]

    def train(self, **kwargs):

        # collect embeddings during the trainning session
        kwargs['session_run_object_names']  = ['_embeddings']
        kwargs['session_run_objects']  = [self._embedings_tensor_object]
        kwargs['session_run_feed_dict'] = None
        
        super().train(**kwargs)

        # print 
        self._count_parameters()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='configure parsing.')
    parser.add_argument('conf_file',    type=str, help='specify configuration file.')
    parser.add_argument('--suffix',     type=str, default='',   help='specify suffix for output files.')
    parser.add_argument('--output-dir', type=str, default='./', help='specify output dir path')
    opts = parser.parse_args()

    # import stuff
    import json
    from pprint import pprint
    from scipy.stats import ttest_ind as ttest

    from sequence_manipulation import truncate_pad_shufflle_sequence_data
    
    # parse configuration
    cnf = json.load(open(opts.conf_file,'r'))

    np.random.seed(cnf['numpySeed'])

    # preprocessing
    train_args = truncate_pad_shufflle_sequence_data(**cnf)
    
    # build rnn
    rnn = RnnTextClassifierWrapper(**cnf)
    
    rnn.build()

    # train rnn
    rnn.train(**train_args, **cnf)

    # print stats
    print('\nBest epoch: %s'%rnn.best_epoch())
    print('Best epoch stats:\n')
    pprint(rnn.best_epoch_stats())

    # process output
    out_path = lambda x: '{root}/out_{what}_{id}_{suf}'.format(**{'root': opts.output_dir,
                                                                  'what': x,
                                                                  'id': cnf['confId'],
                                                                  'suf': opts.suffix}) 
    #  write out run stats
    out_results = rnn.best_epoch_stats()
    out_results.update({'conf':opts.conf_file})

    with open(out_path('results') + '.json', 'w') as fl:
        json.dump(out_results, fl)

    # write out embeddings matrix
    if cnf['trainEmbeddings']:
        np.save(out_path('embeddings'),rnn.embeddings)


    # compute sentence separetion metrics
    from sentences_statistics import SentenceClasteringMetrics

    distances = []
    for index_status, indices in zip(['normal',
                                      'random'],
                                     [range(len(train_args['y_test'])),
                                      np.random.permutation(np.arange(len(train_args['y_test'])))]
                                     ):

        args = {'input_sequences': train_args['x_train'],
                'input_labels': train_args['y_train'][indices],
                'embeddings_matrix': rnn.embeddings}

        metrics_instance = SentenceClasteringMetrics(**args)

        print('\nSentence cluster distances (%s)'%index_status)
        metrics_instance.process_sentences()

        metrics_instance.plot_sentences()

        distances += [metrics_instance.sentence_cluster_distances.mean(axis=1)]

    print('\nt-Test: (null = identical means)')
    print(ttest(distances[0],distances[1]))
