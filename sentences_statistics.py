
from sklearn.decomposition import PCA
from matplotlib import pyplot
import numpy as np
import tensorflow as tf

class SentenceClasteringMetrics():
    def __init__(self, **kwargs):
        
        self._embs_matr = self._open_if(kwargs, 'embeddings_matrix')
        self._sequences = self._open_if(kwargs, 'input_sequences')
        self._seq_labls = self._open_if(kwargs, 'input_labels')

        self._mean_sentece_vectors = []
        self._sentence_cluster_distances = []
        self._sentence_embeding = {}

    @property
    def mean_sentece_vectors(self):
        return self._mean_sentece_vectors

    @property
    def sentence_cluster_distances(self):
        return self._sentence_cluster_distances
    
    def _open_if(self, kwargs, arg_key):

        try:
            itm = kwargs[arg_key]
            
            if type(itm) == str: # open file
                obj = np.load(itm)
            elif type(itm) == np.ndarray: # pass
                obj = itm
            else: # everything else
                msg  = 'Unsoported input type, %s, for argument %s. '%(type(itm),arg_key)
                msg += 'Need either "numpy.ndarray" or "str"'
                raise RuntimeError(msg)

        except Exception as exc:
            err_type = exc.__class__.__name__
            raise RuntimeError('Cannot open item "%s": %s(%s).'%(arg_key,err_type,exc.args[0])) from exc
        
        return obj
   
    def process_sentences(self):

        labels = self._seq_labls
        sentences = self._sequences
        embeddings_matrix = self._embs_matr

        # sort sentences by catgory
        sorted_sentences = { k : [] for k in set(labels) }
        for lbl, snt in zip(labels, sentences):
            sorted_sentences[lbl] += [snt]
    
        num_labels = len(set(labels))
    
        # compute distances
        input_sentences = tf.placeholder(tf.int32, [None,len(sentences[0])], name='sentences')
            
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            
            for cat, sents in sorted_sentences.items():
    
                # input
                in_dict = {input_sentences: sents}

                # initial embedeengs matrix
                word_embeddings = tf.nn.embedding_lookup(embeddings_matrix, input_sentences)

                # add words in a sentence
                sentence_embeddings = tf.reduce_sum(word_embeddings, axis=1)

                sentence_embeddings_normed = tf.nn.l2_normalize(sentence_embeddings, axis=1)
    
                # compute mean sentence vector
                sentence_mean = tf.reduce_mean(sentence_embeddings, axis=0)
                
                sentence_mean_normed = sess.run(tf.nn.l2_normalize(sentence_mean, axis=0),
                                                feed_dict=in_dict )

                self._mean_sentece_vectors += [sentence_mean_normed]

                self._sentence_embeding[cat] = [sents, sess.run(sentence_embeddings_normed, feed_dict=in_dict)] 
                
                # cosine distance of sentences from their mean
                #  (inner product of normalised vectors)
                cosine_distance = tf.tensordot(sentence_embeddings_normed,
                                               sentence_mean_normed,
                                               axes = [1,0]).eval(feed_dict=in_dict)

                # mean distance of sentences around their mean
                mean_cosine_distance = tf.reduce_mean(cosine_distance).eval(feed_dict=in_dict)

                print(' Average cosine distance of label "%s" is %s '%(cat,mean_cosine_distance))


            # compute intercluster distance aka seperation between word clsuters.
            sentence_cluster_vectors = tf.constant(np.asarray(self._mean_sentece_vectors))

            for i in range(num_labels):

                tmp = [tf.tensordot(sentence_cluster_vectors[i],
                                    sentence_cluster_vectors[j],
                                    axes=[0,0]).eval()
                       for j in range(num_labels) if i!=j]
                self._sentence_cluster_distances += [tmp]

            self._sentence_cluster_distances = np.array(self._sentence_cluster_distances)
            mean_intercluster_sentence_distance = np.mean(self._sentence_cluster_distances, axis=1)


            for lab, dist in zip(labels,mean_intercluster_sentence_distance):
                print(' Average distance between sentence cluster %s and all the rest sentence clusters is: %s'%(lab,dist))


    def _make_sentence_matrix(self):

        nlabs = len(set(self._seq_labls)) 

        self._sentence_matrix = np.concatenate([self._sentence_embeding[i][1] for i in range(nlabs)])
        init_shape = self._sentence_matrix.shape
        
        shuffled_indices = np.random.permutation(np.arange(self._sentence_matrix.shape[0]))

        self._pca = PCA(n_components=3)
        self._pca.fit(self._sentence_matrix[shuffled_indices])


    def plot_sentences(self, colors = []):
        self._make_sentence_matrix()

        nlabs = len(set(self._seq_labls))    
        if not colors:
            colors = ['black', 'red', 'blue', 'purple', 'cyan', 'orange']
            if len(colors) < nlabs:
                print('Provide list of colors via the "colors" argument.')
        embeddings = [self._sentence_embeding[i][1] for i in range(nlabs)]

        for idx, (sent_embd, col) in enumerate(zip(embeddings,colors[:nlabs])):

            reduced_sentence_matrix_batch = self._pca.transform(sent_embd)

            x = reduced_sentence_matrix_batch[:,0]
            y = reduced_sentence_matrix_batch[:,1]
            # z = reduced_sentence_matrix_batch[:,1]

            pyplot.scatter(x,y, c = col)

            if idx == 0:
                print ('Reduced sentences matrix from %s fetures to %s.'%(self._sentence_matrix.shape[1],
                                                                          reduced_sentence_matrix_batch.shape[1]))
                print ('Explaiend variance ration: %s'%self._pca.explained_variance_ratio_.sum())

        pyplot.ion()
        pyplot.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='configure parsing.')
    parser.add_argument('--conf-file',         type=str, default='',   help='specify configuration file.')
    parser.add_argument('--input-embeddings',  type=str, default='',   help='specify embeddings numpy file.')
    parser.add_argument('--input-sentences',   type=str, default='./', help='specify input sentences numpy file.')
    parser.add_argument('--input-labels',      type=str, default='./', help='specify input sentence labels numpy file.')
    parser.add_argument('--sequence-length',   type=str, default='./', help='specify input sequence length.')
    parser.add_argument('--random',            action='store_true',    help='randominze to asssess significance.')
    opts = parser.parse_args()


    from sequence_manipulation import truncate_pad_sequence_data
          
    if opts.conf_file:
        import json
        cnf = json.load(open(opts.conf_file,'r'))

        try: # find seed
            np.random.seed(cnf['numpy_seed'])
        except Exception:
            pass

        # truncate pad sentences
        data = truncate_pad_sequence_data(**cnf)

        # prepare args for sentence metrics
        args = {'input_sequences': data['x_train'],
                'input_labels': data['y_train'],
                'embeddings_matrix': cnf['embeddings_matrix']}
        
        
    else:

        # truncate pad sentences
        data = truncate_pad_sequence_data(**{'input_sentences': opts.input_sequences,
                                             'input_labels': opts.input_labels,
                                             'sequence_length': opts.sequence_length,
                                             'training_examples_frac': 1.}
                                          )

        # prepare args for sentence metrics
        args = {'input_sequences': data['x_train'],
                'input_labels': data['y_train'],
                'embeddings_matrix': cnf['embeddings_matrix']}

    # compute sentence metrics
    metrics_instance = SentenceClasteringMetrics(**args)

    metrics_instance.process_sentences()
    
