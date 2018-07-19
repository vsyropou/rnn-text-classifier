import abc
import tensorflow as tf
import numpy as np 

class RnnAbsClassifierWrapper(abc.ABC):
    @abc.abstractmethod
    def __init__(**kwargs):
        pass
    
    @abc.abstractmethod
    def build(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def best_epoch_stats(self):
        pass
   
    @abc.abstractmethod
    def stats(self):
        pass

    @property
    @abc.abstractmethod
    def best_epoch(self):
        pass

    
class RnnBaseClassifierWrapper(RnnAbsClassifierWrapper):
    def __init__(self, **kwargs):

        # invoke class proxies
        class_names = filter(lambda itm: itm[0].endswith('Proxy'), kwargs.items())
        self._invoke_class_proxies(class_names, kwargs)

        # invoke object proxies
        object_names = filter(lambda itm: itm[0].endswith('Conf'), kwargs.items())
        self._invoke_object_proxies(object_names, kwargs)

        # parse arguments
        base_class_args = ['sequenceLength','numberOfLabels','rnnCellProxy','rnnCellConf',
                           'regularizationMethodProxy','regularizationMethodConf','rnnBuilderProxy',
                           'rnnBuilderConf','outputLayerProxy','outputLayerConf', 'lossFunctionProxy',
                           'lossFunctionConf','minimizerProxy','minimizerConf','minimizeMethodConf',
                           'sessionConf']
        try:
          for arg_name in base_class_args:
              self.__dict__['_%s'%arg_name] = kwargs[arg_name]
        except KeyError as exc:
            raise RuntimeError('Cannot find required kwarg: %s.'%exc.args[0]) from exc    

        # output stats container
        self._stats = {}


    def _invoke_class_proxies(self, class_names, kwargs):

        for arg_nam, prx_nam in dict(class_names).items():
            try:
                kwargs[arg_nam] = eval(prx_nam)
            except Exception as exc:
                msg = 'Cannot invoke proxy %s for argument %s. %s'%(prx_nam,arg_nam,exc.args)
                raise RuntimeError (msg) from exc


    def _invoke_object_proxies(self, object_names, kwargs):
        
        for arg_nam, cnf_dict in dict(object_names).items():
            for cnf_arg_nam, cnf_arg_prx in cnf_dict.items():
                if type(cnf_arg_prx) == str and cnf_arg_prx.startswith('tf.'):
                    try:
                        kwargs[arg_nam][cnf_arg_nam] = eval(cnf_arg_prx)
                    except Exception as exc:
                        msg = 'Cannot invoke proxy %s for object %s. %s'%(cnf_arg_prx,cnf_arg_nam,exc.args)
                        raise RuntimeError (msg) from exc


    def _declare_placeholders(self):
        self._x = tf.placeholder(tf.int32, [None,self._sequenceLength], name='nn_input')
        self._y = tf.placeholder(tf.int32, [None], name='labels')  

        self._pred = tf.placeholder(tf.int32, [None], name = 'prediction') 
        self._otpt = tf.placeholder(tf.int32, [None], name = 'true_values')
        self._cvec = tf.placeholder(tf.int32, [None], name = 'ith_class_vector')

    
    @property
    def stats(self):
        return self._stats


    def train(self,**kwargs):
        print('\nStart trainning.')
        try: # parse args

            x_train = kwargs.get('x_train')
            y_train = kwargs.get('y_train')
            x_test  = kwargs.get('x_test')
            y_test  = kwargs.get('y_test')

            num_epochs = kwargs.get('numEpochs')
            batch_size = kwargs.get('batchSize')

        except KeyError as exc:
            raise RuntimeError('Cannot find required kwarg: %s.'%exc.args[0]) from exc                    

        cast = lambda it: [float(i) for i in it] 

        # train
        init = tf.global_variables_initializer()

        with tf.Session(**self._sessionConf) as sn:
            init.run()

            for epoch in range(1,num_epochs+1):
                info = {}

                num_batches = int(len(x_train) // batch_size) + 1

                for i in range(num_batches):
                    min_ix = i * batch_size
                    max_ix = np.min([len(x_train),((i+1)*batch_size)])

                    x_train_batch = x_train[min_ix:max_ix]
                    y_train_batch = y_train[min_ix:max_ix]

                    sn.run(self._train_step,
                           feed_dict = {self._x: x_train_batch,
                                        self._y: y_train_batch})

                # train metrics
                train_metrics =  sn.run(self._metrics[:3],
                                        feed_dict = {self._x:x_train,
                                                     self._y:y_train})

                # test metrics
                test_metrics = sn.run(self._metrics,
                                      feed_dict ={self._x:x_test,
                                                  self._y:y_test})

                
                #  prepare input for f1 score
                info['f1score'] = {}
                for i in list(set(y_train)):

                    f1_metrics = sn.run(self._f1score_computations,
                                        feed_dict = {self._pred:test_metrics[3],
                                                     self._otpt:y_test,
                                                     self._cvec: np.array([i]*len(x_test))})

                    info['f1score']['lbl%s'%i] = dict(zip(['tp', 'fp', 'fn', 'cnt'],
                                                          f1_metrics))


                info['train'] = dict(zip(['loss','loss_std','acc'],
                                         cast(train_metrics)))


                info['test'] = dict(zip(['loss','loss_std','acc'],
                                        cast(test_metrics[:3])))

                self._stats[epoch] = info

                # print staff
                self._print_epoch_stats(epoch, info, last_epoch=epoch==num_epochs)
            
            # derived calss's method 'train()' can run tensorflow
            #  objects during the trainning session
            for nam, item in zip(kwargs['session_run_object_names'],
                                 sn.run(kwargs['session_run_objects'],
                                        feed_dict = kwargs['session_run_feed_dict'])
                                 ):
                self.__dict__[nam] = item



    def best_epoch(self):
        sts = [(k, s['test']['loss'], s['test']['acc']) for k, s in self.stats.items()]        
        return sorted(sts, key=lambda s: s[2], reverse=True)[0][0]


    def best_epoch_stats(self):

        best_stats = self.stats[self.best_epoch()]

        return { 'test':  best_stats['test'],
                 'train': best_stats['train'],
                 'f1scores':  self._f1scores(best_stats['f1score'], average=False)[1],
                 'avf1score': self._f1scores(best_stats['f1score'], average=True)[1],
                 'classimbalance': self._f1scores(best_stats['f1score'], average=False)[2]
                 }


    def _count_parameters(self):
        print('\nParameter Summary')
        total_parameters = 0

        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            print('Variable name: %s, shape: %s'%(variable.name, shape))
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(' parameters: %s'%variable_parameters)
            total_parameters += variable_parameters
        print('toal parameters: %s'%total_parameters)


    def _f1scores(self, inf, average=True):

        try: # compute
            counts, precision, recall = [], [], []
            for key in sorted(inf.keys()):
                vals = inf[key]
                precision += [float(vals['tp']) / (vals['tp'] + vals['fp'])]
                recall    += [float(vals['tp']) / (vals['tp'] + vals['fn'])]
                counts    += [int(vals['cnt'])]
        except Exception as exc:
            print('Cannot parse f1scores. %s: "%s"'%(exc.__class__.__name__,exc.args[0]))
            return None

        try: # average
            smw = float(np.sum(counts))
            weights = list(map(lambda c: float(c)/smw, counts))
            weighted = 'weighted'
        except ZeroDivisionError:
            weights = [1.] * max_lbl
            weighted = ''

        f1score = [2 * float(p*r) / (p+r) for p,r in zip(precision,recall)]

        if average:
            return (weighted, np.average(f1score, weights = weights))
        else:
            return ('', f1score, counts)


    def _print_epoch_stats(self, epoch, inf, last_epoch=False):

        try:
            train_loss = inf['train']['loss']
            test_loss  = inf['test']['loss']

            train_loss_std = inf['train']['loss_std']

            test_acc  = inf['test']['acc']
        except Exception as exc:
            print ('Cannot print training stats: %s: "%s"'%(exc.__class__.__name__,exc.args[0]))
            return None

        loss_diff_sgnf = (train_loss - test_loss) / float(train_loss_std)

        weighted, f1score = self._f1scores(inf['f1score'], average=True)

        # print some stuff
        msg  = 'Epoch: {}, Test Loss: {:+.16f}, Train Loss Diff: {:+.3f}[std], '.format(epoch,test_loss,loss_diff_sgnf)
        msg += 'Test Acc: {:.3f}, Mean {} F1 score: {:.3f}'.format(test_acc,weighted, f1score)
        print (msg)

        # print f1 scores per class
        if last_epoch:

            scores, counts = self._f1scores(inf['f1score'], average=False)[1:]
            for idx, (fs, cn) in enumerate(zip(scores,counts)):
                print(' Class {}: Count: {},  F1 score: {:.3f}'.format(idx,cn,fs))
