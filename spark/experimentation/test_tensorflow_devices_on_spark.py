
# here I am trying to see how can I utilize the cluster nodes as tensorflow devices nodes.
# I am not sure if it is possible


from distributed_rnn_text_classifier import RnnTextClassifierWrapper
from sequence_manipulation import truncate_pad_shufflle_sequence_data

import tensorflow as tf

conf_file = '/mnt/c/Users/vsyro/WorkDir/bloom-classifier/conf-files/rnn_conf_e4dc92d40f9354aceed6c221647b7825ce1a9ffba7c5744a30bde197f8edaab6.json'

# cluster specs
# task_index = 0 # make this cmd arg
cluster = tf.train.ClusterSpec({"local": ["localhost:2222","localhost:2223"]})
# ssh sshuser@spark-datascience-ssh.azurehdinsight.net

# prep
import json
cnf = json.load(open(conf_file,'r'))
# cnf['minimizeMethodConf'] = {'global_step' : 'tf.train.get_or_create_global_step()'}
train_args = truncate_pad_shufflle_sequence_data(**cnf)



# # job1
# server1 = tf.train.Server(cluster, job_name="local", task_index=0)
# with tf.device('/job:local/task:0') as dev:
        
#     rnn = RnnTextClassifierWrapper(**cnf)
#     rnn.build()
# print ('preped job1')

# job2
server1 = tf.train.Server(cluster, job_name="local", task_index=0)
with tf.device('/job:local/task:0') as dev:
    
    var0 = tf.Variable(tf.random_uniform([5,5], -1., 1.),name='var0')
print ('preped job1')

# job2
server2 = tf.train.Server(cluster, job_name="local", task_index=1)
with tf.device('/job:local/task:1') as dev:
    
    var = tf.Variable(tf.random_uniform([5,5], -1., 1.), name='var')
print ('preped job2')

# run 1
# srv1 = server1.target
# print(srv1)
# rnn.train(srv = srv1, **cnf,**train_args)


def execute(tpl):
    srv,tns = tpl

    init = tf.global_variables_initializer()
    with tf.Session(master=srv.target,is_chief=True,checkpoint_dir='./',hooks=hooks) as sn:
        # init.run()
        tmp = sn.run(tns)
    return tmp


from multiprocessing.pool import ThreadPool
tp = ThreadPool(2)

res = tp.map(execute,[(server1,var0),(server2,var)])

# # run2
# with tf.Session(server2.target) as sn:
#     tmp = sn.run(var)
# print(tmp)





# init = tf.global_variables_initializer()
# with tf.Session('grpc://localhost:2222') as sn:
#     print ('wtf')
    
#     init.run()
#     # tmp = sn.run(res)
    
#     tmp = sn.run(rnn._train_step,
#                  feed_dict = {rnn._x: train_args['x_train'] ,
#                               rnn._y: train_args['y_train']})

#     print (tmp)
#     print('nai kala')
# # assert False



def execute(tpl):
    srv,tns = tpl

    hooks=[tf.train.StopAtStepHook(last_step=1000000)]
    gs = tf.train.get_or_create_global_step()
    #init = tf.global_variables_initializer()
    with tf.train.MonitoredTrainingSession(master=srv.target,is_chief=True,checkpoint_dir='./',hooks=hooks) as sn:
        # init.run()
        tmp = sn.run(tns)
    return tmp


