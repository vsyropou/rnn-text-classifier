
# import
import os
import time
import subprocess
from glob import glob

rnn_script = 'rnn_text_classifier.py'
cnf_files_path = './inputs/conf-files/'
root_dir = './'
out_dir = './output'

# configuration
max_active_procs = 4
waiting_time = 3

for p in [os.path.join(out_dir,'tune-logs'),
          os.path.join(out_dir,'results')]:
    if not os.path.exists(p):
        os.makedirs(p)


# hyper parameters grid
conf_files = glob('%s*.json'%cnf_files_path)

# main loop
active_processes = lambda prs: list(filter(lambda p: p.poll() == None, prs))
processes = []
cnt = 0
for cnf in conf_files:

    args = [ 'python', rnn_script, cnf,
             '--output-dir', os.path.join(out_dir,'results'),
    ]
    

    print('Will spawn process with args :%s'%args)

    key = cnf.split('/')[-1].replace('.json','')
    processes += [subprocess.Popen(args,
                                   stdout = open('%s/tune-logs/stdout_%s.stdout'%(out_dir,key), 'wb'),
                                   stderr = open('%s/tune-logs/stderr_%s.stderr'%(out_dir,key), 'wb'),
                                   cwd = root_dir,
                                   env = os.environ)
                  ]

    cnt +=1
    print('\n Progress: %s / %s\n'%(cnt,len(conf_files)))
    while len(active_processes(processes)) > max_active_procs:
        print('Sleeping for %s seconds:'%waiting_time)
        #with ChunkNorisJoke(): pass # time nicelly

        time.sleep(waiting_time)

while len(active_processes(processes)) !=0:
    ## TODO: Print more info while waiting, or maybe random chuck noris jokes.
    print('Waiting %s seconds for processes to finish:'%waiting_time)
    # with ChunkNorisJoke(): pass # time nicelly

    time.sleep(waiting_time)
