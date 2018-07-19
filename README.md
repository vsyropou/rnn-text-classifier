This is a wrapper around a standard recurent neural network text classfier. I essentially organised my prototype classifier so that it is easier to maintain and configure. The code assumes that the data are cleaned, see inputs/ for examples. The configuration is doen via json files that can be generated with the make-rnn-conf-file.oy. I also include some utilities to pad and truncate the tokenized sentences as well as some custome metrics on word clsuter seperation distances. Finally I experimented a bit with runnign tensorflow on spark and realised that it does not make much sence, see details in the scripts under spark/experimentation.

You can run an example as, form the top level directory:

python -i rnn_text_classifier.py inputs/conf-files/rnn_conf_<big_unique_hash>.json