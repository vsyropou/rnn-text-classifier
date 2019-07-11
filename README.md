This is my first nlp releated recurent neural network project in tensorflow. It is text classifier of small sentences, very similar to sentiement analysis workflow. The code is not production ready, but more like a portfolio item. The network is built with utilizing low level tensorflow api classes which are subsequently wrapped around python objects. The preprocesing of the sentences is done in advance, see inputs/data for examples.

In order to run the classifier:

``python
python -i rnn_text_classifier.py inputs/conf-files/rnn_conf_<big_unique_hash>.json
``

The configuration files are made with the **make-rnn-conf.py** script, wher eyou can tune the network architecture, hyper parameters and inputs.

Furthermore, custom metrics on word clsuter seperation distances are included, **sentence_statistics.py**, as well as a multicategory f1-score metric (one-vs-all style), see **base_classiefiers.py**. 

Finally, I experimented a bit with running tensorflow on spark, which gave me an idea on how to **not** build spark applications :-P
