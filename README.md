This is my first nlp releated recurent neural network project in tensorflow. It is a text classifier of small sentences, very similar to a sentiement analysis. The code is not production ready, but more like a portfolio item. For details on the motivation behind the project check:

``
doc/popular.pdf
``


The network is built with classes from the low level tensorflow classes which are subsequently wrapped around python objects. The preprocesing of the sentences has taken place in advance, see inputs/data for examples on data schemas.

In order to run the classifier:

``python
python -i rnn_text_classifier.py inputs/conf-files/rnn_conf_<big_unique_hash>.json
``

The configuration files are made with the **make-rnn-conf.py** script, where you can choose the network architecture, hyper parameters and inputs.

Furthermore, custom metrics on word clsuter seperation distances are included, **sentence_statistics.py**, as well as a multicategory f1-score metric (one-vs-all style), see **base_classiefiers.py**. 

Finally, I experimented a bit with running tensorflow on spark, which gave me an idea on how to **not** build spark applications :-P

![popular.pdf](https://github.com/vsyropou/rnn-text-classifier/files/12709713/popular.pdf)
