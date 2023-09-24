This is my first nlp releated recurent neural network project in tensorflow. It is a text classifier of small sentences, very similar to a sentiement analysis. The code is not production ready, but more like a portfolio item. I foudn the motivation for this project quite intertasting. If you navigate to ![doc/popular.pdf](https://github.com/vsyropou/rnn-text-classifier/blob/master/doc/popular.pdf) I hope you will also find intereasting too. 

## The problem
Apart from building my first nlp pipeline in one week my most satisfing moment was as I exaplain in the post above when I summed the word embedings of the words in a given question. I wanted to see if the cosine distnace of these similar questions was small. I also wanted to know how the reduced embedings space, say if you you pca, will look like if you plot all the questions. Will similar questiosn cluster. What do you think happened..??

## The algorithm
The network is built with classes from the low level tensorflow classes which are subsequently wrapped around python objects. The preprocesing of the sentences has taken place in advance, see inputs/data for examples on data schemas.

## How to run
In order to run the classifier:

``python
python -i rnn_text_classifier.py inputs/conf-files/rnn_conf_<big_unique_hash>.json
``

The configuration files are made with the **make-rnn-conf.py** script, where you can choose the network architecture, hyper parameters and inputs.

Furthermore, custom metrics on word clsuter seperation distances are included, **sentence_statistics.py**, as well as a multicategory f1-score metric (one-vs-all style), see **base_classiefiers.py**. 

Finally, I experimented a bit with running tensorflow on spark, which gave me an idea on how to **not** build spark applications :-P

