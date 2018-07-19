import numpy as np

def truncate_pad_shufflle_sequence_data(**kwargs):

    try: #parse arguments
        in_seqs = kwargs.get('inputSequences') # indexed raw sentences
        in_labs = kwargs.get('inputLabels') # sentence labels

        seq_length = kwargs.get('sequenceLength')
        frc_train  = kwargs.get('trainingExamplesFrac')
    except KeyError as exc:
        raise RuntimeError('Cannot find required kwarg: %s.'%exc.args[0]) from exc    

    try: # open files
        x_raw = np.load(in_seqs) if type(in_seqs) == str else in_seqs
        y_raw = np.load(in_labs) if type(in_labs) == str else in_labs
    except Exception as exc:
        raise RuntimeError('Unable to parse arguments: %s.'%exc.args[1]) from exc

    assert [type(arg) in [np.array,np.ndarray] for arg in[in_seqs,in_labs]] ,\
        '"inputSequences" and "inputLabels" must be of "numpy.array" type.'

    # traincate and pad sequences accordingly
    trc_func = lambda x: x[:seq_length]
    pad_func = lambda x: np.pad(x, (0,seq_length-len(x)),'constant')
    indexed_series = map(pad_func, map(trc_func, x_raw))

    # suffle
    x = np.array(list(indexed_series))
    y = y_raw    
    
    sufflled_indices = np.random.permutation(np.arange(len(x)))
    
    x_suffled = x[sufflled_indices]
    y_suffled = y[sufflled_indices]
    
    # split
    training_examples = int(frc_train * len(x_suffled)) 

    return {'x_train' : x_suffled[:training_examples],
            'y_train' : y_suffled[:training_examples],
            'x_test'  : x_suffled[training_examples:len(x_suffled)],
            'y_test'  : y_suffled[training_examples:len(x_suffled)]
            }

