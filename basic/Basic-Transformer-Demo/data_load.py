from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex

def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('data/de.vocab.tsv','r','utf-8').read().splitlines()
             if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word:idx for idx,word in enumerate(vocab)}
    idx2word = {idx:word for idx,word in enumerate(vocab)}

    return word2idx,idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('data/en.vocab.tsv','r','utf-8').read().splitlines()
             if int(line.split()[1])>=hp.min_cnt]

    word2idx = {word:idx for idx,word in enumerate(vocab)}
    idx2word = {idx:word for idx,word in enumerate(vocab)}
    return word2idx,idx2word

def create_data(source_sents,target_sents):
    de2idx,idx2de = load_de_vocab()
    en2idx,idx2en = load_en_vocab()

    x_list ,y_list,Sources,Targets = [],[],[],[]
    for source_sent,target_sent in zip(source_sents,target_sents):
        x = [de2idx.get(word,1) for word in (source_sent+u" </S>").split()] # 1: OOV, </S>: End of Text
        y = [en2idx.get(word,1) for word in (target_sent+u" </S>").split()]

        if max(len(x),len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    #Pad
    X = np.zeros([len(x_list),hp.maxlen],np.int32)
    Y = np.zeros([len(y_list),hp.maxlen],np.int32)

    for i,(x,y) in enumerate(zip(x_list,y_list)):
        X[i] = np.lib.pad(x,[0,hp.maxlen-len(x)],'constant',constant_values=(0,0))
        Y[i] = np.lib.pad(y,[0,hp.maxlen-len(y)],'constant',constant_values=(0,0))
    return X,Y,Sources,Targets

def load_train_data():
    def _refine(line):
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split('\n') if
                line and line[0] != "<"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split('\n') if
                line and line[0] != '<']

    X, Y, Sources, Targets = create_data(de_sents, en_sents)
    return X, Y

def load_test_data():
    def _refine(line):
        line = regex.sub("<[^>]+>", "", line)
        line = regex.sub("[^\s\p{Latin}']", "", line)
        return line.strip()

    de_sents = [_refine(line) for line in codecs.open(hp.source_test,'r','utf-8').read().split('\n') if line and line[:4] == "<seg"]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test,'r','utf-8').read().split('\n') if line and line[:4] == '<seg']

    X,Y,Sources,Targets = create_data(de_sents,en_sents)
    return X,Sources,Targets

def get_batch_data():
    X, Y = load_train_data()

    num_batch = len(X) // hp.batch_size

    print(X[:10])
    print(Y[:10])
      
#[[ 129 1622    6  358    7 6349    3    0    0    0]
# [  59 2320 2736    7  249 1486    3    0    0    0]
# [  59  265  572  276   10   22 5922    3    0    0]
# [  34    7   16    1    3    0    0    0    0    0]
# [  37   63  136    9  935  396    3    0    0    0]
# [ 672   14  165    4 1550  746    3    0    0    0]
# [ 209   40  624   11    3    0    0    0    0    0]
# [  37   51    1    4   36    1    3    0    0    0]
# [  37    7   20 4103   17 1286    3    0    0    0]
# [ 159    7    8    1    3    0    0    0    0    0]]
#[[1062    6    4  413   12  661  230    3    0    0]
# [  47  749 3628   12   25  115  768    3    0    0]
# [1062    6    4  463   20   10    4 1453    3    0]
# [ 611    8    1    3    0    0    0    0    0    0]
# [  96  511   59    5   25  687  244    3    0    0]
# [  49   66   32  112  143  257   91    3    0    0]
# [  11  241    9  131    6  327    3    0    0    0]
# [  79  117    1    1    1  140   35    9    3    0]
# [  79    8    1  726    3    0    0    0    0    0]
# [ 611    4    1  648    1    3    0    0    0    0]]

    X = tf.convert_to_tensor(X,tf.int32)
    Y = tf.convert_to_tensor(Y,tf.int32)

    input_queues = tf.train.slice_input_producer([X,Y])

    x,y = tf.train.shuffle_batch(input_queues,
                                 num_threads=8,
                                 batch_size=hp.batch_size,
                                 capacity = hp.batch_size*64,
                                 min_after_dequeue=hp.batch_size * 32,
                                 allow_smaller_final_batch=False)

    return x,y,num_batch
#    x, y <tf.Tensor 'shuffle_batch:0' shape=(32, 10) dtype=int32>
#    num_batch 1703
