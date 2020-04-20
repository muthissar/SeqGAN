import numpy as np
import contextlib
from scipy.stats import norm
import tensorflow as tf

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class TabularSimple:
    def __init__(self, seq_length,n_vocabulary,n_modes):
        with temp_seed(0):
            self.vocab = range(n_vocabulary)
            self.seq_length = seq_length
            self.table = {"": self._create_dist(n_vocabulary,n_modes)}
            for _ in range(self.seq_length - 1 ):
                new_dict = {}
                for key in self.table:
                    for i in self.vocab:
                        new_dict[key + "|" + str(i)] = self._create_dist(n_vocabulary,n_modes)
                self.table.update(new_dict)
    def _create_dist(self,n_vocabulary,n_modes):
        dist = np.zeros(len(self.vocab))
        modes = np.random.choice(self.vocab,n_modes,False)
        for i in self.vocab:
            if i in modes:
                dist[i] = np.random.uniform(7,10)
            else:
                dist[i] = np.random.uniform(1,4)
        dist /= np.sum(dist)
        return dist
    
    def sample(self,size):
        samples = np.zeros((size,self.seq_length),dtype=np.int32)
        for sample_j in range(size):
            key = ""
            for i in range(self.seq_length):
                word = np.random.choice(self.vocab,p = self.table[key])
                samples[sample_j][i] = word
                key +=  "|" + str(word)
        return samples
    '''
    def cross_entropy(self,model,sess):
        for key in self.table.keys():
            seq = key.split('|')
            seq[0] = model.start_token[0]
            h_t = model.h0
            for token in seq:
                x_t = tf.nn.embedding_lookup(model.g_embeddings, [token]*128)
                h_t = model.g_recurrent_unit(x_t, h_t)  # hidden_memory_tuple
                o_t = model.g_output_unit(h_t)  # batch x vocab , logits not prob
            #log_prob = tf.log(tf.nn.softmax(o_t))
            q = sess.run(tf.nn.softmax(o_t))[0,:]
            p = self.table[key]
            return  - np.mean(p * np.log(q))
    '''

    def ll(self,samples):
        batch_size , seq_length = samples.shape
        ll = 0
        for sample_j in range(batch_size):
            key = ""
            for i in range(seq_length):
                word = samples[sample_j][i]
                ll += np.log(self.table[key][word])
                key +=  "|" +str(word)
        ll /= batch_size * seq_length
        return ll


class TabularNormal(TabularSimple):
    def _create_dist(self,n_vocabulary,n_modes):
        dist = np.zeros(len(self.vocab))
        modes = np.random.choice(self.vocab,n_modes,False)
        x = np.linspace(0, 10, len(self.vocab), endpoint=False)
        for mode in modes:
            dist += norm.pdf(x, x[mode], 1)
        dist /= np.sum(dist)
        return dist

#class LSTMTarget(Tar)