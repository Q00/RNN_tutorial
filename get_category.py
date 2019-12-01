import os
import json
import numpy as np
import gensim
import tensorflow as tf
from gensim.models import Word2Vec
import Bi_LSTM as Bi_LSTM

def tokenize(headline):
    rawdata = nltk.word_tokenize(headline)
    sentence = nltk.pos_tag(rawdata)
    temp = []
    for k in range(len(sentence)):
        temp.append(sentence[k][0] + '/' + sentence[k][1])
    return np.array(temp)

def Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size):
    zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
    for i in range(Batch_size):
        zero_pad[i,:np.shape(train_batch_X[i])[0],:np.shape(train_batch_X[i])[1]] = train_batch_X[i]
    return zero_pad

def Convert2Vec(model_name, doc):
    word_vec = []
    model = Word2Vec.load(model_name)
    for sent in doc:
        temp_vec = []
        for word in sent:
            if word in model.wv.vocab:
                temp_vec.append(model.wv[word]) # Word Vector Input
            else:
                #Conficient
                temp_vec.append(np.random.uniform(-0.25,0.25,300)) # used for OOV words
        word_vec.append(temp_vec)
    return word_vec


Batch_size = 1
Vector_size = 300
Maxseq_length = 1000
learning_rate = 0.001
lstm_units = 256
num_class = 5
keep_prob = 0.75


X = tf.placeholder(dtype=tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
seq_len = tf.placeholder(dtype=tf.int32, shape = [None])

print('BILSTM')

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)


with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)

modelName = "BiLSTM.model-4"
init = tf.global_variables_initializer()
saver = tf.train.import_meta_graph('BiLSTM.model-4.meta') 
def Grade(sentence, sess):

      tokens = tokenize(sentence)
      
      embedding = Convert2Vec('./post.embedding', tokens)
      zero_pad = Zero_padding(embedding, Batch_size, Maxseq_length, Vector_size)

      #global prediction
      result =  sess.run(tf.argmax(prediction,1), feed_dict = {X: zero_pad , seq_len: [len(tokens)] } ) 
      category_list = {'tech' : 0, 'business' : 1, 'sport' : 2, 'entertainment' : 3, 'politics' : 4}
      re_category_list = {v:k for k, v in category_list.items()}
      print(re_category_list[int(result)])
      print(result)
with tf.Session() as sess:
    
    sess.run(init)
    # load the variables from disk.
    saver.restore(sess, modelName)


    print("Model load")

    s = input("헤드라인 입력해주세요.")
    Grade(s, sess)





    
