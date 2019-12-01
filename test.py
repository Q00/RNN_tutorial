import time
import csv
import nltk
import os
import json
import numpy as np
import gensim
import tensorflow as tf
from gensim.models import Word2Vec
import Bi_LSTM as Bi_LSTM

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def Zero_padding(train_batch_X, Batch_size, Maxseq_length, Vector_size):
    zero_pad = np.zeros((Batch_size, Maxseq_length, Vector_size))
    for i in range(Batch_size):
        zero_pad[i,:np.shape(train_batch_X[i])[0],:np.shape(train_batch_X[i])[1]] = train_batch_X[i]
    return zero_pad

def One_hot(data):
    index_dict = {value:index for index,value in enumerate(set(data))}
    encodings = []
    for value in data:
        one_hot = np.zeros(len(index_dict))
        index = index_dict[value]
        one_hot[index] = 1
        encodings.append(one_hot)
    return np.array(encodings)

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

#Proccessing
file = open('test.csv', 'r', encoding='ISO-8859-1')
data = csv.reader(file)


category_list = {'tech' : 0, 'business' : 1, 'sport' : 2, 'entertainment' : 3, 'politics' : 4}
token =[]
embeddingmodel = []


for i in data:
    headline = i[1]
    rawdata = nltk.word_tokenize(headline)
    sentence = nltk.pos_tag(rawdata)
    temp = []
    temp_embedding = []
    all_temp = []
    for k in range(len(sentence)):
        temp_embedding.append(sentence[k][0])
        temp.append(sentence[k][0] + '/' + sentence[k][1])
    all_temp.append(temp)
    #embeddingmodel.append(temp_embedding)
    category = i[0] 
    all_temp.append(category_list.get(category))
    token.append(all_temp)

tokens = np.array(token)
print("token 처리 완료")



test_X = tokens[:,0]
test_Y = tokens[:,1]
test_Y_ = One_hot(test_Y)
test_X_ = Convert2Vec('post.embedding',test_X)
Batch_size = 32
test_size = len(test_X)
test_batch = int(test_size / Batch_size)
Vector_size = 300
seq_length = [len(x) for x in test_X]
Maxseq_length = max(seq_length)
learning_rate = 0.001
lstm_units = 128
num_class = 5
keep_prob = 0.75


X = tf.placeholder(dtype=tf.float32, shape = [None, Maxseq_length, Vector_size], name = 'X')
Y = tf.placeholder(dtype=tf.float32, shape = [None, num_class], name = 'Y')
seq_len = tf.placeholder(dtype=tf.int32, shape = [None])

print('BILSTM')

BiLSTM = Bi_LSTM.Bi_LSTM(lstm_units, num_class, keep_prob)


with tf.variable_scope("loss", reuse = tf.AUTO_REUSE):
    logits = BiLSTM.logits(X, BiLSTM.W, BiLSTM.b, seq_len)
    loss, optimizer = BiLSTM.model_build(logits, Y, learning_rate)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

modelName = "./abc/content"
init = tf.global_variables_initializer()
saver = tf.train.import_meta_graph('./abc/content.meta') 

with tf.Session() as sess:
    
    sess.run(init)
    # load the variables from disk.
    saver.restore(sess, modelName)
    #check point 안부르면 이전에 학습된 parameter가 안들어감
    #saver.restore(sess, tf.train.latest_checkpoint('./'))

    
    print("Model restored")

    total_acc = 0

    for step in range(test_batch):

        test_batch_X = test_X_[step*Batch_size : step*Batch_size+Batch_size]
        test_batch_Y = test_Y_[step*Batch_size : step*Batch_size+Batch_size]
        batch_seq_length = seq_length[step*Batch_size : step*Batch_size+Batch_size]
        test_batch_X = Zero_padding(test_batch_X, Batch_size, Maxseq_length, Vector_size)

        acc = sess.run(accuracy , feed_dict={Y: test_batch_Y, X: test_batch_X, seq_len: batch_seq_length})
        print("step :{} Accuracy : {}".format(step+1,acc))
        total_acc += acc/test_batch

    print("Total Accuracy : {}".format(total_acc))
