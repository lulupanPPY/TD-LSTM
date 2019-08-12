#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: rony_pan@163.com


import tensorflow as tf
from utils import *


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.02, 'l2 regularization')
tf.app.flags.DEFINE_float('att_l2_reg', 0.01, 'l2 regularization')    # regularizer for attention weights
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 12, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob1', 0.3, 'dropout keep prob')   # for embeddings
tf.app.flags.DEFINE_float('keep_prob2', 0.3, 'dropout keep prob')

tf.app.flags.DEFINE_string('operation', 'train', 'train or predict')
#tf.app.flags.DEFINE_string('train_file_path', 'data/restaurant/rest_2014_lstm_train_new.txt', 'training file')
tf.app.flags.DEFINE_string('train_file_path', 'D://Sentiment Classification//corpus_clean//Restaurants_Train_v2.csv', 'training file')
tf.app.flags.DEFINE_string('validate_file_path', 'D://Sentiment Classification//corpus_clean//Restaurants_Test_v2.csv', 'validating file')
tf.app.flags.DEFINE_string('test_file_path', 'D://Sentiment Classification//corpus_clean//Restaurants_Test_v2.csv', 'testing file')
#tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/rest_2014_word_embedding_300_new.txt', 'embedding file')
#tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/word_embedding.txt', 'embedding file')
#tf.app.flags.DEFINE_string('embedding_file_path', 'D://glove.6B//glove.6B.300d.txt', 'embedding file')
#tf.app.flags.DEFINE_string('embedding_file_path', 'C://Users//panlu//PycharmProjects//aspect_sentiment_classification//log//metadata.tsv', 'embedding file')
tf.app.flags.DEFINE_string('embedding_file_path', 'data/restaurant/word_embedding_active.txt', 'embedding file')
#tf.app.flags.DEFINE_string('embedding_file_path_new', 'data/restaurant/word_embedding.txt', 'embedding file')
tf.app.flags.DEFINE_string('embedding_file_path_new', 'data/restaurant/word_embedding_active.txt', 'embedding file')
tf.app.flags.DEFINE_string('aspect_embedding_file_path', 'data/restaurant/aspect_embedding.txt', 'aspect embedding file')
#tf.app.flags.DEFINE_string('word_id_file_path', 'data/restaurant/word_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('word_id_file_path', 'C://Users//panlu//PycharmProjects//aspect_sentiment_classification//log//word_id_mapping.txt', 'word-id mapping file')
#tf.app.flags.DEFINE_string('word_id_file_path', 'D://glove.6B//mapping.300d.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('aspect_id_file_path', 'data/restaurant/aspect_id_new.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('alpha_out_path', 'data/restaurant/alpha_out.txt', 'output of weights')
tf.app.flags.DEFINE_string('method', 'AT', 'model type: AE, AT or AEAT')
tf.app.flags.DEFINE_string('t', 'last', 'model type: ')

_dir = 'logs/' + 'checkpoint'

class LSTM(object):

    def __init__(self, embedding_dim=300, batch_size=64, n_hidden=100, learning_rate=0.01,
                 n_class=3, max_sentence_len=400, l2_reg=0.,att_l2_reg=0., display_step=4, n_iter=100, type_=''):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.att_l2_reg = att_l2_reg
        self.display_step = display_step
        self.n_iter = n_iter
        self.type_ = type_
        self.index = None
        self.true_y = None
        self.pred_y = None
        self.alpha = None
        self.reversed_dict = None
        self.word_id_mapping, self.w2v = load_word_embedding(FLAGS.word_id_file_path, FLAGS.embedding_file_path, self.embedding_dim)
        # self.word_embedding = tf.constant(self.w2v, dtype=tf.float32, name='word_embedding')
        self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding',trainable=True)
        # self.word_id_mapping = load_word_id_mapping(FLAGS.word_id_file_path)
        # self.word_embedding = tf.Variable(
        #     tf.random_uniform([len(self.word_id_mapping), self.embedding_dim], -0.1, 0.1), name='word_embedding')
        self.aspect_id_mapping, self.aspect_embed = load_aspect2id(input_file=FLAGS.aspect_id_file_path,
                                                                   embedding_dim=self.embedding_dim,
                                                                   aspect_emb_file=FLAGS.aspect_embedding_file_path)
        self.aspect_embedding = tf.Variable(self.aspect_embed, dtype=tf.float32, name='aspect_embedding',trainable=True)

        self.keep_prob1 = tf.placeholder(tf.float32)
        self.keep_prob2 = tf.placeholder(tf.float32)
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x')
            self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y')
            self.sen_len = tf.placeholder(tf.int32, None, name='sen_len')
            self.aspect_id = tf.placeholder(tf.int32, None, name='aspect_id')

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.get_variable(
                    name='softmax_w',
                    shape=[self.n_hidden, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.get_variable(
                    name='softmax_b',
                    shape=[self.n_class],
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                )
            }

        self.W = tf.get_variable(
            name='W',
            shape=[self.n_hidden*2 + self.embedding_dim, self.n_hidden*2 + self.embedding_dim],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.att_l2_reg)
        )
        self.w = tf.get_variable(
            name='w',
            shape=[self.n_hidden*2 + self.embedding_dim, 1],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.att_l2_reg)
        )
        self.Wp = tf.get_variable(
            name='Wp',
            shape=[self.n_hidden*2, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wx = tf.get_variable(
            name='Wx',
            shape=[self.n_hidden*2, self.n_hidden],
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x)
        aspect = tf.nn.embedding_lookup(self.aspect_embedding, self.aspect_id)
        if FLAGS.method == 'AE':
            prob = self.AE(inputs, aspect, FLAGS.t)
        elif FLAGS.method == 'AT':
            prob = self.AT(inputs, aspect, FLAGS.t)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob, self.y))
            self.cost = - tf.reduce_mean(tf.cast(self.y, tf.float32) * tf.log(prob)) + sum(reg_loss)

        with tf.name_scope('train'):
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(self.y, 1))
            self.true_y = tf.argmax(self.y, 1)
            self.pred_y = tf.argmax(prob, 1)
            self.accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            self._acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def restore(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tf.tables_initializer().run()
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = 'models/' + _dir + '/'
            ckpt = tf.train.get_checkpoint_state(save_dir)
            if ckpt is not None:
                path = ckpt.model_checkpoint_path
                print('loading pre-trained model from %s.....' % path)
                saver.restore(sess, path)
            else:
                print ('no model exists')
        return  sess

    def dynamic_rnn(self, cell, inputs, length, max_len, scope_name, out_type='all'):
        outputs, state = tf.nn.dynamic_rnn(
            cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        )  # outputs -> batch_size * max_len * n_hidden
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, self.n_hidden]), index)  # batch_size * n_hidden
        elif out_type == 'all_avg':
            outputs = LSTM.reduce_mean(outputs, length)
        return outputs

    def bi_dynamic_rnn(self, cell, inputs, length, max_len, scope_name, out_type='all'):
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell(self.n_hidden),
            cell_bw=cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        )
        if out_type == 'last':
            outputs_fw, outputs_bw = outputs
            outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
            outputs = tf.concat([outputs_fw, outputs_bw], 2)
        else:
            outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
        batch_size = tf.shape(outputs)[0]
        if out_type == 'last':
            index = tf.range(0, batch_size) * max_len + (length - 1)
            outputs = tf.gather(tf.reshape(outputs, [-1, 2 * self.n_hidden]), index)  # batch_size * 2n_hidden
        elif out_type == 'all_avg':
            outputs = LSTM.reduce_mean(outputs, length)  # batch_size * 2n_hidden
        return outputs

    def AE(self, inputs, target, type_='last'):
        """
        :params: self.x, self.seq_len, self.weights['softmax_lstm'], self.biases['sof
        :return: non-norm prediction values
        """
        print('I am AE.')
        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.embedding_dim])
        target = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * target
        inputs = tf.concat([inputs, target], 2)
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob1)

        cell = tf.nn.rnn_cell.LSTMCell
        outputs = self.dynamic_rnn(cell, inputs, self.sen_len, self.max_sentence_len, 'AE', FLAGS.t)

        return LSTM.softmax_layer(outputs, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    def AT(self, inputs, target, type_=''):
        print('I am AT.')
        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.embedding_dim])
        target = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * target
        in_t = tf.concat([inputs, target], 2)
        in_t = tf.nn.dropout(in_t, keep_prob=self.keep_prob1)
        cell = tf.nn.rnn_cell.LSTMCell
        hiddens = self.bi_dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT', 'all')
        #hiddens = tf.nn.dropout(hiddens, keep_prob=self.keep_prob1)
        # hidden size (?,80,600)
        h_t = tf.reshape(tf.concat([hiddens, target], 2), [-1, self.n_hidden*2 + self.embedding_dim])
        #h_t shape (?, 900)
        #print ('shape of self.W ',self.W.shape)
        M = tf.matmul(tf.tanh(tf.matmul(h_t, self.W)), self.w)
        alpha = LSTM.softmax(tf.reshape(M, [-1, 1, self.max_sentence_len]), self.sen_len, self.max_sentence_len)
        self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len])
        #print('hiddens shape ',hiddens.shape)
        r = tf.reshape(tf.matmul(alpha, hiddens), [-1, self.n_hidden*2])
        self.index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
        hn = tf.gather(tf.reshape(hiddens, [-1, self.n_hidden*2]), indices=self.index)  # batch_size * n_hidden
        #hn = tf.nn.dropout(hn, keep_prob=self.keep_prob1)
        h = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(hn, self.Wx))

        return LSTM.softmax_layer(h, self.weights['softmax'], self.biases['softmax'], self.keep_prob2)

    @staticmethod
    def softmax_layer(inputs, weights, biases, keep_prob):
        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
            predict = tf.matmul(outputs, weights) + biases
            predict = tf.nn.softmax(predict)
        return predict

    @staticmethod
    def reduce_mean(inputs, length):
        """
        :param inputs: 3-D tensor
        :param length: the length of dim [1]
        :return: 2-D tensor
        """
        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keep_dims=False) / length
        return inputs

    @staticmethod
    def softmax(inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, 2, keep_dims=True)
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    def run(self):


        with tf.Session() as sess:
            title = '-d1-{}d2-{}b-{}r-{}l2-{}sen-{}dim-{}h-{}c-{}'.format(
                FLAGS.keep_prob1,
                FLAGS.keep_prob2,
                FLAGS.batch_size,
                FLAGS.learning_rate,
                FLAGS.l2_reg,
                FLAGS.max_sentence_len,
                FLAGS.embedding_dim,
                FLAGS.n_hidden,
                FLAGS.n_class
            )
            summary_loss = tf.summary.scalar('loss' + title, self.cost)
            summary_acc = tf.summary.scalar('acc' + title, self._acc)
            train_summary_op =  tf.summary.merge([summary_loss, summary_acc])
            validate_summary_op =  tf.summary.merge([summary_loss, summary_acc])
            test_summary_op =  tf.summary.merge([summary_loss, summary_acc])
            import time
            timestamp = str(int(time.time()))
            train_summary_writer = tf.summary.FileWriter(_dir + '/train', sess.graph)
            test_summary_writer = tf.summary.FileWriter(_dir + '/test', sess.graph)
            validate_summary_writer = tf.summary.FileWriter(_dir + '/validate', sess.graph)

            init = tf.global_variables_initializer()
            sess.run(init)

            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = 'models/' + _dir + '/'
            ckpt = tf.train.get_checkpoint_state(save_dir)
            if ckpt is not None:
                path = ckpt.model_checkpoint_path
                print('loading pre-trained model from %s.....' % path)
                saver.restore(sess, path)


            # saver.restore(sess, 'models/logs/1481529975__r0.005_b2000_l0.05self.softmax/-1072')


            import os
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if (FLAGS.operation == 'train'):
                tr_x, tr_sen_len, tr_target_word, tr_y,self.reversed_dict = load_inputs_twitter_at(
                    FLAGS.train_file_path,
                    self.word_id_mapping,
                    self.aspect_id_mapping,
                    self.max_sentence_len,
                    self.type_
                )
                te_x, te_sen_len, te_target_word, te_y,self.reversed_dict = load_inputs_twitter_at(
                    FLAGS.test_file_path,
                    self.word_id_mapping,
                    self.aspect_id_mapping,
                    self.max_sentence_len,
                    self.type_
                )

                max_acc = 0.
                max_alpha = None
                max_ty, max_py = None, None
                training_cnt = 0
                for i in range(self.n_iter):
                    acc, loss, cnt = 0., 0., 0
                    for train, num in self.get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, self.batch_size, FLAGS.keep_prob1, FLAGS.keep_prob2):
                        _, step, summary,word_embedding,aspect_embedding,_loss, _acc, ty, py,x_ind,alpha  \
                            = sess.run([self.optimizer, self.global_step, train_summary_op,self.word_embedding,self.aspect_embedding,
                                        self.cost, self.accuracy,self.true_y, self.pred_y,self.x,self.alpha
                                        ], feed_dict=train)
                        train_summary_writer.add_summary(summary, step)
                        acc += _acc
                        loss += _loss * num
                        cnt += num
                    print('training :all samples={}, correct prediction={}'.format(cnt, acc))
                    print('training :Iter {}: mini-batch loss={:.6f}, training acc={:.6f}'.format(i, loss / cnt, acc / cnt))

                    # show some training progress:

                    acc, loss, cnt = 0., 0., 0
                    flag = True
                    summary, step = None, None
                    alpha = None
                    ty, py = None, None
                    ind_output_alpha = False
                    if (i%3 == 0 and i > 0):
                        print ('********** saving embeddings and alphas to text for iteration ',i,' might take a few minutes')
                        ind_output_alpha = True
                        alpha_out = open(FLAGS.alpha_out_path,'w')
                    for test, num in self.get_batch_data(te_x, te_sen_len, te_y, te_target_word, 2000, 1.0, 1.0, False):
                        _loss, _acc, _summary, _step, alpha, ty, py,x_ind = sess.run([self.cost, self.accuracy, validate_summary_op, self.global_step, self.alpha, self.true_y, self.pred_y,self.x],
                                                                feed_dict=test)
                        acc += _acc
                        loss += _loss * num
                        cnt += num
                        ty_map_back = []
                        for pos in ty:
                            ty_map_back.append(self.reversed_dict[pos])
                        ty = ty_map_back
                        py_map_back = []
                        for pos in py:
                            py_map_back.append(self.reversed_dict[pos])
                        py = py_map_back
                        if (ind_output_alpha):
                            #output alpha
                            save_word_embedding_file(FLAGS.embedding_file_path_new,word_embedding,self.word_id_mapping)
                            save_aspect_embedding_file(FLAGS.aspect_embedding_file_path, aspect_embedding, self.aspect_id_mapping)
                            for i in range(len(ty)):
                                alpha_out.write(str(ty_map_back[i])+" "+str(py_map_back[i])+" "+" ".join(str(j) for j in alpha[i]) + '\n')
                            ind_output_alpha = False
                            alpha_out.close()

                        if flag:
                            summary = _summary
                            step = _step
                            flag = False
                            alpha = alpha
                            ty = ty
                            py = py

                    print('testing  :all samples={}, correct prediction={}'.format(cnt, acc))
                    test_summary_writer.add_summary(summary, step)
                    saver.save(sess, save_dir, global_step=step)
                    print('testing  :Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, loss / cnt, acc / cnt))
                    if acc / cnt > max_acc:
                        max_acc = acc / cnt
                        max_alpha = alpha
                        max_ty = ty
                        max_py = py

                print('Optimization Finished! Max acc={}'.format(max_acc))
                fp = open(FLAGS.alpha_out_path, 'w')
                for y1, y2, ws in zip(max_ty, max_py, max_alpha):
                    fp.write(str(y1) + ' ' + str(y2) + ' ' + ' '.join([str(w) for w in ws]) + '\n')

                print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
                    self.learning_rate,
                    self.n_iter,
                    self.batch_size,
                    self.n_hidden,
                    self.l2_reg
                ))




    def get_batch_data(self, x, sen_len, y, target_words, batch_size, keep_prob1, keep_prob2, is_shuffle=True):
        for index in batch_index(len(y), batch_size, 1, is_shuffle):
            '''
            print ('training data :',x[index])
            print('training data :', y[index])
            print('training data :', sen_len[index])
            print('training data :', target_words[index])
            print('training data :', keep_prob1)
            print('training data :', keep_prob2)
            '''
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.aspect_id: target_words[index],
                self.keep_prob1: keep_prob1,
                self.keep_prob2: keep_prob2,
            }
            yield feed_dict, len(index)
    #def predict(self):


def save_aspect_embedding_file(file_loc,embedding, word_id_mapping):
    f = open(file_loc,'w')
    word_list = list(word_id_mapping.keys())
    for i in range(0,len(word_list)):
        f.write(word_list[i]+' '+" ".join(str(j) for j in embedding[i]) + '\n')
    f.close()

def save_word_embedding_file(file_loc,embedding, word_id_mapping):
    f = open(file_loc,'w',encoding='utf-8')
    word_list = list(word_id_mapping.keys())
    for i in range(0,len(word_list)):
        f.write(word_list[i]+' '+" ".join(str(j) for j in embedding[i]) + '\n')
    f.close()



def main(_):
    lstm = LSTM(
        embedding_dim=FLAGS.embedding_dim,
        batch_size=FLAGS.batch_size,
        n_hidden=FLAGS.n_hidden,
        learning_rate=FLAGS.learning_rate,
        n_class=FLAGS.n_class,
        max_sentence_len=FLAGS.max_sentence_len,
        l2_reg=FLAGS.l2_reg,
        att_l2_reg=FLAGS.att_l2_reg,
        display_step=FLAGS.display_step,
        n_iter=FLAGS.n_iter,
        type_=FLAGS.method
    )
    if FLAGS.operation == 'train':
        lstm.run()
    elif FLAGS.operation == 'predict':
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            tf.tables_initializer().run()
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = 'models/' + _dir + '/'
            ckpt = tf.train.get_checkpoint_state(save_dir)
            if ckpt is not None:
                path = ckpt.model_checkpoint_path
                print('loading pre-trained model from %s.....' % path)
                saver.restore(sess, path)
            else:
                print ('no model exists')
            text = 'I went to have a meal of steam seafood today ordered Jiwei shrimp razor clam scallop oyster all tasts not good'
            aspect = 'food'
            print('try predict: ', text)
            print('with aspect: ', aspect)
            tr_x, tr_sen_len, tr_target_word = process_console_input(
                word_id_file=lstm.word_id_mapping,
                aspect_id_file=lstm.aspect_id_mapping,
                sentence=text,
                aspect_word=aspect,
                sentence_len=FLAGS.max_sentence_len,
                type_='', encoding='utf8'
            )
            print(lstm.alpha)
            print(lstm.pred_y)
            alpha, py = sess.run(
                [lstm.alpha, lstm.pred_y],
                feed_dict=
                {
                    lstm.x: tr_x,
                    lstm.sen_len: tr_sen_len,
                    lstm.aspect_id: tr_target_word,
                    lstm.keep_prob2: 1.0,
                    lstm.keep_prob1: 1.0,
                }
            )
            print('prediction is :')
            if py == 0 :
                pred = 'Positive'
            elif py == 1:
                pred = 'Negative'
            else:
                pred = 'Unknown predict value of:'+str(py)
            print(pred)
            print(alpha)




if __name__ == '__main__':
    tf.app.run()
