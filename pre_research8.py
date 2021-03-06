# -*- coding: utf-8 -*-
"""pre_research3.py

Convolutional VAE with seq2seq
ファイルに保存した後にレイヤーを呼び出すために名前をつけておきたい...!
"""
#以下は必要な関数などの定義
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K

#import matplotlib.pyplot as plt
#import argparse
#import os
import joblib
import matplotlib.pyplot as plt

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
        # Returns:
        z (tensor): sampled latent vector
        """
    
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
#関数定義終わり

#KL-divergence annealingのコールバックの定義
#hp_lambdaはkl_lossの係数として用意する。
hp_lambda = K.variable(0)  # default values

from keras import callbacks

class AneelingCallback(callbacks.Callback): #callbacks.Callbackはhttps://keras.io/ja/callbacks/#callbackを参照したい。
    '''Aneeling theano shared variable.
        # Arguments
        schedule(関数): a function that takes an epoch index as input
        (integer, indexed from 0) and returns a new
        learning rate as output (float).
        '''
    def __init__(self, schedule, variable):
        super(AneelingCallback, self).__init__()
        self.schedule = schedule
        self.variable = variable #hp_lambdaにあたるもの
    
    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        value = self.schedule(epoch)
        assert type(value) == float, 'The output of the "schedule" function should be float.'
        K.set_value(self.variable, value) #上のvalueで得た値をvariableにセットする。
        #print(K.eval(self.variable)) 

#epochごとにkl_lossの係数を変更する関数
def schedule(epoch):
    if epoch < 11:
        return 0.0
    else:
        return (epoch-10)/epochs #epochsの部分がヤバいかもしれない...。

aneeling_callback = AneelingCallback(schedule, hp_lambda)
#コールバックの定義終わり

#使うGPUを自分から指定するための設定
#commndでgpustatと打てば自分の使っているGPUを確認できる
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
                        gpu_options=tf.GPUOptions(
                                                  visible_device_list="2", # specify GPU number
                                                  # allow_growth=True
                                                  )
                        )
set_session(tf.Session(config=config))

#単語のベクトルデータを読み込む
from gensim.models import word2vec
word_model = word2vec.Word2Vec.load("./aozora.model")

import pickle
import numpy as np


# 0: EMTPY
# 1: START
# 2: END
# 3: UNKNOWN
# 4: Word Index Head(単語IDの中で最も小さいもの)

#with open('aozora_seqs_wakati.dump', 'rb') as f:
    #aozora_seqs_wakati = pickle.load(f)
with open('all_sentences.dump', 'rb') as f:
    all_sentences = pickle.load(f)

#データの分布に偏りがあるためlogなどで正規化したい。
#with open('published_date.dump', 'rb') as f:
    #published_date = pickle.load(f)
with open('sentence_times.dump', 'rb') as f:
    sentence_times = pickle.load(f)

with open('word2index.dump', 'rb') as f:
    word2index = pickle.load(f)

with open('index2word.dump', 'rb') as f:
    index2word = pickle.load(f)

all_sentences_indexed = [[word2index[w]  for w in sentence] for sentence in all_sentences]
all_sentences_indexed = np.array(all_sentences_indexed)

sentence_times = np.array(sentence_times)

from tensorflow.keras.preprocessing import sequence

#pre_research2のtimestepsにあたるもの
max_len = 100

#各単語リストの後ろを空白で埋めるなり削るなりして、単語リストの長さをmax_lenで統一する。
all_sentences_indexed = sequence.pad_sequences(all_sentences_indexed, maxlen=max_len, padding='post', truncating='post', value=3)

#aozora_seqs_indexed内の単語idを単語ベクトルに変換するための関数の定義
def id2vec(data, dim, mask_value = 1):
    num_data = data.shape[0]
    max_len = data.shape[1]
    vector_seqs = np.zeros((num_data, max_len, dim+3), dtype=np.float32)
    for i, vector_seq in enumerate(vector_seqs):
        for j, index in enumerate(data[i]):
            if index != 0: # MEMO: [EMPTY]情報を0ベクトル以外で渡してはいけない。0しか出力されなくなる！
                if index ==1:
                    vector_seq[j][0]=mask_value
                elif index ==2:
                    vector_seq[j][1]=mask_value
                elif index ==3:
                    vector_seq[j][2]=mask_value
                else:
                    if index2word[index] in word_model.wv.vocab:
                        vector_seq[j] = np.array([0,0,0]+word_model.wv[index2word[index]].tolist())
                    else:
                        vector_seq[j][2]=mask_value
    return vector_seqs
    #return np.array(vector_seqs)
#関数定義終わり
'''

aozora_vector_seqs = id2vec(all_sentences_indexed, dim=200, mask_value=1)

joblib.dump(aozora_vector_seqs, open('/mnt/sdb/firebird555/aozora_vector_seqs.dat', 'wb'), compress=3)

all_sentences_indexed_shifted = []
for s in all_sentences_indexed:
    rolled = np.roll(s, 1)
    rolled[0] = 1 #STARTを先頭に持ってきた。# rolled[0] = 0
    rolled = rolled.tolist()
    rolled.append(2) #ENDを文末に持ってきた。
    rolled = np.array(rolled)
    all_sentences_indexed_shifted.append(rolled)
all_sentences_indexed_shifted = np.array(all_sentences_indexed_shifted)
#このコード文末のEND(idは2)を削ってるだけな気がする...。いらないのでは...?
all_sentences_indexed_shifted = sequence.pad_sequences(all_sentences_indexed_shifted, maxlen=max_len, padding='post', truncating='post')

aozora_shifted_vector_seqs = id2vec(all_sentences_indexed_shifted, dim=200,mask_value=1)

joblib.dump(aozora_shifted_vector_seqs, open('/mnt/sdb/firebird555/aozora_shifted_vector_seqs.dat', 'wb'), compress=3)

'''
#joblibが無いなら事前にインストールするべき
#get_ipython().system('pip install joblib')

import joblib
aozora_vector_seqs = joblib.load(open('/mnt/sdb/firebird555/aozora_vector_seqs_100.dat', 'rb'))
aozora_shifted_vector_seqs = joblib.load(open('/mnt/sdb/firebird555/aozora_shifted_vector_seqs_100.dat', 'rb'))

#以下学習モデルの設計
from keras.models import Sequential,Model
from keras.layers import Input, Softmax, CuDNNLSTM, Dense, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, RepeatVector
#import sys
from keras.layers.wrappers import TimeDistributed
from keras import regularizers

vector_dim = 203
latent_dim = 128  # Latent dimensionality of the encoding space.
# 入力シークエンスを定義してそれを処理します。
encoder_inputs = Input(shape=(None, vector_dim), name='encoder_inputs')
#encoder_inputs = Input(shape=(max_len, vector_dim))
softmax_encoder_inputs = Softmax(axis=2)(encoder_inputs)
inputs_regularizer = Model(encoder_inputs, softmax_encoder_inputs)

#lossのnanを避けたいという気持ち
#plus_epsilon = Lambda(lambda x: x + 0.01)(softmax_encoder_inputs)
#inputs_regularizer = Model(encoder_inputs, plus_epsilon)
#上のinputs_regularizerは後でlossを定義する時に参照する。

lstm_1 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True, name='encoder_LSTM')
lstm_1_outputs, state_h, state_c = lstm_1(softmax_encoder_inputs)

encoder_states = [state_h, state_c]

z_mean_h = Dense(latent_dim, activation='softmax', name='z_mean_h')(state_h)
z_log_var_h = Dense(latent_dim, activation='softmax', name='z_log_var_h')(state_h)
z_h = Lambda(sampling, output_shape=(latent_dim,), name='z_h')([z_mean_h, z_log_var_h])

z_mean_c = Dense(latent_dim, activation='softmax', name='z_mean_c')(state_c)
z_log_var_c = Dense(latent_dim, activation='softmax', name='z_log_var_c')(state_c)
z_c = Lambda(sampling, output_shape=(latent_dim,), name='z_c')([z_mean_c, z_log_var_c])

regularized_encoder_states = [z_h, z_c]

# now model.output_shape == (None, max_len, latent_dim), where None is the batch dimension.

reshape = Reshape((latent_dim, max_len, -1))(lstm_1_outputs)
#データのフォーマットはchannels_lastを想定している。
# now: model.output_shape == (None, latent_dim, max_len, 1)

#CNNのカーネル
#kernel_height=128
kernel_height=16
kernel_width=3
#kernel_height*kernel_width

# 入力: サイズがlatent_dim x max_lenで1チャンネルをもつ画像 -> (latent_dim, max_len, 1) のテンソル
# それぞれのlayerで(kernel_height, kernel_width)の畳み込み処理を適用している

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
#形がどう変遷していくのかはmodel.summaryなどで確認できる。
vision_model = Sequential()
vision_model.add(Conv2D(32, (kernel_height, kernel_width), activation='relu', input_shape=(latent_dim, max_len, 1)))
vision_model.add(Conv2D(32, (kernel_height, kernel_width), activation='relu'))
vision_model.add(MaxPooling2D(pool_size=(2, 2)))
vision_model.add(Dropout(0.25))

#vision_model.add(Conv2D(64, (kernel_height, kernel_width), activation='relu'))#shapeに関するエラーがでる...。
#vision_model.add(Conv2D(64, (kernel_height, kernel_width), activation='relu'))
#vision_model.add(MaxPooling2D(pool_size=(2, 2)))
#vision_model.add(Dropout(0.25))

vision_model.add(Flatten())
#なんらかのActivationを入れても良いかも

#encode_dimはvision_modelが出力するベクトルの次元を表す。
encoded_dim=256
#vision_model.add(Dense(encoded_dim))
vision_model.add(Dense(encoded_dim, activation='softmax'))

#上で定義したvision_modelの適用
encoded_image = vision_model(reshape)

#追加で入力する発行年
time_dim = 1
#time_inputのshapeはどう定義すべきか...?年月以外の属性も加えるべきか...?
time_input = Input(shape=(1,), name='time_input')
#時間の正規化
regular_time = Lambda(lambda x: (x - 1888)/(2018-1888))(time_input)
#２つの入力の結合
merged_vector = concatenate([encoded_image, regular_time])
#encoder?
#h = Dense(latent_dim)(merged_vector)
h = Dense(latent_dim, activation='softmax')(merged_vector)

#hという次元intermediate_dimのベクトルが与えられたとして、次元latent_dimのzを生成する
#z_mean = Dense(64, name='z_mean')(h)
z_mean = Dense(32, activation='softmax', name='z_mean')(h)
#z_mean = Dense(64, kernel_regularizer=regularizers.l2(0.001),activation='softmax', name='z_mean')(h)
#z_log_var = Dense(64, name='z_log_var')(h)
z_log_var = Dense(32, activation='softmax', name='z_log_var')(h)
#z_log_var = Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='softmax', name='z_log_var')(h)
#...上のやつ本当にDenseで良いのだろうか?
#下のsamplingは上に関数として定義されている。samplingの定義をよく見ればわかるが、z_mean,z_log_var,zの次元は同じ。
#結果を見る限りzの次元が大きすぎる可能性がある...。
z_dim = 32
z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean, z_log_var])

#zを複製
context_info = RepeatVector(max_len)(z)

#shifted_inputs = Input(shape=(None, vector_dim), name='shifted_inputs')
shifted_inputs = Input(shape=(max_len, vector_dim), name='shifted_inputs')

#どのaxis方向で結合するのが良いんだろう...。時間?ということはaxis=1?
shift_with_context_dim = vector_dim + z_dim
shift_with_context = concatenate([shifted_inputs, context_info], axis=-1)
#ここでなんらかのActivationを加えてデータを正規化すべきか...?

lstm_2 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_LSTM')
#lstm_2 = CuDNNLSTM(vector_dim, return_sequences=True, return_state=True)
pre_decoder_outputs,_,_ = lstm_2(shift_with_context, initial_state=regularized_encoder_states)#pre_research4.pyとの違い
#pre_decoder_outputs,_,_ = lstm_2(shift_with_context)

decoder_dense = Dense(vector_dim, activation='softmax',name='decoder_dense')
decoder_outputs = decoder_dense(pre_decoder_outputs)
#decoder_outputs = Lambda(lambda x: x + 0.01)(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, shifted_inputs, time_input], decoder_outputs)

encoder_model = Model([encoder_inputs,time_input], [z_mean, z_log_var, z], name='encoder_model')
eoncoder_h = Model(encoder_inputs, [z_mean_h, z_log_var_h, z_h], name='encoder_h')
encoder_c = Model(encoder_inputs, [z_mean_c, z_log_var_c, z_c], name='encoder_c')

# VAE loss = mse_loss or xent_loss + kl_loss
#うまくいった...!
def loss(inputs, times, outputs):
    #shape = (batch_size, max_len, vector_dim)
    """
        損失関数の定義
    """
    from keras.losses import binary_crossentropy, categorical_crossentropy, mean_squared_error
    softmax_epsilon_inputs = inputs_regularizer(inputs)
    z_mean, z_log_var, _ = encoder_model([inputs,times])
    z_mean_h, z_log_var_h, _ = encoder_h(inputs)
    z_mean_c, z_log_var_c, _ = encoder_c(inputs)
    #z_mean, z_log_var, _ = encoder_model.predict([inputs,times])
    #reconstruction_loss = K.sum(binary_crossentropy(inputs, outputs), axis=-1)
    #output_shape==(batch_size,)
    reconstruction_loss = K.sum(categorical_crossentropy(softmax_epsilon_inputs, outputs), axis=-1)
    #output_shape==(batch_size,)
    #reconstruction_loss = K.sum(mean_squared_error(softmax_epsilon_inputs, outputs), axis=-1)
    #output_shape==(batch_size,)
    #reconstruction_loss = categorical_crossentropy(K.flatten(inputs), K.flatten(outputs))
    #output_shape==(1,)
    #reconstruction_loss = mean_squared_error(K.flatten(softmax_epsilon_inputs), K.flatten(outputs))
    #output_shape==(1,)
    #reconstruction_loss *= NUM_INPUT_DIM*NUM_TIMESTEPS
    reconstruction_loss *= vector_dim #もしかして大き過ぎる...?
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss_h = 1 + z_log_var_h - K.square(z_mean_h) - K.exp(z_log_var_h)
    kl_loss += K.sum(kl_loss_h, axis=-1)
    kl_loss_c = 1 + z_log_var_c - K.square(z_mean_c) - K.exp(z_log_var_c)
    kl_loss += K.sum(kl_loss_c, axis=-1)
    kl_loss *= -aneeling_callback.variable
    #kl_loss *= -0.5
    
    lam = 0.01 #そのままじゃうまく行かなかったので重み付け
    #return K.mean((1-lam)*reconstruction_loss + lam*kl_loss)
    return K.mean((1-lam)*reconstruction_loss + kl_loss)
#損失関数の定義終わり

# 損失関数をこのモデルに加える
model.add_loss(loss(encoder_inputs, time_input, decoder_outputs))

#model.compile(optimizer='adam')

#オプティマイザをカスタムする場合
from keras import optimizers
# All parameter gradients will be clipped to
# a maximum norm of 1.
#adam = optimizers.Adam(lr=0.001, clipnorm=1.)
adam = optimizers.Adam(lr=0.5)
model.compile(optimizer=adam)

#model.summary()

#モデルの可視化してファイルに保存する。
#plot_model(model, to_file='model.png', show_shapes=True)
#plot_model(vision_model, to_file='model2.png', show_shapes=True)

# Run training
#epochsは本当は40くらいにしたいが一応...
epochs=40

'''
model.fit([aozora_vector_seqs, aozora_shifted_vector_seqs, sentence_times],
          batch_size=8192,
          epochs=epochs,
          callbacks=[aneeling_callback],
          validation_split=0.2)
'''

#lossの可視化
fit = model.fit([aozora_vector_seqs, aozora_shifted_vector_seqs, sentence_times],
                batch_size=8192,
                epochs=epochs,
                callbacks=[aneeling_callback],
                validation_split=0.2)

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.plot(fit.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

plot_history_loss(fit)
fig.savefig('./NLP_loss3.png')
plt.close()
