# -*- coding: utf-8 -*-
"""pre_research2.py

pre_research.pyを参考にしてこれからは作ります...。コードをいじるのはこちらのファイルということで...。
とりあえず学習と生成ができるようになった。おそらくzの次元が大きすぎて単語を生成する際に同じ単語ばかり生成してしまうので、もっと減らすべきかもしれない...。

"""
#以下は必要な関数などの定義
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import matplotlib.pyplot as plt
import argparse
import os


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

#単語のベクトルデータを読み込む
from gensim.models import word2vec
word_model = word2vec.Word2Vec.load("./aozora.model")

#使うGPUを自分から指定するための設定
#commndでgpustatと打てば自分の使っているGPUを確認できる
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
                        gpu_options=tf.GPUOptions(
                                                  visible_device_list="1,3", # specify GPU number
                                                  # allow_growth=True
                                                  )
                        )
set_session(tf.Session(config=config))

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
max_len = 10

#各単語リストの後ろを空白で埋めるなり削るなりして、単語リストの長さをmax_lenで統一する。
all_sentences_indexed = sequence.pad_sequences(all_sentences_indexed, maxlen=max_len, padding='post', truncating='post')

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
#関数定義終わり

aozora_vector_seqs = id2vec(all_sentences_indexed, dim=200, mask_value=1)

'''
    pickle.dump(aozora_vector_seqs, f)
    OverflowError: cannot serialize a bytes object larger than 4 GiB
'''
#with open('aozora_vector_seqs.dump', 'w') as f:
    #pickle.dump(aozora_vector_seqs, f)
#with open('aozora_vector_seqs.dump', 'r') as f:
    #aozora_vector_seqs = pickle.load(f)

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
aozora_seqs_indexed_shifted = sequence.pad_sequences(aozora_seqs_indexed_shifted, maxlen=max_len, padding='post', truncating='post')

aozora_shifted_vector_seqs = id2vec(all_sentences_indexed_shifted, dim=200,mask_value=1)

#以下学習モデルの設計
from keras.models import Sequential,Model
from keras.layers import Input, CuDNNLSTM, Dense, Reshape, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate, RepeatVector
from keras.layers.wrappers import TimeDistributed

# MEMO: CuDNNLSTMが使える環境では使った方が高速です
vector_dim = 203
latent_dim = 128  # Latent dimensionality of the encoding space.
# 入力シークエンスを定義してそれを処理します。
encoder_inputs = Input(shape=(None, vector_dim)) #ここにtimeの条件を加えたい...。
lstm_1 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
lstm_1_outputs, state_h, state_c = lstm_1(encoder_inputs)

encoder_states = [state_h, state_c]

# now model.output_shape == (None, max_len, latent_dim), where None is the batch dimension.

reshape = Reshape((latent_dim, max_len, -1))(lstm_1_outputs)
#データのフォーマットはchannels_lastを想定している。
# now: model.output_shape == (None, latent_dim, max_len, 1)

#CNNのカーネル
#kernel_height=128
kernel_height=16
kernel_width=3
#kernel_height*kernel_width

# 入力: サイズがlatent_dimxmax_lenで1チャンネルをもつ画像 -> (latent_dim, max_len, 1) のテンソル
# それぞれのlayerで(kernel_height, kernel_width)の畳み込み処理を適用している

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
#形がどう変遷していくのか理解していない...。
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

#encode_dimはvision_modelが出力するベクトルの次元を表す。
encoded_dim=256
vision_model.add(Dense(encoded_dim)) #ここでエラーが出ている...?

#上で定義したvision_modelの適用
encoded_image = vision_model(reshape)

#追加で入力する発行年
account_dim = 1
#time_inputのshapeはどう定義すべきか...?
time_input = Input(shape=(1,), name='time_input')

#２つの入力の結合
merged_vector = concatenate([encoded_image, time_input])
#encoder?
h = Dense(128)(merged_vector)

#hという次元intermediate_dimのベクトルが与えられたとして、次元latent_dimのzを生成する
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)
#...上のやつ本当にDenseで良いのだろうか?
#下のsamplingは下に関数として定義されている。
#結果を見る限りzの次元が大きすぎる可能性がある...。
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

#zを複製
context_info = RepeatVector(max_len)(z)

#shifted_inputs = Input(shape=(None, vector_dim), name='shifted_inputs')
shifted_inputs = Input(shape=(max_len, vector_dim), name='shifted_inputs')

#どのaxis方向で結合するのが良いんだろう...。時間?ということはaxis=1?
shift_with_context = concatenate([shifted_inputs, context_info], axis=-1)
shift_with_context_dim = vector_dim + latent_dim

lstm_2 = CuDNNLSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs,_,_ = lstm_2(shift_with_context)

decoder_dense = Dense(vector_dim, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, shifted_inputs, time_input], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#epochsは本当は40くらいにしたいが一応...
model.fit([aozora_vector_seqs, aozora_shifted_vector_seqs, sentence_times], aozora_vector_seqs,
            batch_size=32,
            epochs=1,
            validation_split=0.2)
# Save model
model.save('s2s_conv.h5')
#model.save_weights('s2s_1_weights.h5')
          
from keras.models import load_model
#model = load_model('s2s.h5')
#model = load_weights('s2s.h5')
"""テスト・センテンスをデコードするためには、以下を反復します :

適当な入力センテンスをエンコードして初期デコーダ状態を取得します。
この初期デコーダ状態とターゲットとしての “start of sequence” トークン[START]でデコーダの 1 ステップを実行します。出力は次のターゲット文字です。
予測されたターゲット文字をデコーダに追加して繰り返します。
"""

# Define sampling models
#encoder_inputs = Input(shape=(None, num_words)), encoder_states = [state_h, state_c]が上ですでに定義されている。
encoder_model = Model([encoder_inputs,time_input], z)
encoded_state = Model(encoder_inputs, encoder_states)


decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

#decoder_inputs = Input(shape=(None, num_words))は上ですでに定義されている。
#decoder_embedding = TimeDistributed(Dense(256))は上ですでに定義されている。

decoder_inputs = Input(shape=(None, shift_with_context_dim))

decoder_outputs, state_h, state_c = lstm_2(decoder_inputs)
decoder_states = [state_h, state_c] #decoderから出力される内部状態。(decoder_states_inputsと区別!!)
#decoderに入力される方の内部状態はdecoder_states_inputsとしている。
#decoder_dense = Dense(200+3, activation='softmax')は上で定義されている。
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq, time, random_state=False):
    # Encode the input as state vectors.
    if random_state:
        latent_vector = np.random.random(1,latent_dim) #zにあたるもの
        states_value = [np.random.random((1, latent_dim)), np.random.random((1, latent_dim))]
    else:
        #encoder_model = Model(encoder_inputs, encoder_states)は上で定義されている。
        time = np.array([time])
        #latent_vector = encoder_model.predict(input_seq, time) #timeも引数に必要...?
        latent_vector = encoder_model.predict([input_seq, time]) #こっち...?
        states_value = encoded_state.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, shift_with_context_dim))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, word2index['[START]']] = 1.
    target_seq[0, 0, vector_dim:shift_with_context_dim] = latent_vector
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        #h,cは後でstates_valueを更新するために使う
        output_tokens, h,c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        #sampled_token_index = np.argmax(output_tokens[0, -1, :])
        word = word_model.most_similar( [ output_tokens[0,0,3:203] ], [], 1)[0][0] #最も近いword_modelにある単語
        if word in word2index:
            sampled_token_index = word2index[word]
        else:
            sampled_token_index = 3 #この一行色々とヤバそう...。

        similarity = word_model.most_similar( [ output_tokens[0,0,3:203] ], [], 1)[0][1]
        if output_tokens[0,0,0] > similarity:
            similarity = output_tokens[0,0,0]
            sampled_token_index = 1
        if output_tokens[0,0,1] > similarity:
            similarity = output_tokens[0,0,1]
            sampled_token_index = 2
        if output_tokens[0,0,2] > similarity:
            similarity = output_tokens[0,0,2]
            sampled_token_index = 3
        
        sampled_char = index2word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '[END]' or
           len(decoded_sentence) > max_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, shift_with_context_dim)) #182行目と同じように初期化する。
        if sampled_token_index < 4:#sampled_charが単語ではなくトークンの場合
            target_seq[0, 0, sampled_token_index-1] = 1
            target_seq[0, 0, vector_dim:shift_with_context_dim] = latent_vector
        else:
            if index2word[sampled_token_index] in word_model.wv.vocab:
                target_seq[0, 0, :vector_dim] = np.array([0,0,0]+word_model.wv[index2word[sampled_token_index]].tolist())
            else:
                target_seq[0, 0, 2]=1
            target_seq[0, 0, vector_dim:shift_with_context_dim] = latent_vector
        # Update states
        states_value = [h, c]
    #ここまでwhile文
    return decoded_sentence

for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = aozora_vector_seqs[seq_index: seq_index + 1]
    time = sentence_times[seq_index]
    decoded_sentence = decode_sequence(input_seq, time, random_state=False)
    print('-')
    #print('Input sentence:', all_sentences[seq_index])
    print('Decoded sentence:', decoded_sentence)

