import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# LSTM training, test
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import warnings
warnings.filterwarnings(action='ignore')

from konlpy.tag import Okt
okt = Okt()


# data preprocessing
def preprocessing(data):
  # 데이터 중복 값 확인 및 제거
  # document: 약 4,000개 중복(총150,000개 - 146,182개),
  # label 2값 확인(0, 1만 가지기 때문에)
  data['document'].nunique(), data['label'].nunique()

  # null 값을 가진 값이 어느 인덱스의 위치에 존재하는지 확인
  data.loc[data.document.isnull()]

  # Null 값이 존재하는 행 제거
  data = data.dropna(how = 'any')

  # 한글(자음, 모음)과 공백을 제외하고 모두 제거
  data['document'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
  # 댓글 중, 기존에 한글이 없었을 경우 무의미한 데이터이며
  # 해당 데이터를 empty 값으로 변경하여 제거
  data['document'] = data['document'].str.replace('^ +', "")
  data['document'].replace('', np.nan, inplace=True)

  # null 값들 의미가 없는 데이터이므로 삭제
  data = data.dropna(how = 'any')

  return data


# 불용어 제거
stopwords = ['1','저','그','ㅋ','을','때','의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
okt = Okt()


# train data tokenization
# 입력값: Null값이 제거된 데이터셋 / 출력값: 불용어를 제거한 후 Null값도 제거한 데이터셋
# 기능: 불용어 제거, 단어 빈도수가 2회 이하인 단어 수를 찾아내고 공백 제거(공백 제거 목적)
def train_tokenizer(data):
    X_data = []
    for sentence in data['document']:
        # 형태소 분석기(Okt())에서 토큰화(한글은 띄어쓰기) 실행. stem=True로 일정 수준 정규화(동사,명사화)
        X_tmp = okt.morphs(sentence, stem=True)
        X_tmp = [word for word in X_tmp if not word in stopwords]  # 불용어 사전에 없으면 리스트에 추가
        X_data.append(X_tmp)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_data)

    # 빈도수가 3회 미만인 단어들이 이 데이터에서 얼만큼 비중을 차지하는지 확인
    threshold = 3  # 단어의 등장 빈도수 기준
    rare_cnt = 0  # 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0  # 빈도수가 threshold보다 작은 단어의 빈도수 총 합
    total_cnt = len(tokenizer.word_index)  # 단어의 수

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어 빈도수가 threshold보다 작을 경우
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    # 빈도수가 2회 이하인 단어들은 제외
    # 0번일 경우를 고려하여 크기는 +1을 해준다.
    voca_size = total_cnt - rare_cnt + 1

    # 토큰화(단어단위로 쪼갬)
    tokenizer = Tokenizer(voca_size)
    # 단어에 숫자(인덱스)를 부여
    tokenizer.fit_on_texts(X_data)

    y_data = np.array(data['label'])
    global tk
    tk = tokenizer
    X_data, y_data = train_padding(X_data, y_data)

    return X_data, y_data, voca_size


# train data padding
def train_padding(X_data, y_data):
    # texts_to_sequences, 단어에 순번을 지정, 0 ~ 19,415번 단어가 있음
    X_data = tk.texts_to_sequences(X_data)
    # empty samples 제거
    # 단어가 1개 미만의 값이 없는 데이터 제거
    drop_data = [index for index, sentence in enumerate(X_data) if len(sentence) < 1]

    # 빈 샘플 제거
    X_data = np.delete(X_data, drop_data, axis=0)  # X_data에서 drop_data을 사용해서 제거
    y_data = np.delete(y_data, drop_data, axis=0)

    # 전체 훈련 데이터중 94%가 길이가 30 이하이므로 모든 샘플의 길이를 30으로 조정
    X_data = pad_sequences(X_data, maxlen=30)

    return X_data, y_data


# test data tokenization
# 입력값: Null값이 제거된 데이터셋 / 출력값: 불용어를 제거한 후 Null값도 제거한 데이터셋
# 기능: 불용어 제거, 단어 빈도수가 2회 이하인 단어 수를 찾아내고 공백 제거(공백 제거 목적)
def test_tokenizer(data):
    X_data = []
    for sentence in data['document']:
        # 형태소 분석기(Okt())에서 토큰화(한글은 띄어쓰기) 실행. stem=True로 일정 수준 정규화(동사,명사화)
        X_tmp = okt.morphs(sentence, stem=True)
        X_tmp = [word for word in X_tmp if not word in stopwords]  # 불용어 사전에 없으면 리스트에 추가
        X_data.append(X_tmp)

    y_data = np.array(data['label'])

    X_data = test_padding(X_data)

    return X_data, y_data


# test data padding
def test_padding(X_data):

    # texts_to_sequences, 단어에 순번을 지정, 0 ~ 19,415번 단어가 있음
    X_data = tk.texts_to_sequences(X_data)

    #전체 훈련 데이터중 94%가 길이가 30 이하이므로 모든 샘플의 길이를 30으로 조정
    X_data = pad_sequences(X_data, maxlen = 30)

    return X_data


# data modeling
def lstm_modeling(X_data, y_data, X_data2, y_data2, voca_size): #입력값: / 출력값: /함수의 기능:

    model = Sequential()
    model.add(Embedding(voca_size, 100))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))

    # 검증 데이터 손실이 증가하면, 과적합 위험. 검증 데이터 손실이 4회 증가하면 학습을 조기 종료
    # ModelCheckpoint를 사용하여 검증 데이터의 정확도(val_acc)가 이전보다 좋아질 경우에만 모델 저장
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_data, y_data, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)

    loaded_model = load_model('best_model.h5')
    print("\n 테스트 정확도 : %0.4f" % (loaded_model.evaluate(X_data2, y_data2)[1]))


# test data prediction
def predict(new_sentence):
    loaded_model = load_model('best_model.h5')
    # 토큰화
    new_sentence = okt.morphs(new_sentence, stem=True)
    # 불용어
    new_sentence = [word for word in new_sentence if not word in stopwords]

    pad_new = test_padding([new_sentence])
    # 예측
    score = float(loaded_model.predict(pad_new))
    if (score > 0.5):
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))







