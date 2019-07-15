---
layout: post
comments: true
title:  "Convolutional neural network for sentence classification, 문장 분류를 위한 CNN"
date:   2019-07-15 15:50:27
author: devsaka
categories:
tags:
  - 논문구현
  - AI
  - CNN
cover:
---


# 0. 시작하면서 
NLP에서 많은 주목을 받았던 Yoon Kim 님의 ["Convolutional Neural Networks for Sentence Classification"](https://www.aclweb.org/anthology/D14-1181)의 논문을 구현해보았습니다. 

전체 코드는 [여기](https://github.com/catSirup/Convolutional-neural-network-for-sentence-classification)에 있습니다.

# 1. 요약
CNN(Convolutional neural network)은 원래 Computer vision 용으로 개발되었으나 이후 NLP에 효과적이며 의미 구분 분석, 검색, 문장 모델링 등 기타 전통적인 NLP 작업에서 우수한 결과를 얻었다고 합니다. 

이 논문에서는 비지도된 NL 모델에서 얻은 단어 벡터 위에 하나의 레이어를 가진 CNN을 얹은 모델을 구현했습니다. 여기서는 Mikolove 외 연구진들이 만든 Google News 1000억개의 단어에 대한 훈련 벡터를 사용했으나, 영어로 되어있기 때문에 [박규병 님이 만드신 한국어로 학습된 모델](https://github.com/Kyubyong/wordvectors)을 사용했습니다. 

데이터셋은 [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)을 사용했으며, 한국어로 된 영화 리뷰 데이터입니다.

정확도는 대략 80% 초중반 정도 나옵니다.

# 2. 전처리
전처리는 오픈소스인 [KoNLPy](https://konlpy-ko.readthedocs.io/ko/v0.4.3/)를 사용했습니다. 

```python
from konlpy.tag import Okt
```

먼저 데이터 셋을 불러오고 나서 공백을 모두 제거한 뒤 문장과 라벨로 분류한 뒤, Tokenize를 진행합니다.
```python
def tokenize(sentence):
    okt = Okt()
    tokenized_sentence = []

    for line in sentence:
        result = []
        temp_sentence = okt.pos(line, norm=True, stem=True)
        print(temp_sentence)
        for i in temp_sentence:                             
            if (i[1] == 'Noun' or i[1] == 'Adjective' or i[1] == 'Alpha'):                  
                result.append(i[0])
            
        tokenized_sentence.append(result)

    return tokenized_sentence
```
여기서 `norm`과 `stem`을 모두 True로 해서 최대한 데이터의 원형을 살린 뒤, 명사와 동사 그리고 영어로 된 단어만 골라내서 저장했습니다. ... 이나 다른 기호들은 전부 삭제했습니다.

그리고 Tokenize는 시간이 꽤 소요가 되는데, 데이터 셋의 총 문장 개수가 20만 개이기 때문에 매번 Tokenize를 진행하면 시간이 너무 오래 걸려, 한 번 해놓고 json으로 저장한 다음 꺼내쓰는 방식으로 구현했습니다.

```python
data, labels = load_data_and_label('./Datasets/ratings.txt')    

data = tokenize(data)
datas = []
for i in range(len(data)):
    datas.append([data[i], labels[i]])

    with open('data.json', 'w', encoding='utf-8') as make_file:
        json.dump(datas, make_file, ensure_ascii=False, indent='\t')
```

이후 데이터를 불러오고 나서 모든 문장을 `단어의 개수가 가장 많은 문장의 길이` 에 맞춰주는 작업을 진행하고, 각 단어에 인덱싱을 해주면 전처리는 끝입니다.

# 3. 모델 구현
프레임워크는 Tensorflow를 사용했으나 전통적인 방식이 아닌, Keras를 사용하는 방법을 선택했습니다. 이 부분이 좀 더 직관적이고 편하기 때문입니다. 구글에서도 [Tensorflow 2.0부터는 본격적으로 keras를 사용하는 것](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b8)을 밀고 있고, [튜토리얼](https://www.tensorflow.org/tutorials/keras?hl=ko)도 Keras를 이용하는 방식으로 알려주니, 시대의 흐름에 몸을 맡기기로 했습니다.

```python
import tensorflow as tf
from tensorflow import keras
```

논문에서 나온 모델은 총 세 가지(CNN-rand, CNN-non-static, CNN-static)이며 각각 다음과 같습니다.
- __CNN-rand__: baseline값으로 사용하기 위해 사용. 모든 단어 벡터를 임의의 값으로 초기화해서 사용하는 모델입니다.
- __CNN-non-static__ : 위의 모델과 같이 학습된 벡터를 사용했지만 각 task에서 벡터값은 update되는 모델입니다.
- __CNN-static__ : 앞서 말한 사전 학습된 word2vec 단어 벡터를 사용한 모델입니다.

그래서 string으로 모델을 선택해서 학습을 시킬 수 있도록 했습니다.
```python
model_type = "CNN-static"
```

파라미터는 논문에 나온 그대로 적용했다. 확실히 케라스로 작성하니 직관적으로 모델의 모형이 어떻게 되어있는지 코드 상에서 바로 확인이 편하다는 점이 너무 좋았다

```python
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])

elif model_type == "CNN-rand":
    embedding_weights = None

if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = keras.layers.Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = keras.layers.Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

z = keras.layers.Dropout(dropout)(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = keras.layers.Conv1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = keras.layers.MaxPooling1D(pool_size=2)(conv)
    conv = keras.layers.Flatten()(conv)
    conv_blocks.append(conv)
z = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

z = keras.layers.Dropout(dropout)(z)
z = keras.layers.Dense(hidden_dims, activation="relu")(z)
model_output = keras.layers.Dense(1, activation="sigmoid")(z)
```

# 4. 결과

결과는 다음과 같이 나왔으며, 요약하자면 adam을 사용하는 쪽이 좀 더 정확도가 높게 나왔다. 왜 사람들이 대부분 adam을 사용했는 지 알 것 같다.

### 1-1. CNN-non-static with adam

<pre>
Epoch 1/10
 - 985s - loss: 0.4867 - acc: 0.7572 - val_loss: 0.4057 - val_acc: 0.8187
Epoch 2/10
 - 1007s - loss: 0.3931 - acc: 0.8206 - val_loss: 0.3878 - val_acc: 0.8265
Epoch 3/10
 - 970s - loss: 0.3646 - acc: 0.8371 - val_loss: 0.3819 - val_acc: 0.8259
Epoch 4/10
 - 979s - loss: 0.3471 - acc: 0.8468 - val_loss: 0.3812 - val_acc: 0.8289
Epoch 5/10
 - 991s - loss: 0.3333 - acc: 0.8541 - val_loss: 0.3758 - val_acc: 0.8300
Epoch 6/10
 - 997s - loss: 0.3208 - acc: 0.8607 - val_loss: 0.3816 - val_acc: 0.8295
Epoch 7/10
 - 961s - loss: 0.3108 - acc: 0.8647 - val_loss: 0.3835 - val_acc: 0.8284
Epoch 8/10
 - 881s - loss: 0.3005 - acc: 0.8688 - val_loss: 0.3862 - val_acc: 0.8298
Epoch 9/10
 - 876s - loss: 0.2919 - acc: 0.8738 - val_loss: 0.4100 - val_acc: 0.8288
Epoch 10/10
 - 875s - loss: 0.2831 - acc: 0.8776 - val_loss: 0.3906 - val_acc: 0.8285
</pre>

### 1-2. CNN-static with adam

<pre>
Epoch 1/10
 - 731s - loss: 0.6245 - acc: 0.6434 - val_loss: 0.5579 - val_acc: 0.7015
Epoch 2/10
 - 777s - loss: 0.5759 - acc: 0.6896 - val_loss: 0.5404 - val_acc: 0.7207
Epoch 3/10
 - 778s - loss: 0.5617 - acc: 0.6999 - val_loss: 0.5233 - val_acc: 0.7279
Epoch 4/10
 - 786s - loss: 0.5530 - acc: 0.7073 - val_loss: 0.5210 - val_acc: 0.7333
Epoch 5/10
 - 786s - loss: 0.5478 - acc: 0.7096 - val_loss: 0.5142 - val_acc: 0.7364
Epoch 6/10
 - 789s - loss: 0.5427 - acc: 0.7142 - val_loss: 0.5088 - val_acc: 0.7373
Epoch 7/10
 - 788s - loss: 0.5402 - acc: 0.7167 - val_loss: 0.5110 - val_acc: 0.7406
Epoch 8/10
 - 793s - loss: 0.5354 - acc: 0.7201 - val_loss: 0.5088 - val_acc: 0.7405
Epoch 9/10
 - 780s - loss: 0.5343 - acc: 0.7209 - val_loss: 0.5065 - val_acc: 0.7432
Epoch 10/10
 - 791s - loss: 0.5319 - acc: 0.7222 - val_loss: 0.4999 - val_acc: 0.7462
</pre>

### 1-3. CNN-rand with adam
<pre>
Epoch 1/10
180000/180000 - 1072s - loss: 0.4160 - acc: 0.8019 - val_loss: 0.3810 - val_acc: 0.8199
Epoch 2/10
180000/180000 - 1066s - loss: 0.3362 - acc: 0.8520 - val_loss: 0.3808 - val_acc: 0.8243
Epoch 3/10
180000/180000 - 1069s - loss: 0.2881 - acc: 0.8760 - val_loss: 0.4039 - val_acc: 0.8213
Epoch 4/10
180000/180000 - 1071s - loss: 0.2512 - acc: 0.8932 - val_loss: 0.4162 - val_acc: 0.8181
Epoch 5/10
180000/180000 - 1070s - loss: 0.2234 - acc: 0.9056 - val_loss: 0.4504 - val_acc: 0.8138
Epoch 6/10
180000/180000 - 1039s - loss: 0.2025 - acc: 0.9152 - val_loss: 0.4789 - val_acc: 0.8154
Epoch 7/10
180000/180000 - 1035s - loss: 0.1872 - acc: 0.9209 - val_loss: 0.5084 - val_acc: 0.8134
Epoch 8/10
180000/180000 - 1034s - loss: 0.1772 - acc: 0.9250 - val_loss: 0.5325 - val_acc: 0.8101
Epoch 9/10
180000/180000 - 1030s - loss: 0.1686 - acc: 0.9289 - val_loss: 0.5442 - val_acc: 0.8102
Epoch 10/10
180000/180000 - 1031s - loss: 0.1605 - acc: 0.9320 - val_loss: 0.5656 - val_acc: 0.8109
</pre>

### 2-1. CNN-non-static with adadelta

<pre>
Epoch 1/10
 - 1789s - loss: 0.5819 - acc: 0.6855 - val_loss: 0.4978 - val_acc: 0.7563
Epoch 2/10
 - 971s - loss: 0.4991 - acc: 0.7516 - val_loss: 0.4629 - val_acc: 0.7783
Epoch 3/10
 - 970s - loss: 0.4706 - acc: 0.7709 - val_loss: 0.4389 - val_acc: 0.7898
Epoch 4/10
 - 971s - loss: 0.4527 - acc: 0.7827 - val_loss: 0.4327 - val_acc: 0.7954
Epoch 5/10
 - 971s - loss: 0.4430 - acc: 0.7888 - val_loss: 0.4233 - val_acc: 0.7988
Epoch 6/10
 - 971s - loss: 0.4352 - acc: 0.7939 - val_loss: 0.4260 - val_acc: 0.8027
Epoch 7/10
 - 971s - loss: 0.4296 - acc: 0.7968 - val_loss: 0.4179 - val_acc: 0.8015
Epoch 8/10
 - 1091s - loss: 0.4251 - acc: 0.8003 - val_loss: 0.4125 - val_acc: 0.8060
Epoch 9/10
 - 1133s - loss: 0.4199 - acc: 0.8043 - val_loss: 0.4092 - val_acc: 0.8089
Epoch 10/10
 - 1214s - loss: 0.4186 - acc: 0.8044 - val_loss: 0.4117 - val_acc: 0.8074
</pre>

### 2-2. CNN-rand with adadelta

<pre>
Epoch 1/10
 - 1326s - loss: 0.4602 - acc: 0.7678 - val_loss: 0.4125 - val_acc: 0.8046
Epoch 2/10
 - 1239s - loss: 0.3932 - acc: 0.8200 - val_loss: 0.3985 - val_acc: 0.8153
Epoch 3/10
 - 1089s - loss: 0.3757 - acc: 0.8316 - val_loss: 0.3916 - val_acc: 0.8202
Epoch 4/10
 - 1116s - loss: 0.3629 - acc: 0.8398 - val_loss: 0.3866 - val_acc: 0.8214
Epoch 5/10
 - 1028s - loss: 0.3527 - acc: 0.8476 - val_loss: 0.3830 - val_acc: 0.8246
Epoch 6/10
 - 1000s - loss: 0.3428 - acc: 0.8523 - val_loss: 0.3845 - val_acc: 0.8249
Epoch 7/10
 - 1022s - loss: 0.3340 - acc: 0.8581 - val_loss: 0.3812 - val_acc: 0.8249
Epoch 8/10
 - 1011s - loss: 0.3257 - acc: 0.8630 - val_loss: 0.3837 - val_acc: 0.8247
Epoch 9/10
 - 1000s - loss: 0.3184 - acc: 0.8670 - val_loss: 0.3917 - val_acc: 0.8253
Epoch 10/10
 - 983s - loss: 0.3117 - acc: 0.8701 - val_loss: 0.3870 - val_acc: 0.8284
 </pre>

 ### 2-3. CNN-static with adadelta

<pre>
 Epoch 1/10
 - 700s - loss: 0.6254 - acc: 0.6411 - val_loss: 0.5680 - val_acc: 0.7005
Epoch 2/10
 - 727s - loss: 0.5834 - acc: 0.6851 - val_loss: 0.5433 - val_acc: 0.7255
Epoch 3/10
 - 731s - loss: 0.5677 - acc: 0.6979 - val_loss: 0.5298 - val_acc: 0.7279
Epoch 4/10
 - 719s - loss: 0.5574 - acc: 0.7061 - val_loss: 0.5230 - val_acc: 0.7354
Epoch 5/10
 - 729s - loss: 0.5507 - acc: 0.7109 - val_loss: 0.5214 - val_acc: 0.7384
Epoch 6/10
 - 732s - loss: 0.5461 - acc: 0.7145 - val_loss: 0.5221 - val_acc: 0.7365
Epoch 7/10
 - 727s - loss: 0.5444 - acc: 0.7168 - val_loss: 0.5125 - val_acc: 0.7441
Epoch 8/10
 - 725s - loss: 0.5417 - acc: 0.7189 - val_loss: 0.5095 - val_acc: 0.7401
Epoch 9/10
 - 729s - loss: 0.5392 - acc: 0.7207 - val_loss: 0.5048 - val_acc: 0.7504
Epoch 10/10
 - 738s - loss: 0.5368 - acc: 0.7215 - val_loss: 0.5348 - val_acc: 0.7442
 </pre>
# 5. 참고
이 논문을 구현하면서 아래의 문서들을 참조했습니다.
- [NLP를 위한 CNN (2): Convolutional Neural Network for Sentence Classification](https://reniew.github.io/26/)
- [Alexander Rakhli's Keras-based implementation](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras)
- [[Keras] KoNLPy를 이용한 한국어 영화 리뷰 감정 분석](https://cyc1am3n.github.io/2018/11/10/classifying_korean_movie_review.html)