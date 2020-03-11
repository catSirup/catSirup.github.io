---
layout: post
comments: true
title:  "[Python] 선생님 저도 로또에 당첨되어보고 싶어요 (로또번호 생성기 개발 후기 1)"
date:   2020-03-11 23:34:19
author: devsaka
categories: develop
tags:
  - python
  - lotto
cover:
---

# 로또 당첨되고 싶다.

서민이라면 평일에 로또를 구매해서 품에 안고, 1등했을 때의 상상을 그리며 희망을 안고 살다가 토요일 저녁에 '그럼 그렇지' 와 함께 실망했다가, 다시 로또를 구매하는 루틴을 경험해본 적이 있을 것이다. `개발자답게 로또 번호를 에측하는 프로그램을 만들어보자!` 라는 막연한 생각과 함께 `진짜 랜덤인데 이걸 어떻게 예측해`에 대한 생각이 공존하던 와중... 김태영 님의 [딥로또](https://tykimos.github.io/2020/01/25/keras_lstm_lotto_v895/) 포스트를 보게 되었다.

# 진짜 딥러닝으로 로또 번호 예측을 한다고?

포스트를 들어가보면 알겠지만, 역대 로또 당첨 번호를 크롤링해서, (n-1회차 정답, n회차 정답) 셋으로 데이터 셋을 만들고, train/validate/test 셋으로 나눈 뒤, 간단한 LSTM 모델로 학습시키는 내용이었다. 

학습 결과는 당연히 처참했다. validate loss가 줄지 않은 채 상승하는 전형적인 overfitting을 보여주면서, 김태영 님도 더 이상 진행하는 것이 의미는 없지만 교육적인 측면에서 끝까지 해보시겠다고 하시며, 예측값으로 로또를 사셨는데 10줄에서 4줄, 5천원씩 총 2만원 당첨되셨다..

김태영 님은 본인이 희생할테니 하지 말라고 하셨지만, 나는 5천원이라도 당첨되어보기 위해 이 프로젝트를 시작했다.

# 다양하게 로또 번호를 뽑아보자

우선 다섯가지 방법으로 로또 번호를 뽑아보기로 했다.
1. 김태영님이 작성하신 코드를 그대로 가져와, 나름대로 모델을 튜닝해서 번호를 뽑아보자([LSTM 모델에서 비순환 유닛에 드롭아웃 걸면 결과가 좋다](https://catsirup.github.io/ai/2020/03/02/Recurrent_Neural_Network_Regularization.html)는 논문을 본 직후에 테스트해볼 것이 필요해서 요건 고대로 진행해보기로 했다.)
2. 역대 당첨 번호들 중 번호들의 빈도수를 계산해, 빈도수 별로 가중치를 부여하고 거기서 랜덤으로 뽑아내는 방법
    - 1회부터 900회까지를 기준
    - 직전 50회를 기준

3. 역대 당첨 번호들 중 번호들의 빈도수를 게산해, 빈도수 별로 역 가중치를 부여하고 거기서 랜덤으로 뽑아내는 방법
    - 1회부터 900회까지를 기준
    - 직전 50회를 기준

솔직히 1번은 김태영님의 포스트를 보고 큰 기대는 하지 않았지만.. 혹시 하는 마음으로 LSTM에 드롭아웃만 걸어주는 방식으로 코드를 작성했다.

```python
keras.layers.LSTM(128, batch_input_shape=(1, 1, 45), return_sequences=False, stateful=True, dropout=0.2),
keras.layers.Dense(45, activation='sigmoid')
```

2번과 3번의 경우, 기존의 크롤링을 통해 `lotto.csv` 을 생성하고 다음 코드를 작성했다.

```python
# 데이터 읽어옴.
def read_data():
    df = pd.read_csv("lotto.csv")

    lotto_number_list = []
    for row in range(len(df)):
        l = [df.values[row][1], df.values[row][2], df.values[row][3], df.values[row][4], df.values[row][5], df.values[row][6]]
        lotto_number_list.append(l)

    return lotto_number_list

# 빈도수를 기준으로 로또 번호를 뽑아주는 함수
def get_lotto_number_by_frequency(number_range=0, frequency_high=False):
    # 최근 몇 게임을 가져와서 빈도수를 측정할 것인지.
    if number_range == 0:
        numbers = read_data()
        number_range = len(numbers)

    else:
        numbers = read_data()[:number_range]

    # 각 번호 별로 빈도수 체크
    count = np.zeros(45)
    for i in numbers:
        for j in i:
            count[int(j) - 1] += 1
    
    # 번호 당 나온 개수만큼 추가할 배열
    number_list = []

    # 높은 빈도수를 선택했을 때
    if frequency_high == True:     
        # n: 로또 번호
        # j: 번호당 나온 수
        for n, j in enumerate(count):
            # j만큼 (n+1)한 값을 추가
            for i in range(int(j)):
                number_list.append(n + 1)

        # 섞어주고
        np.random.shuffle(number_list)
        lotto_number = []
        # 번호 여섯개 뽑기
        for i in range(6):
            idx = random.randint(0, len(number_list))
            selected_number = number_list[idx]
            lotto_number.append(selected_number)

            while selected_number in number_list:
                number_list.remove(selected_number)

        print("높은 빈도수, 최근 " + str(number_range) + " 게임 내에서 : " + str(sorted(lotto_number)))
     
    # 낮은 빈도수를 선택했을 때
    else:
        min_value = min(count)
        max_value = max(count)

        limit_value = min_value + max_value    
        for n, j in enumerate(count):
            for i in range(int(limit_value) - int(j)):
                number_list.append(n + 1)

        np.random.shuffle(number_list)
        lotto_number = []
        for i in range(6):
            idx = random.randint(0, len(number_list))
            selected_number = number_list[idx]
            lotto_number.append(selected_number)

            while selected_number in number_list:
                number_list.remove(selected_number)

        print("낮은 빈도수, 최근 " + str(number_range) + " 게임 내에서 : " + str(sorted(lotto_number)))
```

그리고 이 다섯가지를 조합해 모델에서 10개, 빈도수 별로 10개를 뽑아내는 코드를 작성했다.

```python
# 모델로 예측하는 코드는 김태영님의 포스트에 있으므로 생략
# 빈도수 별로 10개 랜덤하게 뽑아냄.
for i in range(10):
    # 0: 모델
    # 1: 50개, 높은 빈도수
    # 2: 50개, 낮은 빈도수
    # 3: 전체, 높은 빈도수
    # 4: 전체, 낮은 빈도수
    idx = random.randint(1, 4)
    # l[idx] += 1


    if idx == 0:
        predict_number(1)
    elif idx == 1:
        get_lotto_number_by_frequency(number_range=50, frequency_high=True)
    elif idx == 2:
        get_lotto_number_by_frequency(number_range=50, frequency_high=False)
    elif idx == 3:
        get_lotto_number_by_frequency(frequency_high=True)
    elif idx == 4:
        get_lotto_number_by_frequency(frequency_high=False)
```

그렇게 총 20개의 번호를 뽑았고, 이미 번호를 본 순간 망했다는 것을 직감했지만 인생사 모르니 일단 찍어보자 하는 마음으로, 난생 처음 수동을 해봤다.
![image-center]({{ site.url }}{{ site.baseurl }}/assets/images/lotto/1.png){: .align-center}
```
인증샷. 이미 번호본 순간부터 글러먹었다고 생각했다.
```

그렇게 7일 토요일 901회 당첨을 확인해보았는데
![image-center]({{ site.url }}{{ site.baseurl }}/assets/images/lotto/2.png){: .align-center}

진짜 단 한 줄도 5등 이상 당첨된 것이 없었다. 어찌보면 당연하다 싶었다.. 하지만 여기서 포기할 수 없었다. 나의 계산이 어디서 잘못되었는지를 생각하고, 좀 더 잘 뽑아줄 수 있을 것 같은 방법을 고민하기 시작했다.

### 잘못 계산한 것 메모
- 확률이 제일 낮은 모델에 10개나 투자했다. 이는 기회비용적으로도 실패였다.
- 단순히 번호별 빈도수로만 계산했다는 것이 문제였다. 


902회 로또 결과는 후기 2편으로 함께 작성하겠다.