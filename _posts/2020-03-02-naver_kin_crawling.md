---
layout: post
title:  "[Python] 네이버 지식인 크롤링하기"
date:   2020-03-02 15:40:49
author: devsaka
categories:
  - develop
tags:
  - text mining
  - crawling
  - 
  - 
image: https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/hand-held-bbq-favorites-royalty-free-image-694189032-1564779029.jpg?resize=980:*
comments: true
---

# 데이터가 필요했다.
개인적으로 연구에 사용할 데이터가 필요했다. 트위터 같은 곳에서 구해볼까 했는데 데이터 퀄리티가 너무 떨어졌다. 대부분 사진 위주에 해시태그만 있다보니 텍스트 데이터가 너무 부족한 것도 큰 이유였다.

그래서 고른 곳이 네이버 지식인이었다. 답변은 광고로 도배되어 있지만, 어쨌든 내가 필요한 건 제목과 질문이었으므로 크게 문제될 것은 아니었다.

# 개발 코드
<https://github.com/catSirup/naver_kin_crawling>

네이버 지식인을 통한 크롤링 코드를 올려두었다. 작동 방식은 단어와 기간을 설정해 검색한 뒤, 페이지 별로 질문 URL을 수집한다. 이후 수집한 URL을 한 번씩 돌면서 제목과 질문을 크롤링하게 만들었다.

# 후기
크롤링을 처음 해보아서 자주 IP 밴도 먹고 데이터도 못받아내고 그랬는데, 여러 가지를 찾아 많이 보완했다. 여전히 잘 쓰고 있는 걸 보면 가볍게 잘 만든 것 같다.