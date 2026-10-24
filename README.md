# sEMG 데이터 기반 사용자 동작 분류 및 머신러닝 모델 성능 비교

## 1. 문제 정의 및 실습 목표

### 1.1 연구 목적

본 실습의 목적은 표면 근전도(surface Electromyography, sEMG) 데이터를 이용하여 손 동작 및 팔꿈치 각도에 따른 클래스를 분류하고, 여러 머신러닝 모델의 성능을 비교하는 것이다.

sEMG는 근육의 활성 과정에서 발생하는 전기적 신호를 측정한 데이터이다. 이러한 신호에는 동작에 따라 서로 다른 특징이 포함될 수 있으며, 이를 활용하면 사용자의 동작이나 동작 유형을 분류할 수 있다.

본 실습에서는 제공된 sEMG 데이터셋을 활용하여 SVM(Support Vector Machine), RF(Random Forest), XGB(XGBoost) 모델을 학습하고 성능을 평가하였다. 이후 5-fold 교차검증, 대응표본 t-검정, 학습 시간 및 추론 시간 측정을 통해 모델별 성능과 계산 비용을 종합적으로 분석하였다.

### 1.2 실습 목표

1. sEMG 데이터셋을 불러오고 데이터의 크기와 클래스 구성을 확인한다.
2. 준비된 특징 데이터를 사용하여 세 가지 분류 모델을 학습한다.
3. 정확도, 정밀도, 재현율 및 F1-score를 비교한다.
4. 5-fold 교차검증으로 모델 성능의 변동을 확인한다.
5. 대응표본 t-검정으로 모델 간 성능 차이의 통계적 유의성을 확인한다.
6. 학습 시간과 추론 시간을 측정하여 모델 선택에 필요한 계산 비용을 비교한다.

### 1.2 실행코드
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. 데이터셋 불러오기
d = np.load("step5_dataset.npz", allow_pickle=True)

Xtr = d["Xtr"]
ytr = d["ytr"]
Xte = d["Xte"]
yte = d["yte"]

print("학습 데이터:", Xtr.shape)
print("학습 라벨:", ytr.shape)
print("테스트 데이터:", Xte.shape)
print("테스트 라벨:", yte.shape)

# 2. 라벨을 정수로 변환
le = LabelEncoder()
ytr = le.fit_transform(ytr)
yte = le.transform(yte)

# 3. 세 가지 모델 정의
models = {
    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        decision_function_shape="ovr"
    ),

    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGB": XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )
}

# 4. 세 모델 학습
for name, model in models.items():
    model.fit(Xtr, ytr)
    print(name, "학습 완료")

print("세 모델 학습이 모두 끝났습니다.")

# ======================================

# 3주차 실습 2단계: Hold-out 평가

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd

rows = []

for k, m in models.items():
    # 테스트 데이터 예측
    p = m.predict(Xte)

    # 네 가지 평가 지표 계산
    rows.append([
        k,
        accuracy_score(yte, p),
        precision_score(yte, p, average="macro", zero_division=0),
        recall_score(yte, p, average="macro", zero_division=0),
        f1_score(yte, p, average="macro", zero_division=0)
    ])

# 평가 결과를 표로 만들기
df = pd.DataFrame(
    rows,
    columns=["model", "acc", "precision", "recall", "f1"]
)

print("\n===== Hold-out 평가 결과 =====")
print(df.round(4).to_string(index=False))

# ===========================================


# ===== 실습 3단계: 혼동행렬 시각화 =====

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

L = le.classes_  # 실제 인코딩 순서

# RF 모델의 테스트 데이터 예측
pred = models["RF"].predict(Xte)

# 혼동행렬 계산
cm = confusion_matrix(
    yte,
    pred,
    labels=np.arange(len(L))
)

# 그래프 작성
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(L)))
ax.set_xticklabels(L, rotation=45, ha="right")
ax.set_yticks(range(len(L)))
ax.set_yticklabels(L)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("RF Confusion Matrix")

# 각 칸에 개수 표시
for i in range(len(L)):
    for j in range(len(L)):
        if cm[i, j] != 0:
            ax.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                fontsize=8
            )

plt.colorbar(im, ax=ax)
plt.tight_layout()

# 이미지 파일 저장
plt.savefig("confusion8.png", dpi=120)
plt.show()

print("혼동행렬 저장 완료: confusion8.png")

# ==========================================

# ===== 실습 4단계: 5-fold 교차검증 =====


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 저장된 학습 데이터(Xtr, ytr)를 이용한 5-fold 교차검증
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_acc = {k: [] for k in models}

for tr_i, va_i in skf.split(Xtr, ytr):

    # 학습 데이터와 검증 데이터 분리
    Xa, Xb = Xtr[tr_i], Xtr[va_i]
    ya, yb = ytr[tr_i], ytr[va_i]

    # 학습 데이터 기준으로 표준화
    sc = StandardScaler()
    Xa = sc.fit_transform(Xa)
    Xb = sc.transform(Xb)

    # 각 모델을 독립적으로 학습하고 평가
    for k, proto in models.items():
        m = clone(proto)
        m.fit(Xa, ya)
        pred = m.predict(Xb)

        acc = accuracy_score(yb, pred)
        fold_acc[k].append(acc)

# 평균 정확도와 표준편차 출력
print("\n===== 5-fold 교차검증 결과 =====")

for k, v in fold_acc.items():
    print(
        f"{k}: "
        f"{np.mean(v) * 100:.2f} ± "
        f"{np.std(v) * 100:.2f} %"
    )

# ==========================================

# ===== 실습 5단계: 통계 검정 =====

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from xgboost import XGBClassifier
from scipy.stats import ttest_rel
from itertools import combinations


# ==================================================
# 1단계: 데이터 불러오기
# ==================================================

print("===== 1단계: 데이터 불러오기 =====")

data = np.load("step5_dataset.npz", allow_pickle=True)

Xtr = data["Xtr"]
ytr = data["ytr"]
Xte = data["Xte"]
yte = data["yte"]

print("학습 데이터:", Xtr.shape)
print("학습 라벨:", ytr.shape)
print("테스트 데이터:", Xte.shape)
print("테스트 라벨:", yte.shape)


# ==================================================
# 2단계: 라벨 인코딩
# ==================================================

print("\n===== 2단계: 라벨 인코딩 =====")

le = LabelEncoder()

ytr = le.fit_transform(ytr)
yte = le.transform(yte)

num_classes = len(le.classes_)

print("클래스:", le.classes_)
print("클래스 수:", num_classes)


# ==================================================
# 3단계: 모델 정의 및 학습
# ==================================================

print("\n===== 3단계: 모델 학습 및 평가 =====")

models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", random_state=42)
    ),

    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGB": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1
    )
}

test_results = {}

for name, model in models.items():
    print(f"\n--- {name} 학습 ---")

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec = precision_score(
        yte, pred, average="weighted", zero_division=0
    )
    rec = recall_score(
        yte, pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        yte, pred, average="weighted", zero_division=0
    )

    test_results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n분류 보고서:")
    print(
        classification_report(
            yte,
            pred,
            labels=np.arange(num_classes),
            target_names=[str(c) for c in le.classes_],
            zero_division=0
        )
    )


# ==================================================
# 4단계: 5-fold 교차검증
# ==================================================

print("\n===== 4단계: 5-fold 교차검증 =====")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_acc = {name: [] for name in models}

for fold, (tr_i, va_i) in enumerate(skf.split(Xtr, ytr), start=1):
    Xa, Xb = Xtr[tr_i], Xtr[va_i]
    ya, yb = ytr[tr_i], ytr[va_i]

    print(f"\n--- Fold {fold} ---")

    for name, prototype in models.items():
        model = clone(prototype)

        model.fit(Xa, ya)
        pred = model.predict(Xb)

        acc = accuracy_score(yb, pred)
        fold_acc[name].append(acc)

        print(f"{name}: {acc * 100:.2f}%")

print("\n===== 교차검증 평균 정확도 =====")

for name, scores in fold_acc.items():
    mean_acc = np.mean(scores) * 100
    std_acc = np.std(scores) * 100

    print(
        f"{name}: {mean_acc:.2f} ± {std_acc:.2f}%"
    )


# ==================================================
# 5단계: 통계 검정
# ==================================================

print("\n===== 5단계: 대응표본 t-검정 =====")

alpha = 0.05

for a, b in combinations(models.keys(), 2):
    scores_a = np.asarray(fold_acc[a])
    scores_b = np.asarray(fold_acc[b])

    t, p = ttest_rel(scores_a, scores_b)

    if not np.isfinite(p):
        print(f"{a} vs {b}: 검정 불가")
        continue

    verdict = "유의함" if p < alpha else "유의하지 않음"

    print(
        f"{a} vs {b}: "
        f"t = {t:.4f}, p = {p:.4f} ({verdict})"
    )


# ==================================================
# 6단계: Random Forest 혼동행렬 저장
# ==================================================

print("\n===== 6단계: 혼동행렬 저장 =====")

rf_model = clone(models["RF"])
rf_model.fit(Xtr, ytr)
rf_pred = rf_model.predict(Xte)

cm = confusion_matrix(
    yte,
    rf_pred,
    labels=np.arange(num_classes)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion8.png", dpi=300)
plt.show()

print("혼동행렬 저장 완료: confusion8.png")
print("\n모든 단계가 완료되었습니다.")

# =======================================


# ==================================================
# 6단계: 계산 비용과 리포트
# ==================================================

import time

print("\n===== 6단계: 계산 비용 측정 =====")

timing_results = {}

for k, m in models.items():

    # 학습 시간 측정
    t0 = time.time()
    m.fit(Xtr, ytr)
    train_t = time.time() - t0

    # 추론 시간 측정
    n_repeat = 100

    t0 = time.time()
    for _ in range(n_repeat):
        m.predict(Xte[:1])

    elapsed = time.time() - t0

    # 100회 실행한 1개 샘플 추론의 평균 시간(ms)
    infer_ms = (elapsed / n_repeat) * 1000

    timing_results[k] = {
        "train_time": train_t,
        "infer_ms": infer_ms
    }

    print(
        f"{k}: 학습 {train_t:.3f}s, "
        f"추론 {infer_ms:.3f} ms/sample"
    )

print("\n===== 계산 비용 측정 완료 =====")

## 2. 데이터 및 전처리

### 2.1 데이터셋 구성

본 실습에서는 sEMG-Hand-Gesture-Elbow-Angle-Dataset을 사용하였다. 데이터는 손 동작과 팔꿈치 각도를 조합한 8개 클래스로 구분된다.

| 항목 | 내용 |
|---|---|
| 데이터 유형 | sEMG 신호 |
| 원본 CSV 파일 수 | 64개 |
| 클래스 수 | 8개 |
| 클래스 구성 | 30°·60°와 extor, fist, open, pro의 조합 |
| 학습 데이터 | 969개 샘플 |
| 테스트 데이터 | 247개 샘플 |
| 특징 수 | 25개 |
| 전체 샘플 수 | 1,216개 |

각 클래스는 원본 CSV 파일 기준으로 8개씩 구성되었다. 다만 학습·테스트 샘플 수는 클래스마다 동일하지 않으며, 실제 분류 결과를 해석할 때 이러한 분포를 고려해야 한다.

### 2.2 데이터 전처리

본 실습에서는 `step5_dataset.npz` 파일에 저장된 학습 및 테스트 데이터를 불러왔다. 입력 데이터는 25개의 특징으로 구성되어 있으며, 각 샘플에는 8개 클래스 중 하나의 라벨이 부여되어 있다.

이후 `LabelEncoder`를 이용하여 클래스 라벨을 모델이 처리할 수 있는 정수로 변환하였다. SVM, RF 및 XGB 모델은 동일한 학습 데이터와 테스트 데이터를 사용하여 비교하였다.

이번 보고서에서는 실행 결과로 확인된 데이터의 크기와 라벨 구성을 기록하였다. 저장된 데이터셋을 생성하는 전체 코드가 별도로 검증되지 않았으므로, 세부 특징 추출 및 학습·테스트 분할 방식은 추가 확인이 필요하다.

## 3. 모델 학습 및 성능 평가

### 3.1 비교 모델

본 실습에서는 다음 세 가지 모델을 사용하였다.

| 모델 | 특징 | 장점 및 고려 사항 |
|---|---|---|
| SVM | 클래스 간 경계를 찾는 분류 모델 | 적절한 특징과 커널을 사용하면 효과적이나 설정에 따라 성능이 달라질 수 있음 |
| Random Forest | 여러 결정 트리의 예측을 결합 | 비선형 관계를 학습할 수 있으나 모델 크기와 추론 비용을 고려해야 함 |
| XGBoost | 부스팅 방식으로 트리를 순차적으로 학습 | 복잡한 특징 관계를 학습할 수 있으나 학습 시간과 하이퍼파라미터의 영향을 받음 |

### 3.2 테스트 데이터 성능

모델을 학습한 후 247개 테스트 샘플에 대한 정확도, 정밀도, 재현율 및 F1-score를 측정하였다. 정밀도, 재현율 및 F1-score는 weighted 평균 기준이다.

| 모델 | 정확도 | 정밀도 | 재현율 | F1-score |
|---|---:|---:|---:|---:|
| SVM | 11.34% | 0.1400 | 0.1134 | 0.1052 |
| RF | 8.10% | 0.0931 | 0.0810 | 0.0804 |
| XGB | 11.74% | 0.1226 | 0.1174 | 0.1145 |

테스트 데이터에서는 XGB가 11.74%로 가장 높은 정확도를 기록하였고, SVM은 11.34%, RF는 8.10%를 기록하였다. F1-score 역시 XGB가 0.1145로 가장 높았다.

그러나 세 모델의 정확도는 모두 낮은 수준이다. 클래스가 8개이므로 균형 잡힌 무작위 추측의 기대 정확도는 약 12.5%이다. 실제 테스트 데이터의 클래스 분포와 예측 방식이 무작위 추측과 동일하지는 않지만, 이번 결과는 모델이 클래스의 특징을 충분히 구분하지 못했을 가능성을 보여준다.

특히 RF는 일부 클래스의 재현율이 0으로 나타났다. 이는 해당 클래스의 테스트 샘플을 하나도 올바르게 분류하지 못했다는 의미이다.

## 4. 5-fold 교차검증

### 4.1 실습 방법

5-fold 교차검증은 학습 데이터를 다섯 부분으로 나누고, 네 부분으로 학습한 모델을 나머지 한 부분에서 검증하는 과정을 반복하는 방법이다.

각 폴드에서 세 모델의 정확도를 기록하고, 평균과 표준편차를 계산하였다. 평균은 전반적인 분류 성능을 나타내며, 표준편차는 폴드에 따라 성능이 얼마나 변하는지 보여준다.

### 4.2 폴드별 정확도

| Fold | SVM | RF | XGB |
|---|---:|---:|---:|
| 1 | 36.60% | 60.31% | 52.58% |
| 2 | 40.21% | 57.22% | 55.15% |
| 3 | 34.54% | 47.94% | 50.52% |
| 4 | 34.02% | 45.88% | 39.69% |
| 5 | 39.90% | 56.99% | 49.22% |
| **평균 ± 표준편차** | **37.05 ± 2.60%** | **53.67 ± 5.68%** | **49.43 ± 5.27%** |

교차검증 결과 RF가 평균 정확도 53.67%로 가장 높은 성능을 보였다. XGB는 49.43%, SVM은 37.05%였다.

RF는 가장 높은 평균 정확도를 보였지만 표준편차도 5.68%로 세 모델 중 가장 컸다. 따라서 평균 성능은 우수했으나 폴드에 따라 정확도가 달라지는 정도도 상대적으로 컸다.

### 4.3 테스트 평가와 교차검증의 차이

단일 테스트 데이터 평가에서는 세 모델 모두 약 8~12%의 정확도를 보였지만, 교차검증에서는 약 37~54%의 정확도를 기록하였다.

이 차이는 단순히 모델의 성능이 달라졌다는 의미만으로 해석하기 어렵다. 학습·테스트 데이터의 분할 방식, 클래스별 샘플 구성, 원본 CSV에서 추출된 유사한 윈도우의 포함 여부, 데이터 생성 과정 등을 확인해야 한다.

따라서 최종 모델을 결정하기 전에 `step5_dataset.npz`를 생성한 코드와 원본 파일 단위의 데이터 분할 방식을 검토할 필요가 있다.

## 5. 통계 검정

### 5.1 대응표본 t-검정 방법

본 실습에서는 각 모델의 5개 폴드 정확도를 이용하여 대응표본 t-검정을 수행하였다. 유의수준은 0.05로 설정하였다.

귀무가설은 비교하는 두 모델의 폴드별 정확도 차이의 평균이 0이라는 것이다. p값이 0.05보다 작으면 귀무가설을 기각하고 통계적으로 유의한 차이가 관찰되었다고 해석한다.

### 5.2 검정 결과

| 모델 비교 | t 통계량 | p값 | 결과 |
|---|---:|---:|---|
| SVM vs RF | -8.1210 | 0.0013 | 유의함 |
| SVM vs XGB | -5.9411 | 0.0040 | 유의함 |
| RF vs XGB | 2.1217 | 0.1011 | 유의하지 않음 |

SVM과 RF의 비교에서 p값은 0.0013으로 유의수준 0.05보다 작았다. SVM과 XGB의 비교에서도 p값이 0.0040으로 나타나 유의한 차이가 확인되었다. 두 비교 모두 폴드별 평균 정확도는 RF 또는 XGB가 SVM보다 높았다.

반면 RF와 XGB의 비교에서는 p값이 0.1011로 유의수준보다 높았다. 따라서 두 모델 사이에서 통계적으로 유의한 차이를 검출하지 못했다.

이는 두 모델의 성능이 같다는 것을 입증한 결과가 아니다. 또한 교차검증 폴드들은 학습 데이터가 서로 겹치므로 완전히 독립적인 표본이 아니라는 한계가 있다. 따라서 이번 검정은 실습에서 지정한 방법에 따른 비교 결과로 해석해야 한다.

## 6. 계산 비용 및 최종 모델 선택

### 6.1 측정 방법

각 모델을 학습하는 데 걸리는 시간과 테스트 샘플 하나를 예측하는 데 걸리는 시간을 측정하였다. 추론 시간은 테스트 데이터의 첫 번째 샘플을 100회 예측한 뒤 평균 시간을 밀리초 단위로 계산하였다.

### 6.2 계산 비용 결과

| 모델 | 학습 시간 | 추론 시간 |
|---|---:|---:|
| SVM | 0.038초 | 0.269ms/샘플 |
| RF | 0.176초 | 28.448ms/샘플 |
| XGB | 0.414초 | 0.596ms/샘플 |

SVM은 학습 시간 0.038초와 추론 시간 0.269ms로 가장 빠른 결과를 기록하였다. XGB의 학습 시간은 0.414초로 가장 길었지만, 추론 시간은 0.596ms로 RF보다 훨씬 짧았다.

RF의 샘플당 추론 시간은 28.448ms로 세 모델 중 가장 길었다. 다만 측정 시간은 컴퓨터의 실행 환경과 측정 횟수에 영향을 받으므로, 실제 배포 환경의 성능을 판단하려면 동일한 조건에서 반복 측정해야 한다.

### 6.3 종합적인 모델 선택

정확도, 교차검증 결과, 통계 검정 및 계산 비용을 함께 고려하면 다음과 같다.

- **SVM:** 학습과 추론은 가장 빠르지만 교차검증 평균 정확도가 가장 낮았다.
- **RF:** 교차검증 평균 정확도가 가장 높았지만 추론 시간이 상대적으로 길었다.
- **XGB:** 교차검증 정확도는 RF보다 낮았지만 SVM보다 높았고, 추론 시간은 RF보다 짧았다.

현재 교차검증 결과만으로는 RF가 가장 유력한 후보이다. 하지만 RF와 XGB 사이에 유의한 차이가 검출되지 않았으므로, 계산 비용과 실제 사용 환경을 고려해 추가 검증을 수행하는 것이 적절하다.

## 7. 한계 및 개선 방향

첫째, 단일 테스트 데이터의 정확도가 교차검증 정확도보다 크게 낮았다. 데이터 분할 및 전처리 과정이 일관적인지 확인하고, 테스트 데이터의 클래스별 분포를 분석해야 한다.

둘째, 현재 교차검증은 저장된 특징 샘플을 기준으로 수행되었다. 같은 원본 CSV에서 나온 유사한 윈도우가 학습과 검증 폴드에 동시에 포함되었다면 정확도가 실제보다 높게 측정될 가능성이 있다. 원본 시행 또는 CSV 파일 단위로 그룹을 유지하는 교차검증이 필요한지 확인해야 한다.

셋째, 세 모델 모두 테스트 정확도가 낮았다. 특징 추출 방식, 모델 하이퍼파라미터, 클래스 라벨 생성 및 데이터 분할을 검토하고, 클래스별 혼동행렬을 이용하여 오분류 원인을 분석할 필요가 있다.

넷째, 추론 시간 측정은 단일 샘플을 100회 반복하는 간단한 실험이었다. 실제 서비스 환경에서는 여러 샘플을 대상으로 반복 측정하고, 워밍업 시간, 배치 크기, 하드웨어 사양을 함께 기록해야 한다.

마지막으로, 통계 검정 결과는 다섯 개 폴드의 정확도에 기반한다. 표본 수가 제한적이고 폴드 간 독립성이 완벽하지 않으므로 결과를 일반화할 때 주의해야 한다.

## 8. 결론

본 실습에서는 sEMG 데이터의 8개 클래스를 분류하기 위해 SVM, Random Forest, XGBoost 모델을 비교하였다. 단일 테스트 데이터 평가에서는 XGBoost가 11.74%로 가장 높은 정확도를 보였으며, 5-fold 교차검증에서는 Random Forest가 53.67%로 가장 높은 평균 정확도를 기록하였다.

대응표본 t-검정에서는 SVM과 RF, SVM과 XGB 사이에서 유의한 차이가 관찰되었다. RF와 XGB의 차이는 유의수준 0.05에서 통계적으로 유의하지 않았다.

계산 비용 측정에서는 SVM이 가장 빠른 학습 및 추론 속도를 보였으며, RF는 가장 높은 교차검증 정확도를 기록했지만 추론 시간이 가장 길었다. 따라서 현재 결과에서는 정확도를 우선할 때 RF를 후보로 고려할 수 있지만, 낮은 테스트 정확도와 테스트·교차검증 결과의 차이를 해결하기 전에는 최종 모델의 성능을 확정하기 어렵다.

향후에는 원본 파일 단위의 데이터 분할을 검증하고, 전처리 및 특징 추출 과정을 확인하며, 클래스별 혼동행렬과 추가 실험을 통해 모델의 일반화 성능을 개선할 필요가 있다.

## 참고 자료

- GitHub 저장소: https://github.com/ackack0228-rgb/palm-semg-user-identification
- 데이터셋: sEMG-Hand-Gesture-Elbow-Angle-Dataset
- 사용 모델: SVM, Random Forest, XGBoost
- 주요 평가 방법: 테스트 데이터 평가, 5-fold 교차검증, 대응표본 t-검정, 학습 시간 및 추론 시간 측정

※ 본 보고서의 모델 성능과 시간은 직접 실행한 실습 결과를 기준으로 작성하였다. GitHub README는 보고서 구성 및 재현 실험의 분석 방식 참고 자료로 사용하였다.
