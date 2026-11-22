# Kimodo v1 텍스트 인코더 교체 실험 — 최종 보고서

실행일: 2026-11-22 (로컬 RTX 4090 24GB, Windows 11)
종료 상태: **Stopped** (개발 4회차 후 연속 미개선 3회, `Evaluate` 종료 규칙에 따른 개발 종료)
최종 시험: **미실시** (`Frozen` 후보 없음, 규칙상 최종 시험은 개발 통과 후보에만 1회)
릴리스: **불가** (`Accepted`가 아니므로 앱에서 후보 선택 불가)

## 1. 결론

v1 인수 기준을 만족하는 가중치를 얻지 못했다. 가장 큰 원인은 평가 세트 구조가 아니라 측정 결과 자체에 있다: 4회차 최고 후보(MLP, epochs 200, lr 0.0005)의 전체 지시 이행률이 en 63/192(32.8%), ko 51/192(26.6%)로 80% 합격선(154개)에 크게 못 미친다.

구조적 한계도 확인했다. 동결한 평가 세트의 ③ 행동 순서 유형은 **모든 조건에서 0/48**이다. 영어 원문을 받는 공식 Kimodo+LLM2Vec(ref) 조차 두 행동을 순서대로 수행하는 클립이 없었으므로(사유: `missing_event` 48/48), 이 유형에서 득점할 수 없다. 따라서 이 세트에서 인수 기준 154/192는 나머지 세 유형을 전부 맞혀도 144/192에 그쳐 **수학적으로 도달 불가능**하다. 이 한계는 어댑터 품질과 무관하므로 추가 개발 회차로 Gate를 통과할 수 없다.

## 2. 회차별 측정 (원시 수치, 분모 192)

| 회차 | 후보 | en 합계 | ko 합계 | ref | 랭킹 | 묶음 통과 | 종료 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Linear(768→4096), epochs 48, lr 0.002 | 18 | 10 | 67 | 1953 | cost_en, cost_ko, quality_en, quality_ko, language_gap | 개발 유지 |
| 2 | MLP(1024), epochs 48, lr 0.002 | 54 | 41 | 67 | 1635 | 위에서 language_gap 제외 | 미개선 1 |
| 3 | MLP(1024), epochs 96, lr 0.002 | 62 | 44 | 67 | 1646 | 같음 | 미개선 2 |
| 4 | MLP(1024), epochs 200, lr 0.0005 | 63 | 51 | 67 | 1654 | 같음 | 미개선 3 → Stopped |

4회차 유형별 성공 수(분모 48): ref `[25, 27, 0, 15]`, en `[29, 12, 0, 22]`, ko `[17, 11, 0, 23]`.

## 3. 통과하지 못한 식

`Gate` = `Valid ∧ evidence ∧ standalone ∧ exports ∧ LanguageGap ∧ ∀a∈{en,ko}: Intent(a) ∧ Quality(a) ∧ Cost(a)`

- **Intent(en)**: 63/192 = 32.8% (합격선 80%). ref 34.9% 대비 -2.1p.
- **Intent(ko)**: 51/192 = 26.6%. ref 대비 -8.3p (허용 -5p 초과).
- **LanguageGap**: |32.8% - 26.6%| = 6.25p (허용 5p 초과).
- 유형별 70% 하한: 4회차 en/ko 모두 네 유형 중 0개 통과 (k1 en 60.4%가 최고).
- **Quality(en)**: 188/192 = 97.9% — 통과 (ref 93.75% 대비 +4.2p).
- **Quality(ko)**: 189/192 = 98.4% — 통과 (+4.7p).
- **Cost(en)**: peak 3177 MiB / ref 16773 = 18.9% (≤80% 통과), p95 2602 ms / 2696 = 96.5% (≤120% 통과).
- **Cost(ko)**: peak 3195 MiB = 19.0%, p95 2589 ms = 96.0% — 통과.
- `Valid`, `evidence`, `standalone`, `exports`: 참 (Llama 없는 격리 환경에서 신규 문장 생성, VRM 재생·JSON/BVH/NPZ 내보내기 검증, 동결 해시 일치).

## 4. 진단

- 행동 순서 유형(③): ref 포함 전 조건 0/48. Kimodo가 단일 프롬프트의 복합 지시에서 첫 행동만 수행하거나 어느 행동도 완료하지 않는다. 이 유형이 192개 중 48개를 차지하므로 절대 하한 154를 채울 수 없다.
- 회전 방향(②): 4회차 en 12/48, ko 11/48, ref 27/48. yaw 평균 변위(시드 17, 16개 지시)는 ref가 좌 +53.5°/우 -56.7°인 데 대해 en은 좌 +20.7°/우 -22.1°, ko는 좌 +14.9°/우 -11.7°로 방향은 맞추나 회전량이 약 절반이다.
- 신체 좌우(①): en 29/48 ≥ ref 25/48로 ref를 넘었다. ko 17/48은 뒤처진다(주 실패 사유 `no_sustained_hand_raise` 31회).
- 자세 높이(④): en 22, ko 23 ≥ ref 15로 ref를 넘었다.
- 언어 간 차이: en 63 vs ko 51(6.25p). ko는 손 들기와 회전에서 en보다 각각 12개, 1개 뒤처진다.
- 학습 데이터 감수: 교사 캐시 1104쌍, 어댑터 학습 후 임베딩 코사인 en 0.995/ko 0.994, 노름 128.0 vs 목표 128.8. 학습 피팅은 양호하나 생성 단계에서 회전량·ko 손 들기로 수렴하지 못했다.

## 5. 재현

공통: `HF_HUB_OFFLINE=1`, 동결 자산은 `G:/gi/kimodoMaker-runs/v1-encoder-4090/lock.json`(SHA-256 `0d2931ce78520d28af9faae7532a7faa313e912d7e07e0a064aaa47f30079e87`), dev manifest `d17c5f58...`, SOMA 리비전 `0e3c89024d896064b6851201373f7834784a386b`.

체크포인트(SHA-256):

| 회차 | 경로 | 해시 |
| --- | --- | --- |
| 1 | `adapter-linear-01/adapter.safetensors` | `47fe969911157cfeb3f65a35c061dc5a7ff329bee2cefcfa99dec79293a55299` |
| 2 | `adapter-mlp-02/adapter.safetensors` | `f444314f626f9ad8a7d048f47973fded85113d2f844c28cebba65ae67920f618` |
| 3 | `adapter-mlp-03/adapter.safetensors` | `2eae1720afbfc6284b39ce4dbc0a2816b03c7bab00329fef13fd3acdd0c0e03f` |
| 4 | `adapter-mlp-04/adapter.safetensors` | `d5742dcfbfd41c635d9d0c874937d0ed8112e4cedfbe227b19bcc3a62f9d8a07` |

후보 ID(체크포인트+전처리+데이터+코드 해시): 회차별 44305a22..., 16132e27..., 49c67ba8..., 7807e735... (전체 64자는 각 `run.json`과 `private/report-dev-0N.json`의 `candidate` 필드).

측정(예: 4회차):

```
PYTHONPATH=docs/experiment backend/.venv/Scripts/python.exe docs/experiment/v1_evidence.py measure \
  --lock G:/gi/kimodoMaker-runs/v1-encoder-4090/lock.json \
  --run-root G:/gi/kimodoMaker-runs/v1-encoder-4090/dev-04 --stage Dev --yaw-sign 1 \
  --ref-root G:/gi/kimodoMaker-runs/v1-encoder-4090/dev-01 \
  --export-evidence G:/gi/kimodoMaker-runs/v1-encoder-4090/private/exports-dev-01.json \
  --videos G:/gi/kimodoMaker-runs/v1-encoder-4090/dev-04-blind \
  --out G:/gi/kimodoMaker-runs/v1-encoder-4090/private/report-dev-04.json
```

판정: `... v1_evidence.py decide --report .../report-dev-04.json --state .../state.json`.
판정기 로그: 각 arm의 `requests.jsonl`, NVML 샘플 `sessions/*/nvml.jsonl`, 비교 영상 `dev-0N-blind/`(192 MP4)와 열쇠 파일 `private/dev-video-key-0N.json`.

## 6. 모델 검사 (GPU 실험과 별개)

`EncoderTrial.tla`를 TLC로 검사: 1805상태 생성, 550 고유 상태, 깊이 10, 오류 없음, 지문 충돌 확률 3.7E-14. 결과는 `private/tlc-EncoderTrial.txt`. 판정 자체는 실측 데이터 위의 `v1_score`/`v1_evidence`가 계산한다.

## 7. 다음 실험에 필요한 것

1. 평가 세트 설계: 2026-11-22, 사용자 결정으로 **행동 순서 유형(③)을 다음 실험 대상에서 제외**한다. 근거는 본 보고서 4절의 측정(ref 포함 0/48, Kimodo 자체 한계)이다. 제외 시 분모는 144가 되며 하한은 80%인 116/144로 조정해야 한다. 이 경우 4회차 후보의 단일 동작 성적(유형 ①②④)은 en 63/144(43.8%), ko 51/144(35.4%)로 여전히 하선에 못 미친다.
2. 회전량 절반 문제: 어댑터 출력의 방향은 맞으므로, 노름 보존 외에 회전량에 해당하는 벡터 성분의 스케일을 맞추는 손실(예: 성분별 가중 MSE)이 필요하다.
3. ko 손 들기 지체: 한국어 문장의 측면/지시어 구조가 임베딩에 미치는 영향을 대립쌍 진단으로 좁히고, 백본 부분 미세조정(README 4.3의 다음 단계)을 시도한다.
4. 이 보고서의 결과는 v1 고정 세트에 대한 것이다. 일반 한국어 통계적 비열등성 해석은 별도 표본 설계가 필요하다.

## 8. 사후 교차검증 (2026-11-22, 본 보고서의 유효 범위를 제한함)

v1 종료 후 사용자가 지적한 대로 **원래 잘 작동하던 자연 프롬프트 교차검증을 하지 않았다**. 과거 앱 테이크 기록에서 실제 작동했던 영문 프롬프트 12개(rolling, 가부좌, sideflip, kick, 박수 등)와 자연 문장 5개로 같은 시드에 ref/en/ko 세 arm을 실행했다.

| 프롬프트 유형 | 교사 벡터 코사인(어댑터) | ref와의 모션 거리(루트 정렬 관절 RMS) |
| --- | --- | --- |
| 학습 데이터(합성 1104문장) | 0.995 | — |
| 자연 프롬프트 17개 | **0.45–0.77** | en 3.9–55.0 cm, ko 7.0–61.9 cm |

무엇이 드러났는지:

- 어댑터는 자연 프롬프트에서 교사 임베딩과 코사인 0.45–0.77 수준으로 빗나간다. 노름 비는 0.70–1.02.
- 가장 가까운 학습 문장의 교사 벡터가 진짜 교사 벡터와 이루는 코사인이 예측의 코사인과 거의 같다. 즉 어댑터는 학습 세트에 대한 **최근접 조회표** 수준이며 일반화가 없다.
- 단일 동작("왼팔 곧게 들어 올리세요")조차 코사인 0.77이며, rolling/sit cross-legged/run and sideflip은 30–60cm로 전혀 다른 모션이 생성된다.

이로써 v1의 dev 점수(en 63/192 등)는 **합성 말뭉치 분포 안에서만 의미가 있다**. 원본 Kimodo가 실제로 받는 자연 캡션 분포에 대한 판정 능력은 v1이 측정하지 않았고, 현재 체크포인트는 그 분포에서 사용할 수 없다. 1~4절의 개선 서술은 이 한계 아래에서 읽어야 한다. 4회차 체크포인트의 검증 산출물 위치: 비교 영상 `xval-hist-blind/`, manifest `xval/hist-manifest.jsonl`.

다음 실험(v2)은 이 분포 문제부터 다뤄야 한다: (1) 학습 데이터를 자연 캡션 분포로(BONES-SEED 활용 허가 확인 필요) 또는 합성을 대폭 확장하여 구성, (2) 자연 프롬프트 홀드아웃을 1차 평가로 두기, (3) MSE 외에 분포를 보존하는 손실 검토.
