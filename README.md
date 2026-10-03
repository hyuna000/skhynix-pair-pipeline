# 검정 파이프라인 (연구계획 v3, 7절 Step 1~4 + 게이트)

SKHYNIXUSDT(본주 퍼프)·SKHYUSDT(ADR 퍼프) 1분봉으로, 매 거래일 T0 이전 윈도우에서
**Step 1 (I(1) 검정) → Step 2 (헤지비율 4종) → Step 3 (스프레드 ADF·KSS 게이트) → Step 4 (모수·반감기)** 를
8개 변형(후보 4 × V1/V2)에 대해 돌리고 `daily_decision_log` 를 남긴다.
임계값·매매·비용(9·10절)은 아직 없다. 그래서 `decision` 은 비용 필터 전 단계 판정이다.

## 실행 (Windows cmd 기준)

```
pip install -r requirements.txt
cd stat_pipeline
python -m pytest -q tests                 :: 달력·통계량 검증 (10개)
python run_gate.py                        :: 실제 데이터, 계획서 규칙 그대로
python run_gate.py --ignore-coverage      :: 결측 10% 초과 윈도우도 통계량 계산 (결정은 DATA_BAD 유지)
python make_synthetic.py
python run_gate.py --synthetic            :: 정답을 아는 합성 쌍으로 작동 확인
python run_gate.py --start 2026-09-30 --end 2026-10-01 --B 99
```

원자료는 `data/raw/` 에 `SKHYNIXUSDT_YYYY-MM-DD.parquet`, `SKHYUSDT_YYYY-MM-DD.parquet` 로 둔다
(이름은 자유, 심볼은 파일 안의 `instrument_raw_symbol` 로 구분).
B=199 기준으로 1,440분 윈도우 하나에 약 15초, 2,880분 윈도우는 약 30초 걸린다(코어 1개).

## 데이터 넣는 곳 (깃에는 없음)

원자료는 용량과 보안 때문에 깃에 올리지 않는다(.gitignore). 팀 공유 드라이브에서 받아 아래에 넣는다.
- `data/raw/` : bar_1m parquet (`SKHYNIXUSDT_YYYY-MM-DD.parquet`, `SKHYUSDT_YYYY-MM-DD.parquet`)
- `data/raw_orderbook/` : order_book parquet (있으면 분 중간가를 자동 사용)
실행 결과 `logs/`, 가공 데이터 `data/processed*` 도 깃에 없고 실행하면 다시 생긴다.

## 구조

| 파일 | 계획서 | 역할 |
|---|---|---|
| `config/config.toml` | 전체 | 유의수준, 휴일, B, 시드, 래그 규칙. 해시가 로그에 기록됨 |
| `schema/daily_decision_log.json` | 12절 | 로그 컬럼 정의. 맞지 않으면 기록 거부 |
| `src/ingest.py` | 3절 | Decimal 파싱, 중복 제거, KST 변환, 사용 가능 시각, 품질 리포트 |
| `src/tradecal.py` | 4·5·6절 | 정상 블록, T0(토 09:00), V1/V2 윈도우, 주말 A안, 휴일 처리, 레짐 라벨 |
| `src/unitroot.py` | 7절 | ADF, ADF-GLS, KSS, KPSS 통계량 + 래그 선택 (순수 함수, 14절에서 재사용) |
| `src/hedge.py` | 2절 Step 2 | OLS, TLS, DOLS(BIC 리드·래그), 구조적 b=1, HAC 신뢰구간 |
| `src/bootstrap.py` | 부록 A-1·A-2·A-4 | sieve wild (단변량), KPSS 용, VAR sieve wild (쌍) |
| `src/steps.py` | 7절 | Step 1 네 조건, Step 3 게이트 표, Step 4 모수 |
| `src/gate_runner.py` | 7절 | 거래일×버전 윈도우 자르기, 같은 윈도우 재사용, 결정·사유 |
| `src/logger.py`, `src/audit.py` | 12절 | 추가 전용 로그, 인과성·재현·키 유일성 감사 |
| `run_gate.py` | | 실행 진입점 |
| `make_synthetic.py` | | ADR 없을 때·검증용 합성 쌍 (정답표 `data/synthetic/truth.csv`) |
| `simulate.py` | 14절 | 크기·검정력 시뮬레이션 (warp-speed), 윈도우 길이별 |

## 판정 순서 (`no_trade_reason`)

`DATA_BAD`(ADR 없음 또는 결측 >10%) → `HOLIDAY_RULE`(쓸 정상 블록 없음) → `STEP1_UNDETERMINED` →
`BAD_HEDGE`(b ≤ 0) → `GATE_FAIL`(ADF·KSS 모두 비기각) → `HALFLIFE_FILTER`(반감기 > 120분) → `TRADE`.
`STEP1_UNDETERMINED`, `HALFLIFE_FILTER` 는 계획서 사유 목록에 없어 새로 붙인 이름이다.

## 계획서를 코드로 옮기며 정한 것 (연구자 확인 필요)

1. **Step 1 은 거래일×버전마다** 한 번 (V1·V2 윈도우가 달라서). 4개 후보가 공유한다.
2. **결측 분은 버리고 이어 붙인다** (두 심볼 모두 있는 분만). 10% 이하 결측일 때만 판정에 쓰이므로 영향은 작다.
3. **RMAIC** 는 MAIC 의 tau 를 이분산 보정형 `b0²(Σy²)² / Σ(e²y²)` 로 바꾼 형태로 구현했다.
   Cavaliere-Phillips-Smeekes-Taylor(2015) 원식과 대조하지 못했다.
4. **KSS 래그** 는 같은 시리즈의 선형 ADF 에서 고른 k 를 그대로 쓴다.
5. **TLS** 는 평균 제거 후 직교회귀. TLS 의 b 신뢰구간(A-4 비귀무 부트스트랩)은 아직 없다(NaN).
6. **화요일 V2** 는 일·월 블록이 연속 정상 블록이 아니라서 항상 V1 으로 대체된다 (6절 규칙 그대로).
7. **후보 1(OLS)** 게이트 p 는 MacKinnon EG 표, 부트스트랩 p 는 `adf_p_boot` 에 대조용으로 기록.

## Step 1 설정 스위치 (config `[step1]`, `[data_quality]`)

| 키 | 계획서 값 | 대안 | 비고 |
|---|---|---|---|
| `enforce_missing_rule` | true | **false (2026-10-03 팀 결정)** | 결측 비율은 계속 기록. 최소 길이 `min_bars=240` 은 유지 |
| `kpss_b_method` | bootstrap | table | 조건 (b) 판정 방식 |
| `diff_lag_criterion` | rmaic | bic | 조건 (c) 래그 규칙 |
| `adfgls_c_method` | bootstrap | table | 조건 (c) 판정 방식 |
| `gate_conditions` | "abcd" | "abd" | 게이트에 쓰는 조건. 빠진 조건도 기록은 됨 |

`config/config_step1_table.toml` = 위 대안을 모두 적용한 설정 (`python run_gate.py --config config/config_step1_table.toml`).

## 시뮬레이션 근거 (T=1,440, 참 I(1) = 랜덤워크+GARCH+호가 바운스, 박스권 = 정상 AR(1))

(b) 수준 KPSS 10% 기각률 (I(1) 에서는 높을수록 좋음), R=200, B=199

| DGP | table | boot cap .995 | boot cap .98 | boot cap .95 |
|---|---|---|---|---|
| 참 I(1) | **0.99** | 0.44 | 0.70 | 0.89 |
| 박스권 반감기 60분 | 0.82 | 0.16 | 0.23 | 0.47 |
| 박스권 반감기 240분 | 0.94 | 0.23 | 0.54 | 0.79 |
| 박스권 반감기 720분 | 0.98 | 0.39 | 0.72 | 0.87 |

- 부트스트랩 (b) 는 참 I(1) 의 56% 를 탈락시킨다. table 은 1%.
- 어느 방식도 하루 창에서 박스권과 I(1) 을 잘 구별하지 못한다 (구별력은 데이터 길이의 한계).
- (a) 수준 ADF-GLS 1% (부트스트랩, R=100): 반감기 60분·240분 박스권도 100% 통과. 차분 기반·잔차 기반 sieve 모두 같음.
  박스권 반감기 60분의 ADF-GLS 중앙값 -2.1~-2.4 로 1% 임계값 -2.58 에 못 미침 -> **Step 1 은 하루 창에서 박스권을 걸러내지 못한다.**
- (c): 실제 분 수익률에서 BIC 도 9~25 래그를 골라 통계량이 -1.2~-2.1 에 그침 (래그 2 고정이면 -4.8~-9.0).
  수집 공백 구간의 수익률을 빼도 같음. (d) 가 같은 질문(수익률 정상성)을 맡고 모든 창에서 통과하므로 (c) 는 기록만.

## 현재 기본 설정 (2026-10-03 결정 반영, `config/config.toml`)

| 항목 | 계획서 원안 | 현재 | 설정 키 |
|---|---|---|---|
| 결측 기준 | 10% 초과 DATA_BAD | 사용 안 함 | `enforce_missing_rule = false` |
| 최소 연속구간 | 없음 | 5분 이하 간격은 이어진 것으로 보고, **720분 이상 구간만** 사용 | `max_gap_min`, `min_segment_min` |
| T0 직전 분봉(07:59) | 사용 | **제외** (T0 이후 도착) | `exclude_last_bar = true` |
| 윈도우 | 정상 블록만, 주말 A안 | T0 직전 연속 2,880분, 주말·공휴일 포함 | `window_mode = "continuous"` |
| 버전 | V1 + V2 (8변형) | V2 만 (후보 4개) | `versions = ["V2"]` |
| 토요일 T0 | 09:00 | 09:00 (유지) | `saturday_start_hour` |
| Step 1 게이트 조건 | a·b·c·d | **a·b·d** ((c) 는 기록만) | `gate_conditions = "abd"` |
| Step 1 KPSS (b)(d) | 부트스트랩 | KPSS(1992) 점근 표 | `kpss_b_method`, `kpss_d_method` |
| 래그 선택 | RMAIC | RMAIC = 원논문 RSMAIC 로 재구현 | `[lags] criterion = "rmaic"` |
| Step 3 게이트 | ADF 또는 KSS (or) | **ADF 버전, KSS 버전을 각자 5% 기준으로 따로 운영** (팀 결정, 결합 검정 안 함) | `gate_rules = ["adf", "kss"]` |
| 반감기 | AR(1) | **ARMA(1,1)** (호가 잡음 보정). AR(1) 값도 기록 | `half_life_method = "arma11"` |
| 가격 | 분 중간가 (오더북) | 중간가 있으면 중간가, 없으면 체결가 종가 | `raw_orderbook_dir`, `mid_dir` |
| TLS | | 그대로 | |

### RSMAIC (Cavaliere, Phillips, Smeekes & Taylor 2015, Econometric Reviews 34(4), 식 5-7)
1) 상수 제거 시리즈에 kmax 차 ADF 회귀 -> 잔차, 2) 잔차 제곱을 가우시안 커널(h=0.1)로 평활해 변동성 경로 추정,
3) 차분을 변동성으로 나눠 누적한 재척도 시리즈 생성, 4) 그 시리즈에 MAIC 적용. 선택은 OLS 추세제거 데이터에서 (Perron-Qu 2007).
이전 구현(tau 를 이분산 보정형으로 바꾼 것)은 원논문과 달라 폐기. 확인: 변동성이 4배로 뛰는 랜덤워크(참 래그 0)에서 MAIC 10, RSMAIC 0.

### 오더북 (src/orderbook.py)
depth@100ms 변경분만 있고 시작 스냅샷이 없어 빈 호가창에서 재구성. 시퀀스가 끊기면 새로 시작, 60초 워밍업.
검증(SKHYUSDT 09-18, 19:28~23:52): 스프레드 1틱(0.54bp), 분봉 종가의 82% 가 재구성 bid~ask 안, 나머지도 1~3틱 이내.
현재 오더북은 ADR 4.4시간뿐이라 검정에 쓰인 구간과 겹치지 않음 (mid_frac = 0).
실제 호가 잡음은 작다(종가-중간가 중앙값 0.27bp, 반 틱). 반감기 편향은 합성 데이터에서 본 것보다 작다:
실제 데이터 ARMA 반감기가 AR(1) 보다 6~60% 길다.

### RSMAIC 적용 후 시뮬레이션 (T=2,880, R=300)
게이트(ADF 또는 KSS) 랜덤워크 오통과 0.13(부트스트랩) / 0.11(표). KSS 부트스트랩 크기가 0.10 으로 5% 를 넘는다.
검정력: 반감기 60분 0.99, 120분 0.73. Step 1 (a)+(b표) 참 I(1) 통과 0.99.

## 게이트 규칙 비교 (`python sim_gate_rules.py`, T=2,880)

랜덤워크 오통과 (R=2,000): or 9.0% / 각 2.5% 4.2% / adf 단독 5.6% / **minp 5.6%** / kss 단독 6.0%.
앞서 README 에 적었던 "11~13%" 는 R=300 시뮬레이션의 우연 오차였다. 원인은 거의 전부 "또는" 구조.

검정력 (R=400, +-2~3%p)

| 스프레드 | or | 각 2.5% | adf 단독 | minp | kss 단독 |
|---|---|---|---|---|---|
| OU 반감기 60분 | 0.995 | 0.955 | 0.995 | 0.982 | 0.675 |
| OU 반감기 120분 | 0.635 | 0.475 | 0.575 | 0.522 | 0.362 |
| OU 반감기 240분 | 0.310 | 0.142 | 0.208 | 0.220 | 0.222 |
| ESTAR 강 | 0.998 | 0.998 | 0.995 | 0.998 | 0.982 |
| ESTAR 약 | 0.778 | 0.645 | 0.522 | 0.680 | 0.632 |

로그에는 거래일 x 후보 x `gate_rule` 마다 한 줄. minp 는 두 검정 모두 부트스트랩 p 를 쓰고(후보 1 도),
ADF 와 KSS 가 같은 재표본을 공유한다 (후보 1~3 은 A-4 VAR sieve, 후보 4 는 A-1).
현재 설정은 adf, kss 두 버전 -> 변형 4개(후보) x 2개 = 8개. 11절 다중검정 보정(Romano-Wolf) 대상도 8개.
두 버전의 결과를 합쳐 "어느 쪽이든 통과하면 거래"로 쓰면 다시 or 규칙(오통과 9%)이 되므로 각자 따로 보고한다.

## 윈도우 길이·윈도우 방식·p 값 출처 비교 (2026-10-03)

설정 스위치: `[calendar] window_mode` (blocks | continuous), `[step1] pvalue_mode`, `[step3] pvalue_mode` (plan | bootstrap | table).
비교용 설정 4개: `config/compare/{blocks,continuous}_{bootstrap,table}.toml` (모두 Step 1 조건 a·b·d, (c) 는 기록만).

### 시뮬레이션 (`python simulate.py`, warp-speed, R=500) -> reports/sim_step1.csv, sim_step3.csv

Step 3 게이트 통과율 (ADF 또는 KSS 5% 기각)

| 스프레드 | 1440 부트 | 1440 표 | 2880 부트 | 2880 표 |
|---|---|---|---|---|
| 랜덤워크 (오통과, 낮을수록 좋음) | 0.08 | 0.08 | 0.10 | 0.09 |
| OU 반감기 30분 | 1.00 | 0.99 | 1.00 | 1.00 |
| OU 반감기 60분 | 0.67 | 0.70 | 0.99 | 0.99 |
| OU 반감기 120분 | 0.33 | 0.32 | 0.73 | 0.78 |
| OU 반감기 240분 | 0.20 | 0.17 | 0.36 | 0.31 |

Step 1 (a)+(b) 통과율

| 가격 | 1440 부트 | 1440 표 | 2880 부트 | 2880 표 |
|---|---|---|---|---|
| 참 I(1) (높을수록 좋음) | 0.47 | 0.97 | 0.64 | 0.99 |
| 박스권 반감기 60분 (낮을수록 좋음) | 0.12 | 0.55 | 0.02 | 0.20 |
| 박스권 반감기 240분 | 0.26 | 0.95 | 0.26 | 0.89 |

- 2,880분이면 반감기 2시간 이하 스프레드의 게이트 검정력이 약 2배.
- 게이트는 ADF·KSS 중 하나만 기각해도 통과라서 랜덤워크 오통과가 5% 가 아니라 8~10%.
- Step 3 은 부트스트랩과 표의 차이가 작다. Step 1 은 부트스트랩이 참 I(1) 을 많이 탈락시킨다.

### 실제 데이터 (reports/compare_4configs.csv)

| 설정 | 버전 | 검정 가능 창 | Step 1 통과 | 게이트 통과 (ols/tls/dols/struct) |
|---|---|---|---|---|
| blocks / bootstrap | V1 | 6 | 2 | 0/0/0/0 |
| blocks / bootstrap | V2 | 11 | 1 | 0/0/0/0 |
| blocks / table | V1 | 6 | 6 | 0/1/0/1 |
| blocks / table | V2 | 11 | 11 | 0/2/0/0 |
| continuous / bootstrap | V1 | 9 | 4 | 0/1/0/2 |
| continuous / bootstrap | V2 | 12 | 6 | 0/1/0/3 |
| continuous / table | V1 | 9 | 9 | 2/3/2/3 |
| continuous / table | V2 | 12 | 12 | 3/5/3/3 |

주의: 통과 창 다수가 결측 40~77% 창이다 (공백을 이어 붙여 검정). 실제 데이터 10일로는 방식 간 우열을 판단할 수 없고,
판단 근거는 위 시뮬레이션이다. KSS 표 p 는 0.01~0.10 범위로 잘린다.

## 알려진 문제

- **게이트 오통과 11~13%** (ADF 또는 KSS 규칙 + KSS 부트스트랩 과대 크기). 2026-10-03 결정으로 규칙은 유지.
- **Step 1 이 박스권 가격을 걸러내지 못한다** (위 시뮬레이션). 실질적인 걸러내기는 Step 3 게이트와 반감기 필터가 한다.
- **반감기:** AR(1) 은 호가 잡음에 과소추정 -> ARMA(1,1) 로 보정 (합성: 참값 45분, AR(1) 25.5 -> ARMA 42.5).
- **결측 기준 해제 후:** 공백이 있는 창은 관측된 분만 이어 붙여 검정한다. 수 시간 공백을 한 분으로 취급하는 셈이라, 공백이 큰 창의 결과는 `missing_frac_paired` 와 함께 해석할 것.
- **수신 지연:** 07:59 분봉은 T0 이후 0.1~3초에 수신된다. 감사에서 경고로 보고.
