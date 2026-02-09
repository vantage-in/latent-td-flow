# Latent TD-Flow for Offline GCRL Representation 

## 1. 개요 (Overview)
* **프로젝트명**: Latent TD-Flow: Offline Goal-Conditioned RL을 위한 생성형 상태 표현 
* **학습핵심 아이디어**: 강화학습(RL) 에이전트가 사용할 **Image Encoder**를 학습시키는 것이 목표입니다. **TD-Flow (Temporal Difference Flows)** 방법론을 Latent Space Representation Learning에 맞게 경량화하여 적용합니다.
* **차별점**: 기존 방법들이 Action에 의존적인 반면, 본 방법론은 Action-Free, Reward-Free 설정에서 오직 상태 전이(Dynamics)와 목표 도달 가능성(Reachability)만을 학습하여, 붕괴(Collapse) 없이 미래 지향적인(Long-horizon) Latent Space를 구축하는 것입니다.

## 2. 이론적 배경 (Theoretical Background: TD-Flow & Adaptation)
이 섹션은 TD-Flow 논문의 핵심 개념과 수식을 본 프로젝트(Latent Space & Action-Free)에 맞게 재정의한 것입니다.
### 2.1 Geometric Horizon Model (GHM)
우리가 학습하고자 하는 것은 일반적인 Next-step Dynamics가 아니라, 할인율 $\gamma$가 적용된 미래 방문 상태 분포인 Successor Measure (SM) 입니다.

$$M(s, g) = (1-\gamma) \delta_{s'} + \gamma \mathcal{T} M(s', g)$$

여기서 $\mathcal{T}$는 전이 연산자입니다. 즉, 현재 상태 $s$의 미래 분포는 "바로 다음 상태 $s'$"과 "그 $s'$에서의 미래 분포"의 가중 합으로 정의됩니다.
#### 2.2 Bootstrapped Target Construction 
Flow Matching 모델을 학습시키기 위해 타겟 샘플 $Z$를 생성해야 합니다. TD-Flow는 이를 위해 **Bootstrapping**을 사용합니다.
데이터셋의 전이 $(s, s')$가 주어졌을 때, 타겟 $Z$는 다음과 같은 확률로 결정됩니다:
1. 확률 $1-\gamma$: 실제 데이터의 다음 상태 $z' = \phi(s')$ 그 자체.
2. 확률 $\gamma$: 현재 모델 $v_\theta$를 사용하여 $z'$에서 더 미래로 진행시킨 예측값.
이를 수식으로 표현하면, 타겟 분포 $\Pi_{target}$은 다음과 같습니다:

$$Z \sim (1-\gamma)\delta_{z'} + \gamma \delta_{\tilde{z}_{future}}$$

여기서 $\tilde{z}_{future}$는 현재 학습 중인 Vector Field $v_\theta$를 따라 $z'$에서 시간 $t=1$까지 적분(ODE Solve)한 결과입니다.

$$\tilde{z}_{future} = z' + \int_{0}^{1} v_\theta(z(\tau), \tau | z', z_g) d\tau, \quad \text{where } z(0)=z'$$

### 2.3 Condition Adaptation (Action $\to$ Goal)
원본 TD-Flow 논문에서는 Velocity Field가 $v(z_t, t | s, a)$ 형태, 즉 Action-Conditional로 정의됩니다. 하지만 본 프로젝트에서는 Action을 명시적으로 사용하지 않고, Goal-Conditional 형태로 변경합니다. 
* **Original**: $v_\theta(z_t, t | \text{state}, \text{action})$
* **Ours**: $v_\theta(z_t, t | \text{state}, \text{goal})$

이는 모델이 "특정 행동을 했을 때의 미래"가 아니라, **해당 목표 $g$에 도달하는 궤적 상에 있을 때의 자연스러운 흐름**을 학습하도록 유도합니다.

### 2.4 Coupled TD-CFM (Conditional Flow Matching) ###
학습 시 **Source Noise** ($z_0$)와 Target Data ($Z$) 사이의 경로를 직선(Straight Path)으로 가정하는 Coupled 방식을 사용합니다.
* **Interpolation**: $z_t = t Z + (1-t) z_0$
* **Target Velocity**: $u_t(z_t | z_0, Z) = Z - z_0$

## 3. 알고리즘 및 네트워크 구조 (Algorithm & Architecture)
### 3.1 네트워크 구성
* **Encoder** ($\phi$): Impala CNN (입력 $s \to$ 출력 $z \in \mathbb{R}^D$). 
  * **Default**: **Separate Encoders** (`separate_encoders=True`). State와 Goal에 대해 별도의 Encoder를 사용하여 Metric 불일치 문제를 완화합니다 ($\phi(s) \neq \psi(g)$).
  * **Option**: Siamese Encoder. 만약 데이터셋 특성상 State/Goal 도메인 차이가 크지 않다면 `separate_encoders=False`로 설정하여 파라미터를 공유할 수 있습니다.
* **Flow Model** ($v_\theta$): MLP 구조 (Action 입력 없음).
  * Input: Noised Latent $z_t$, Time $t$, Condition $C = [z_{next}, z_{goal}]$
  * Output: Velocity $\dot{z} \in \mathbb{R}^D$
### 3.2 학습 프로세스 & Loss (Concurrent Training Loop)
배치 데이터 $\mathcal{B} = {(s, s', g_{orig}, \text{done})}$에 대해 다음 과정을 수행합니다.

#### Step 1: Goal Relabeling & Masking
* **Relabeling**: 배치 내의 목표 $g$를 Hindsight Goal($s_{future}$)과 Random Sample로 혼합하여 구성합니다.
* **Masking Strategy**:
    * **Actor & Auxiliary Loss**: 표현력 학습 및 일반화를 위해 전체 배치(Positive + Negative Samples)를 모두 사용합니다.
    * **Flow Loss ($v_\theta$)**: 유효한 궤적에 대한 Velocity Field를 학습해야 하므로, HER로 리레이블링된 Trajectory Sample에 대해서만 마스킹을 적용하여 손실 함수를 계산합니다.

#### Step 2: Latent Encoding & Condition Setup
* $z_s = \phi(s)$
* $z_g = \phi(g)$
* $z' = \text{sg}(\phi(s'))$  (Target Encoder는 Stop Gradient)

#### Step 3: Target $Z$ 생성 (Bootstrapping)
TD-Flow의 핵심인 Bootstrapping을 통해 타겟 $Z$를 생성합니다.
* **확률 $1-\gamma$**: 타겟을 다음 상태인 $z'$으로 설정합니다 ($Z = z'$).
* **확률 $\gamma$**: 현재 모델 $v_\theta$를 사용하여 $z'$에서 더 미래로 진행시킨 예측값 $\tilde{z}_{future}$를 타겟으로 설정합니다.

$$\tilde{z}_{future} = \text{ODE\_Solve}(v_\theta, \text{start}=z', \text{cond}=[z', z_g])$$

* **참고**: $s=g$일 때의 Absorbing State 처리 로직은 선택 사항이며, 자세한 내용은 구현 디테일(4.3절)에서 설명합니다.

#### Step 4: Flow Matching Loss 계산
* Source Noise $z_0 \sim \mathcal{N}(0, I)$ 샘플링.
* Time $t \sim \mathcal{U}[0, 1]$ 샘플링.
* Interpolation: $z_t = t Z + (1-t) z_0$
* Loss: 모델 $v_\theta$가 직선 경로의 속도 $(Z - z_0)$를 예측하도록 학습.
  $$\mathcal{L}_{TD-Flow} = || v_\theta(z_t, t | z_s, z_g) - (Z - z_0) ||^2$$

#### Step 5: Update
Optimizer를 통해 $\phi$와 $v_\theta$를 동시에 업데이트.

## 4. 구현 디테일 (Implementation Details)
#### 1. Flow Model HER 100%의 의미:
Action Condition이 없기 때문에, 데이터셋의 $a'$이 목표 $g$와 무관하다면 모델은 혼란에 빠집니다.
HER을 100%로 설정하면, 모든 $(s, s')$ 전이는 (재설정된) 목표 $g$로 가는 유효한 경로가 되므로, 모델은 **Trajectory Consistency**를 학습하게 됩니다.
#### 2. Auxiliary Losses (Latent Regularization):
Latent Space의 품질 향상을 위해 보조 Loss들을 도입했습니다.
*   **Reward Loss** (`aux_loss_coef=0.1`): $z_s$와 $z_g$를 입력으로 받아 $s=g$ 여부를 예측(BCE)합니다. 이는 Latent Space 상에서 Goal과 Non-Goal 상태를 명확히 구분하도록 돕습니다.
*   **Orthogonality Loss** (`ortho_coef`): Latent Dimension 간의 상관관계를 줄여(Decorrelation) 표현력을 극대화합니다.
*   **Contrastive Loss** (`contrastive_coef`): InfoNCE Loss를 사용하여 $(s, g)$ 페어(Positive)는 가깝게, 그 외(Negative)는 멀어지도록 학습합니다.
#### 3. Absorbing State & Target Encoder:
*   `use_absorbing_state=False` (Default): Absorbing State 로직 대신, 더 간단한 Bootstrapping 방식을 기본으로 사용합니다. 단순화된 로직이 안정적인 학습에 유리할 수 있습니다.
*   `use_target_encoder=True` (Default): 타겟값 계산 시 Target Network(EMA)를 사용하여 학습 안정성을 높입니다.
#### 4. Learning Rate Decay:
Representation LearningPart(`encoder`, `flow_model`)에 대해 Cosine Decay Scheduler를 적용(`lr_decay=True`)하여, 학습 후반부의 안정적인 수렴을 유도합니다.

## 5. 주요 설정 플래그 (Configuration Flags)
`agents/latent_td_flow.py` 및 `agents/latent_td_flow_gciql.py`의 `get_config()`에서 확인 가능한 주요 하이퍼파라미터입니다.

| Flag | Default | Description |
| :--- | :--- | :--- |
| `separate_encoders` | `True` | State와 Goal Encoder 분리 여부. |
| `aux_loss_coef` | `0.1` | Auxiliary Reward Loss (Success Prediction) 가중치. |
| `ortho_coef` | `1e-5` | Orthogonality Regularization 가중치. |
| `contrastive_coef` | `0.05` | Contrastive (InfoNCE) Loss 가중치. |
| `lr_decay` | `True` | Representation Learning 파트에 대한 LR Decay 적용 여부. |
| `decay_steps` | `2e5` | LR Decay가 적용되는 총 스텝 수. |
| `use_absorbing_state` | `False` | Absorbing State ($s=g \implies Z=s$) 로직 사용 여부. |
| `use_target_encoder` | `True` | Target Value 계산 시 Target Encoder (EMA) 사용 여부. |

