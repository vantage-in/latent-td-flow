# Latent TD-Flow for Offline GCRL Representation 

## 1. 개요 (Overview)
* **프로젝트명**: Latent TD-Flow: Offline Goal-Conditioned RL을 위한 생성형 상태 표현 
* **학습핵심 아이디어**: 강화학습(RL) 에이전트가 사용할 **Image Encoder**를 학습시키는 것이 목표입니다. **TD-Flow (Temporal Difference Flows)** 방법론을 Latent Space Representation Learning에 맞게 경량화하여 적용합니다.
* **차별점**: 기존 방법들이 Action에 의존적인 반면, 본 방법론은 Action-Free, Reward-Free 설정에서 오직 상태 전이(Dynamics)와 목표 도달 가능성(Reachability)만을 학습하여, 붕괴(Collapse) 없이 미래 지향적인(Long-horizon) Latent Space를 구축하는 것입니다.

## 2. 이론적 배경 (Theoretical Background: TD-Flow & Adaptation)
이 섹션은 TD-Flow 논문의 핵심 개념과 수식을 본 프로젝트(Latent Space & Action-Free)에 맞게 재정의한 것입니다.
### 2.1 Geometric Horizon Model (GHM)
우리가 학습하고자 하는 것은 일반적인 Next-step Dynamics가 아니라, 할인율 $\gamma$가 적용된 미래 방문 상태 분포인 Successor Measure (SM) 입니다.$$M(s, g) = (1-\gamma) \delta_{s'} + \gamma \mathcal{T} M(s', g)$$
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
  * **Default**: State와 Goal에 가중치 공유(Siamese) 적용 ($z_s = \phi(s), z_g = \phi(g)$).
  * **Backup Plan**: 만약 Metric 불일치가 심하다면, State/Goal Encoder를 분리($\phi_{state} \neq \phi_{goal}$)하는 방안 고려.
* **Flow Model** ($v_\theta$): MLP 구조 (Action 입력 없음).
  * Input: Noised Latent $z_t$, Time $t$, Condition $C = [z_{current}, z_{goal}]$
  * Output: Velocity $\dot{z} \in \mathbb{R}^D$
### 3.2 학습 프로세스 (Concurrent Training Loop)
배치 데이터 $\mathcal{B} = {(s, s', g_{orig}, \text{done})}$에 대해 다음 과정을 수행합니다.
#### Step 1: Hindsight Goal Relabeling (HER 100%) 
* Discriminator나 Reward Predictor를 사용하지 않으므로, 데이터 정합성을 위해 HER 비율을 100%로 설정합니다.
* 즉, 배치의 모든 샘플에서 $g$를 해당 궤적의 미래 상태 $s_{future}$로 덮어씌워, $(s, s')$ 이동이 반드시 $g$를 향한 경로가 되도록 만듭니다.
#### Step 2: Latent Encoding & Condition Setup
* $z_s = \phi(s)$
* $z_g = \phi(g)$
* $z' = \text{sg}(\phi(s'))$  (Target Encoder는 Stop Gradient)

#### Step 3: Target $Z$ 생성 (Bootstrapping with refined Absorbing Logic)
목표에 도달했거나 에피소드가 끝난 경우를 정교하게 처리합니다.
* **Case A: 현재 상태가 이미 목표임 ($s=g$)**
  $$Z = z_s$$ 
  (더 이상 움직이지 않고 현재 상태 유지)
* **Case B: 다음 상태에서 목표 도달 ($s'=g$) 또는 종료 (done)**
  $$Z = z'$$ 
  (다음 스텝까지만 가고 멈춤)
* **Case C: 계속 진행 (Bootstrapping)**
  $$Z = \begin{cases} z' & \text{if } \text{random} < (1-\gamma) \\ \text{ODE\_Solve}(v_\theta, \text{start}=z', \text{cond}=[z', z_g]) & \text{if } \text{random} \ge (1-\gamma) \end{cases}$$
  (주의: ODE Solver는 $z'$에서 출발하여 $t=1$까지 적분)

#### Step 4: Flow Matching Loss 계산
* Source Noise $z_0 \sim \mathcal{N}(0, I)$ 샘플링.
* Time $t \sim \mathcal{U}[0, 1]$ 샘플링.
* Interpolation: $z_t = t Z + (1-t) z_0$
* Loss: 모델 $v_\theta$가 직선 경로의 속도 $(Z - z_0)$를 예측하도록 학습.
  $$\mathcal{L}_{TD-Flow} = || v_\theta(z_t, t | z_s, z_g) - (Z - z_0) ||^2$$

#### Step 5: Update
Optimizer를 통해 $\phi$와 $v_\theta$를 동시에 업데이트.

## 4. 구현 디테일 (Implementation Details)
#### 1. HER 100%의 의미:
Action Condition이 없기 때문에, 데이터셋의 $a'$이 목표 $g$와 무관하다면 모델은 혼란에 빠집니다.
HER을 100%로 설정하면, 모든 $(s, s')$ 전이는 (재설정된) 목표 $g$로 가는 유효한 경로가 되므로, 모델은 **Trajectory Consistency (궤적 일관성)**를 학습하게 됩니다.
#### 2. No Auxiliary Loss (Initial Phase):
Reward/Mask Prediction은 제외합니다.
100% HER과 Absorbing State 로직, 그리고 Siamese 구조가 Latent Space를 자연스럽게 정렬해줄 것으로 기대합니다.
#### 3. Absorbing State Check:
Latent Space 상에서 $s=g$를 판별하기 어려우므로, 데이터셋의 메타데이터(좌표 등)나 Dataset Index를 활용하여 `is_goal(s)`를 정확히 판별해야 합니다.
#### 4. Collapse Monitoring (Backup Plan):
* 학습 중 Latent Vector들의 분산(Variance)을 로깅합니다. 만약 분산이 0으로 수렴(Representation Collapse)한다면, 
  * Discriminator (Classification Loss: $s=g$ vs $s \neq g$)를 추가하여 Regularization을 수행합니다.
  * 또는 VAE처럼 Reconstruction loss나 latent regularization을 직접적으로 수행할 수도 있습니다.
  * 또는 VAE처럼 encoder 학습 시 stochastic을 추가할 수도 있습니다.
* State와 Goal에서 다른 네트워크를 사용할 수도 있습니다. ($\psi$ for Goal)
  * 정보량의 차이: 만약 Goal 이미지는 항상 깨끗한 정면 뷰이고, State 이미지는 노이즈가 심하거나 시점이 꼬인 뷰라면, 서로 다른 특징을 추출해야 할 수도 있습니다. 이럴 땐 네트워크를 분리하여 각자 도메인에 특화된 처리를 하게 할 수 있습니다.
  * 역할의 차이: State는 "현재의 물리적 상황(속도, 장애물 등)"을 다 담아야 하지만, Goal은 "위치 정보"만 중요할 수 있습니다. 네트워크를 분리하면 Goal Encoder가 불필요한 정보를 과감히 버리고 위치 정보만 남기도록 유도할 수 있습니다.
  * Metric 불일치: 가장 큰 문제입니다. 이상적으로는 $s=g$일 때 Latent 거리 $||z_s - z_g||$는 0이 되어야 합니다. 하지만 네트워크가 다르면, 똑같은 이미지를 넣어도 $z_s \neq z_g$가 되어버립니다.
  * 학습 비효율: 비슷한 이미지 처리 능력을 두 번 따로 배워야 하므로 샘플 효율성이 떨어집니다.
* Latent Diffusion을 참고하여 state 등을 normalization할 필요성을 확인해보는 것도 필요합니다.
#### 5. Downstream Task:
RL Agent(Actor-Critic) 학습은 이 루프 안에서 `z.detach()`를 입력으로 받아 Concurrent하게 수행합니다. (Encoder 학습에는 관여하지 않음)


# Scaling Offline Model-Based RL with Action chunking

This repository contains the official implementation of [Scalable Offline Model-Based RL with Action chunking](TODO).

If you use this code for your research, please consider citing our paper:
```
@article{park2025_MAC,
  title={Scalable Offline Model-Based RL with Action Chunking},
  author={Kwanyoung Park, Seohong Park, Youngwoon Lee, Sergey Levine},
  journal={arXiv Preprint},
  year={2025}
}
```

## Overview
This codebase contains implementations of 1) model-free methods, 2) model-based methods, 3) MAC, and ablated version of MAC for ablation studies. Specifically: 

```
# Baselines (Model-free)
from agents.gciql import GCIQLAgent         # GCIQL
from agents.ngcsacbc import NGCSACBCAgent   # n-step GCSAC+BC
from agents.sharsa import SHARSAAgent       # SHARSA

# Baselines (Model-based)
from agents.fmpc import FMPCAgent           # FMPC
from agents.leq import LEQAgent             # LEQ
from agents.mopo import MOPOAgent           # MOPO
from agents.mobile import MOBILEAgent       # MOBILE

# Our method (MAC)
from agents.mac import MACAgent             # MAC
from agents.mbrs_ac import ACMBRSAgent      # MAC (Gau)
from agents.mbfql import MBFQLAgent         # MAC (FQL)
from agents.model_ac import ACModelAgent    # Model inaccuracy analysis
```

## Installation

Please install the libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Datasets

For downloading the datasets, please follow the instruction of [Horizon Reduction Makes RL Scalable](https://github.com/seohongpark/horizon-reduction).

## Example training scripts 

For MVE-based MBRL algorithms (MAC, LEQ, FMPC) and model-free RL algorithms (SHARSA, GCIQL, n-step GCSAC+BC):

```
# MAC in puzzle-4x5-play-oraclerep-v0 (100M)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/mac.py

# SHARSA in puzzle-4x5-play-oraclerep-v0 (100M)
python main.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/sharsa.py
```

For MBPO-based algorithms (MOPO, MOBILE):

```
# MOPO in puzzle-4x5-play-oraclerep-v0 (100M)
python main_mbpo.py --env_name=puzzle-4x5-play-oraclerep-v0 --dataset_dir=<YOUR_DATA_DIRECTORY>/puzzle-4x5-play-100m-v0 --agent=agents/mopo.py
```

## Acknowledgement

This codebase is built on top of [Horizon Reduction makes RL scalable](https://github.com/seohongpark/horizon-reduction)'s codebase.
