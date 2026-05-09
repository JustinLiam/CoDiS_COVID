import torch
import numpy as np
import pandas as pd
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import math
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可重复实验
RNG_SEED = 42

# 因果样本：对未来结局的观测长度（日）。
# （1）流行病学：死亡经「感染→重症→住院→死亡」有明显滞后，适度延长窗使 β 的差异能反映到 D 上。
# （2）数值：若仅传入 t=[0, H] 两个端点，rk4 在整段 [0,H] 上步长过大，易出现非物理解或使不同 β 下 D 反常相同；
#    故未来积分使用 0..H 的逐日网格，与背景池 t_pool 的离散方式一致。
CAUSAL_FUTURE_DAYS = 30

# Bernoulli 概率：抽样事实处理 a_factual == 1（对应较低 β）。
CAUSAL_TREATMENT_ONE_PROB = 0.4

# 因果两臂 β：基线拉大 → 同一时间窗内因传播差异造成的死亡增量更易分离；仍可配合 CAUSAL_FUTURE_DAYS。
# 「感染压力」用 (E..Is)/N 裁剪后归一化到 [0,1]，使 relaxed 在高流行时 β 更高、strict 略低，
# 从而拉开样本间 |ΔD|（缓解 ITE 挤在 0 附近）。
# 仍可继续调：提高 STRICT/RELAXED 落差、PRESSURE_CLIP（压力饱和尺度）、*_GAIN 或 FUTURE_DAYS。
CAUSAL_BETA_STRICT_BASE = 0.18
CAUSAL_BETA_RELAXED_BASE = 0.60
CAUSAL_BETA_RELAXED_CAP = 0.92
# 压力比 (E+Ia+Ip+Im+Is)/pop 的上界裁剪，再归一化到 [0,1]
CAUSAL_BETA_PRESSURE_CLIP = 0.12
CAUSAL_RELAXED_PRESSURE_GAIN = 1.05
CAUSAL_STRICT_PRESSURE_GAIN = 0.45


def _normalized_infection_pressure(y_anchor: torch.Tensor, pop: float) -> float:
    latent_plus_inf = torch.sum(y_anchor[1:6]).item()
    ratio = latent_plus_inf / float(pop)
    ratio = max(0.0, min(ratio, CAUSAL_BETA_PRESSURE_CLIP))
    den = CAUSAL_BETA_PRESSURE_CLIP if CAUSAL_BETA_PRESSURE_CLIP > 0 else 1.0
    return ratio / den


def _beta_for_treatment_arm(is_strict: bool, pressure_n: float) -> float:
    p = pressure_n
    if is_strict:
        b = CAUSAL_BETA_STRICT_BASE * (1.0 - CAUSAL_STRICT_PRESSURE_GAIN * p)
        return max(b, 0.06)
    b = CAUSAL_BETA_RELAXED_BASE * (1.0 + CAUSAL_RELAXED_PRESSURE_GAIN * p)
    return min(b, CAUSAL_BETA_RELAXED_CAP)


# -----------------------------------------------------------------------------
# 因果推断与数据集设计说明（简述）
#
# 【重叠性与城市耦合】若在某一层（例如州）上 Treatment 对所有个体近似确定性（如「加州恒为 0」），
# 则在该层内部的非参数识别不满足经典共同支撑，无法仅用该亚总体数据「无假定」推断 T=1 的反事实；仍可依
#（i）可传输到外层的模型外推；（ii）结构化疾病模型；（iii）在州（城）内保持随机指派或非平凡倾向得分
# e(X) ∈ (ε, 1 − ε)，以恢复重叠假设。
#
# 【反事实在轨迹上可分】β 经传播链影响结局；预测窗应与滞后匹配，且 odeint 须有足够时间离散（与本脚本
# 背景池一致的逐日网格），勿仅用 [0, H] 两个时间点（rk4 整段大步易数值病态，曾导致 y_f=y_cf 的假零效应）。
# -----------------------------------------------------------------------------


def set_global_seed(seed: int = RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 定义状态列名（共10维）
STATE_NAMES = [
    'Susceptible', 'Exposed', 'Infectious_asymptomatic', 
    'Infectious_pre-symptomatic', 'Infectious_mild', 'Infectious_severe', 
    'Hospitalized_recovered', 'Hospitalized_deceased', 'Recovered', 'Deceased'
]

# 2. 修改后的随机干预分配函数
def get_random_mask_policy():
    return 1 if random.random() < CAUSAL_TREATMENT_ONE_PROB else 0

def load_city_data():
    df = pd.read_csv('sub-est2022.csv')
    # link = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2022/cities/totals/sub-est2022.csv"
    # df = pd.read_csv(link)
    
    cities_df = df[
        (df['SUMLEV'] == 162) & 
        (df['ESTIMATESBASE2020'] >= 150000) &
        (df['ESTIMATESBASE2020'] < 200000)
    ].sort_values('ESTIMATESBASE2020', ascending=False)
    
    strict_states = [
        'California', 'New York', 'Illinois', 'Washington', 'Oregon',
        'Massachusetts', 'New Jersey', 'Connecticut', 'Rhode Island',
        'Delaware', 'Hawaii', 'Maryland', 'Michigan', 'Nevada',
        'New Mexico', 'Vermont', 'Virginia'
    ]
    
    strict_cities = [
        'New York city', 'Los Angeles city', 'Chicago city', 'San Francisco city',
        'Seattle city', 'Boston city', 'Portland city', 'San Jose city',
        'Denver city', 'Minneapolis city', 'Philadelphia city', 'Sacramento city',
        'San Diego city', 'Oakland city', 'Long Beach city'
    ]
    
    # def get_mask_policy(row):
    #     if row['NAME'] in strict_cities or row['STNAME'] in strict_states:
    #         return 'Strict'
    #     return 'Relaxed'

    # Strict / Relaxed 随机分配比例与 CAUSAL_TREATMENT_ONE_PROB 对齐，便于对照实验。
    def get_mask_policy(row):
        return 'Strict' if random.random() < CAUSAL_TREATMENT_ONE_PROB else 'Relaxed'
    
    cities_df['mask_policy'] = cities_df.apply(get_mask_policy, axis=1)
    
    def assign_parameters(row):
        if row['mask_policy'] == 'Strict':
            delta = 0.15
            alpha = 0.3
        else:
            delta = 0.1
            alpha = 0.5
        return pd.Series({'delta': delta, 'alpha': alpha})
    
    parameter_df = cities_df.apply(assign_parameters, axis=1)
    cities_df = pd.concat([cities_df, parameter_df], axis=1)
    
    output_df = cities_df[['NAME', 'STNAME', 'ESTIMATESBASE2020', 'mask_policy', 'delta', 'alpha']]
    output_df.to_csv('large_cities_with_policies.csv', index=False)
    
    print(f"Total cities analyzed: {len(cities_df)}")
    print("\nMask policy distribution:")
    print(cities_df['mask_policy'].value_counts())
    
    return cities_df


def generate_initial_conditions():
    cities_df = load_city_data()
    populations = cities_df['ESTIMATESBASE2020'].tolist()
    policies = cities_df['mask_policy'].tolist()
    deltas = cities_df['delta'].tolist()
    alphas = cities_df['alpha'].tolist()
    
    initial_conditions = []
    normalized_populations = []
    for pop in populations:
        exposed = int(pop * 0.0015)
        ia = int(pop * 0.001)
        ip = int(pop * 0.0007)
        im = int(pop * 0.0005)
        is_ = int(pop * 0.0002)
        hr = int(pop * 0.00001)
        hd = int(pop * 0.000005)
        r = int(pop * 0.000005)
        d = int(pop * 0.000001)
        s = pop - (exposed + ia + ip + im + is_ + hr + hd + r + d)
        
        y0 = [s, exposed, ia, ip, im, is_, hr, hd, r, d]
        initial_conditions.append(torch.tensor(y0, dtype=torch.float32, device=DEVICE))
        normalized_populations.append(pop)
    
    return initial_conditions, policies, deltas, alphas, normalized_populations


# class ODEFunc(nn.Module):
#     def __init__(self, population, min_pop, max_pop):
#         super(ODEFunc, self).__init__()
#         self.N = torch.tensor(float(population), device=DEVICE)
        
#         self.population_factor = 0.8 + (population - min_pop) * (0.2) / (max_pop - min_pop)
#         self.Ca = nn.Parameter(torch.tensor(0.425, device=DEVICE))
#         self.Cp = torch.tensor(1.0, device=DEVICE)
#         self.Cm = torch.tensor(1.0, device=DEVICE)
#         self.Cs = torch.tensor(1.0, device=DEVICE)
#         self.alpha = torch.tensor(0.4875, device=DEVICE)
#         self.delta = nn.Parameter(torch.tensor(0.1375, device=DEVICE))
#         self.mu = torch.tensor(0.928125, device=DEVICE)
#         self.gamma = torch.tensor(1/3.5, device=DEVICE)
#         self.lambdaa = torch.tensor(1/7, device=DEVICE)
#         self.lambdap = torch.tensor(1/1.5, device=DEVICE)
#         self.lambdam = torch.tensor(1/5.5, device=DEVICE)
#         self.lambdas = torch.tensor(1/5.5, device=DEVICE)
#         self.rhor = torch.tensor(1/15, device=DEVICE)
#         self.rhod = torch.tensor(1/13.3, device=DEVICE)
#         self.beta = nn.Parameter(torch.tensor(0.5, device=DEVICE))
#         self.t = torch.tensor(0.0, device=DEVICE)

#     def set_beta(self, beta_val):
#     # 使用 with torch.no_grad() 确保这种手动修改不被计入梯度图
#         with torch.no_grad():
#             # copy_ 是原位（in-place）操作，修改现有 tensor 的内容
#             self.beta.copy_(torch.tensor(beta_val, dtype=torch.float32, device=DEVICE))

#     # def set_beta(self, beta):
#     #     self.beta = nn.Parameter(torch.tensor(beta, device=DEVICE))


#     def forward(self, t, y):
#         S, E, Ia, Ip, Im, Is, Hr, Hd, R, D = y
#         dt = t - self.t
#         self.t = t
#         dSE = self.population_factor * S * (1 - torch.exp(-self.beta * (self.Ca * Ia + self.Cp * Ip + self.Cm * Im + self.Cs * Is) * dt / self.N))
#         dEIa = E * self.alpha * (1 - torch.exp(-self.gamma * dt))
#         dEIp = E * (1 - self.alpha) * (1 - torch.exp(-self.gamma * dt))
#         dIaR = Ia * (1 - torch.exp(-self.lambdaa * dt))
#         dIpIm = Ip * self.mu * (1 - torch.exp(-self.lambdap * dt))
#         dIpIs = Ip * (1 - self.mu) * (1 - torch.exp(-self.lambdap * dt))
#         dImR = Im * (1 - torch.exp(-self.lambdam * dt))
#         dIsHr = Is * self.delta * (1 - torch.exp(-self.lambdas * dt))
#         dIsHd = Is * (1 - self.delta) * (1 - torch.exp(-self.lambdas * dt))
#         dHrR = Hr * (1 - torch.exp(-self.rhor * dt))
#         dHdD = Hd * (1 - torch.exp(-self.rhod * dt))

#         dS = -dSE
#         dE = dSE - dEIa - dEIp
#         dIa = dEIa - dIaR
#         dIp = dEIp - dIpIs - dIpIm
#         dIm = dIpIm - dImR
#         dIs = dIpIs - dIsHr - dIsHd
#         dHr = dIsHr - dHrR
#         dHd = dIsHd - dHdD
#         dR = dHrR
#         dD = dHdD
      
#         dy = torch.stack([dS, dE, dIa, dIp, dIm, dIs, dHr, dHd, dR, dD])
#         return dy
    
#     def reset_t(self):
#         self.t = torch.tensor(0.0, device=DEVICE)


class ODEFunc(nn.Module):
    def __init__(self, population, min_pop, max_pop):
        super(ODEFunc, self).__init__()
        self.N = torch.tensor(float(population), device=DEVICE)
        
        # 修正：将常量定义为 Buffer，确保它们随模型移动到正确的设备，但不作为参数优化
        self.register_buffer('Cp', torch.tensor(1.0))
        self.register_buffer('Cm', torch.tensor(1.0))
        self.register_buffer('Cs', torch.tensor(1.0))
        self.register_buffer('mu', torch.tensor(0.928125))
        # ... 其他常量同理 ...

        self.population_factor = 0.8 + (population - min_pop) * (0.2) / (max_pop - min_pop)
        
        # 优化参数
        self.Ca = nn.Parameter(torch.tensor(0.425, device=DEVICE))
        self.alpha = nn.Parameter(torch.tensor(0.4875, device=DEVICE))
        self.delta = nn.Parameter(torch.tensor(0.1375, device=DEVICE))
        self.beta = nn.Parameter(torch.tensor(0.5, device=DEVICE))

        # 速率常量 (1/周期)
        self.gamma = 1/3.5
        self.lambdaa = 1/7
        self.lambdap = 1/1.5
        self.lambdam = 1/5.5
        self.lambdas = 1/5.5
        self.rhor = 1/15
        self.rhod = 1/13.3

    def set_beta(self, beta_val):
        with torch.no_grad():
            self.beta.copy_(torch.tensor(beta_val, dtype=torch.float32, device=DEVICE))

    def forward(self, t, y):
        # y 是当前的状态向量
        S, E, Ia, Ip, Im, Is, Hr, Hd, R, D = y
        
        # 1. 计算各项转移速率 (Rate)，而不是增量
        # 传染速率：beta * S * I / N
        # 注意：这里不再需要 dt，也不再需要 1 - exp(...)
        force_of_infection = self.beta * (self.Ca * Ia + self.Cp * Ip + self.Cm * Im + self.Cs * Is) / self.N
        rate_SE = self.population_factor * S * force_of_infection
        
        rate_EIa = E * self.alpha * self.gamma
        rate_EIp = E * (1 - self.alpha) * self.gamma
        
        rate_IaR = Ia * self.lambdaa
        
        rate_IpIm = Ip * self.mu * self.lambdap
        rate_IpIs = Ip * (1 - self.mu) * self.lambdap
        
        rate_ImR = Im * self.lambdam
        
        rate_IsHr = Is * self.delta * self.lambdas
        rate_IsHd = Is * (1 - self.delta) * self.lambdas
        
        rate_HrR = Hr * self.rhor
        rate_HdD = Hd * self.rhod

        # 2. 计算各状态的导数 (dy/dt)
        dS = -rate_SE
        dE = rate_SE - rate_EIa - rate_EIp
        dIa = rate_EIa - rate_IaR
        dIp = rate_EIp - rate_IpIs - rate_IpIm
        dIm = rate_IpIm - rate_ImR
        dIs = rate_IpIs - rate_IsHr - rate_IsHd
        dHr = rate_IsHr - rate_HrR
        dHd = rate_IsHd - rate_HdD
        dR = rate_IaR + rate_ImR + rate_HrR
        dD = rate_HdD
      
        # 返回导数向量
        return torch.stack([dS, dE, dIa, dIp, dIm, dIs, dHr, dHd, dR, dD])
    



def generate_inference_csv():
    set_global_seed(RNG_SEED)
    initial_conditions, policies, deltas, alphas, populations = generate_initial_conditions()
    
    min_pop = min(populations)
    max_pop = max(populations)
    
    time_points = 363
    t = torch.arange(0, time_points + 1, dtype=torch.float32, device=DEVICE)
    
    def get_beta_schedule(initial_beta=0.5, lambda_=0.005):
        beta_schedule = []
        current_beta = initial_beta
        
        for _ in range(105):
            beta_schedule.append(current_beta)
        
        for _ in range(time_points + 1 - 105):
            current_beta = current_beta * math.exp(-lambda_)
            beta_schedule.append(current_beta)
        
        return torch.tensor(beta_schedule, dtype=torch.float32, device=DEVICE)
    
    results = []
    
    for i, (y0, policy, delta, alpha) in enumerate(zip(initial_conditions, policies, deltas, alphas)):
        total_population = populations[i]
        
        model = ODEFunc(total_population, min_pop, max_pop).to(DEVICE)
        model.delta = nn.Parameter(torch.tensor(delta, device=DEVICE))
        model.alpha = torch.tensor(alpha, device=DEVICE)
        
        if policy == 'Strict':
            beta_values = get_beta_schedule(initial_beta=0.5, lambda_=0.005)
            
            results.append({
                'Time': t[0].item(),
                'y0_index': i,
                'Population': total_population,
                'beta': beta_values[0].item(),
                'delta': delta,
                'alpha': alpha,
                'Susceptible': y0[0].item() * 10 / total_population,
                'Exposed': y0[1].item() * 10 / total_population,
                'Infectious_asymptomatic': y0[2].item() * 10 / total_population,
                'Infectious_pre-symptomatic': y0[3].item() * 10 / total_population,
                'Infectious_mild': y0[4].item() * 10 / total_population,
                'Infectious_severe': y0[5].item() * 10 / total_population,
                'Hospitalized_recovered': y0[6].item() * 10 / total_population,
                'Hospitalized_deceased': y0[7].item() * 10 / total_population,
                'Recovered': y0[8].item() * 10 / total_population,
                'Deceased': y0[9].item() * 10 / total_population,
            })
            
            for time_step in range(len(beta_values)-1):
                t_half = t[time_step:time_step+2].to(DEVICE)
                model.set_beta(beta_values[time_step].item())
                with torch.no_grad():
                    Y_pred = odeint(model, y0, t_half, method='rk4')
                y0 = Y_pred[-1]
                
                results.append({
                    'Time': t_half[1].item(),
                    'y0_index': i,
                    'Population': total_population,
                    'beta': beta_values[time_step].item(),
                    'delta': delta,
                    'alpha': alpha,
                    'Susceptible': Y_pred[-1, 0].item() * 10 / total_population,
                    'Exposed': Y_pred[-1, 1].item() * 10 / total_population,
                    'Infectious_asymptomatic': Y_pred[-1, 2].item() * 10 / total_population,
                    'Infectious_pre-symptomatic': Y_pred[-1, 3].item() * 10 / total_population,
                    'Infectious_mild': Y_pred[-1, 4].item() * 10 / total_population,
                    'Infectious_severe': Y_pred[-1, 5].item() * 10 / total_population,
                    'Hospitalized_recovered': Y_pred[-1, 6].item() * 10 / total_population,
                    'Hospitalized_deceased': Y_pred[-1, 7].item() * 10 / total_population,
                    'Recovered': Y_pred[-1, 8].item() * 10 / total_population,
                    'Deceased': Y_pred[-1, 9].item() * 10 / total_population,
                })
        
        else:
            model.set_beta(0.5)
            with torch.no_grad():
                Y_pred = odeint(model, y0, t, method='rk4')
            
            for j, t_val in enumerate(t):
                results.append({
                    'Time': t_val.item(),
                    'y0_index': i,
                    'Population': total_population,
                    'beta': 0.5,
                    'delta': delta,
                    'alpha': alpha,
                    'Susceptible': Y_pred[j, 0].item() * 10 / total_population,
                    'Exposed': Y_pred[j, 1].item() * 10 / total_population,
                    'Infectious_asymptomatic': Y_pred[j, 2].item() * 10 / total_population,
                    'Infectious_pre-symptomatic': Y_pred[j, 3].item() * 10 / total_population,
                    'Infectious_mild': Y_pred[j, 4].item() * 10 / total_population,
                    'Infectious_severe': Y_pred[j, 5].item() * 10 / total_population,
                    'Hospitalized_recovered': Y_pred[j, 6].item() * 10 / total_population,
                    'Hospitalized_deceased': Y_pred[j, 7].item() * 10 / total_population,
                    'Recovered': Y_pred[j, 8].item() * 10 / total_population,
                    'Deceased': Y_pred[j, 9].item() * 10 / total_population,
                })

    df = pd.DataFrame(results)
    df.to_csv('Data/daily_10_val_train.csv', index=False)
    return df


def generate_causal_dataset(output_path='covid_causal_14d_dataset.csv', rng_seed: int = RNG_SEED):
    set_global_seed(rng_seed)
    # 获取初始数据
    initial_conditions, _, deltas, alphas, populations = generate_initial_conditions()
    
    min_pop = min(populations)
    max_pop = max(populations)
    all_samples = []

    state_names = ['Susceptible', 'Exposed', 'Infectious_asymptomatic', 
                   'Infectious_pre-symptomatic', 'Infectious_mild', 'Infectious_severe', 
                   'Hospitalized_recovered', 'Hospitalized_deceased', 'Recovered', 'Deceased']

    print(f"开始生成样本...")

    for i in range(len(populations)):
        pop = populations[i]
        y0_init = initial_conditions[i]
        
        # 定义该城市的 ODE 模型
        model = ODEFunc(pop, min_pop, max_pop).to(DEVICE)
        # model.delta = nn.Parameter(torch.tensor(deltas[i], device=DEVICE))
        # model.alpha = torch.tensor(alphas[i], device=DEVICE)

        model.alpha = nn.Parameter(torch.tensor(alphas[i], dtype=torch.float32, device=DEVICE))
        model.delta = nn.Parameter(torch.tensor(deltas[i], dtype=torch.float32, device=DEVICE))

        # 1. 生成 180 天的背景轨迹 (a=0, beta=0.5)
        model.set_beta(0.5)
        t_pool = torch.arange(0, 180, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            pool_data = odeint(model, y0_init, t_pool, method='rk4')

        # 2. 滑动窗口采样
        for start_t in range(0, 150, 7):
            hist_end = start_t + 14
            
            # 提取 14 天历史 X
            X_hist = pool_data[start_t:hist_end].cpu().numpy()
            # 与 pool 切片解耦，避免两次 forward 共享同一 storage 产生歧义
            y_anchor = pool_data[hist_end - 1].detach().clone()
            
            # 随机决定事实干预（与「城市层级政策→delta/alpha」可解耦；若按州/城固定处理，见模块文档说明的重叠性）
            a_factual = 1 if random.random() < CAUSAL_TREATMENT_ONE_PROB else 0
            
            # 未来按日积分 0..H（含端点），勿仅用 [0,H] 两点 — 见文件头 CAUSAL_FUTURE_DAYS 注释
            t_future = torch.arange(
                0.0, float(CAUSAL_FUTURE_DAYS) + 1.0, dtype=torch.float32, device=DEVICE
            )
            
            p_n = _normalized_infection_pressure(y_anchor, float(pop))
            beta_s = _beta_for_treatment_arm(True, p_n)
            beta_r = _beta_for_treatment_arm(False, p_n)

            # Strict / relaxed 各积分一次（与分两臂事实赋值一致）；避免重复调用且 ite 等价于 incr_s - incr_r
            model.set_beta(beta_s)
            with torch.no_grad():
                y_s = odeint(model, y_anchor.clone(), t_future, method='rk4')[-1]
            model.set_beta(beta_r)
            with torch.no_grad():
                y_r = odeint(model, y_anchor.clone(), t_future, method='rk4')[-1]

            def death_increment_normalized(y_end: torch.Tensor) -> float:
                return max(0.0, (y_end[9] - y_anchor[9]).item() * 10 / pop)

            incr_s = death_increment_normalized(y_s)
            incr_r = death_increment_normalized(y_r)

            outcome_f = incr_s if a_factual == 1 else incr_r
            outcome_cf = incr_r if a_factual == 1 else incr_s
            ite = incr_s - incr_r
            # 非负口径：若在严格政策下死亡人数增量更小，则表示「避免的额外死亡」（便于看 spread / 画图）
            ite_rel_minus_strict = incr_r - incr_s

            # 构建字典，确保所有键都有值
            sample_data = {
                'sample_id': f"city_{i}_t{start_t}",
                'y0_index': i,
                'start_time': start_t,
                'treatment': a_factual,
                'y_factual': float(outcome_f),
                'y_counterfactual': float(outcome_cf),
                'ite': float(ite),
                'ite_rel_minus_strict': float(ite_rel_minus_strict),
                'pop_size': int(pop)
            }
            
            # 平铺 X 特征
            for d in range(14):
                for s_idx, s_name in enumerate(state_names):
                    sample_data[f"x_d{d}_{s_name}"] = float(X_hist[d, s_idx] * 10 / pop)
            
            all_samples.append(sample_data)

    # 保存并检查
    df = pd.DataFrame(all_samples)
    df.to_csv(output_path, index=False)
    print(f"数据集生成完成！样本总数: {len(df)}")
    return df


# df = generate_inference_csv()
# df.head()

if __name__ == '__main__':
    set_global_seed(RNG_SEED)
    df = generate_causal_dataset()
    df.head()