"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +200（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -500（非法黑8/白球+黑8）, -100（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 500
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -500
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

def analyze_shot_for_reward_own(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（规则判定 + 走位优化）
    """
    
    # ==============================
    # Part A: 现有的规则判定逻辑 (保持不变)
    # ==============================
    
    # 1. 基本分析
    # 注意：这里需要确保 shot.balls 包含所有球的状态
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    # 这里为了通用性，最好动态获取合法的球ID，或者沿用你的硬编码
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # ==============================
    # Part B: 计算基础分数 (保持不变)
    # ==============================
    score = 0.0
    
    # 严重犯规 / 输掉比赛
    if cue_pocketed and eight_pocketed:
        return -500.0
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        return 100.0 if is_targeting_eight_ball_legally else -500.0
    elif cue_pocketed:
        return -100.0 # 洗袋通常也是严重扣分，而且此时不需要计算走位
            
    # 一般犯规
    if foul_first_hit:
        score -= 30.0
    if foul_no_rail:
        score -= 30.0
        
    # 进球得分
    score += len(own_pocketed) * 50.0
    score -= len(enemy_pocketed) * 20.0
    
    # 基础防守分
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10.0

    # ==============================
    # Part C: 新增 - 走位评分 (Position Play)
    # 只有在“未犯规”且“进球或有效击球”时才计算走位
    # ==============================
    if score > 0 and not foul_first_hit and not foul_no_rail:
        
        # 1. 确定下一杆的目标球集合
        # 逻辑：从当前目标中移除掉刚刚打进的
        remaining_targets = [bid for bid in player_targets if bid not in new_pocketed]
        
        # 特殊情况：如果当前球打完了，下一颗就是黑8
        if not remaining_targets and "8" not in new_pocketed:
            # 只有当黑8还在桌面上时（防止已经误进黑8的情况，虽然上面处理了）
            if shot.balls["8"].state.s != 4: 
                remaining_targets = ["8"]
        
        # 2. 计算白球与最近目标球的距离
        if remaining_targets:
            cue_ball = shot.balls['cue']
            cx, cy = cue_ball.state.rvw[0][0], cue_ball.state.rvw[0][1]
            
            min_dist = float('inf')
            found_target = False
            
            for tid in remaining_targets:
                if tid in shot.balls: # 确保球还在 System 对象里
                    t_ball = shot.balls[tid]
                    # 确保目标球在台面上 (state.s通常 1=静止, 2=滑动, 3=滚动, 4=进袋)
                    if t_ball.state.s != 4: 
                        tx, ty = t_ball.state.rvw[0][0], t_ball.state.rvw[0][1]
                        dist = np.sqrt((cx - tx)**2 + (cy - ty)**2)
                        if dist < min_dist:
                            min_dist = dist
                            found_target = True
            
            # 3. 如果找到了下一颗球，给予距离奖励
            if found_target:
                # 奖励函数：距离越近分越高。
                # 假设球桌长度约2米多。
                # 如果贴球 (dist -> 0)，奖励 +40分 (接近进一颗球的分数)
                # 如果距离 0.5米，奖励 +26分
                # 如果距离 2.0米，奖励 +13分
                position_bonus = 40.0 / (1.0 + 2.0 * min_dist) 
                score += position_bonus

    return score  

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    def __init__(self,
                 n_simulations=50,       # 仿真次数
                 c_puct=1.414):          # 探索系数
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.ball_radius = 0.028575
        
        # 定义噪声水平 (与 poolenv 保持一致或略大)
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005
        }

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0: return 0, 0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def generate_heuristic_actions(self, balls, my_targets, table):
        """
        生成候选动作列表
        """
        actions = []
        
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 获取所有目标球的ID
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        
        # 如果没有目标球了（理论上外部会处理转为8号，这里兜底）
        if not target_ids:
            target_ids = ['8']

        # 遍历每一个目标球
        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]

            # 遍历每一个袋口
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center

                # 1. 计算理论进球角度
                phi_ideal, dist = self._get_ghost_ball_target(cue_pos, obj_pos, pocket_pos)

                # 2. 根据距离简单的估算力度 (距离越远力度越大，基础力度2.0)
                v_base = 1.5 + dist * 1.5
                v_base = np.clip(v_base, 1.0, 6.0)

                # 3. 生成几个变种动作加入候选池
                # 变种1：精准一击
                actions.append({
                    'V0': v_base, 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种2：力度稍大
                actions.append({
                    'V0': min(v_base + 1.5, 7.5), 'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0
                })
                # 变种3：角度微调 (左右偏移 0.5 度，应对噪声)
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal + 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })
                actions.append({
                    'V0': v_base, 'phi': (phi_ideal - 0.5) % 360, 'theta': 0, 'a': 0, 'b': 0
                })

        # 如果通过启发式没有生成任何动作（极罕见），补充随机动作
        if len(actions) == 0:
            for _ in range(5):
                actions.append(self._random_action())
        
        # 随机打乱顺序
        random.shuffle(actions)
        return actions[:30]

    def simulate_action(self, balls, table, action):
        """
        [修改点1] 执行带噪声的物理仿真
        让 Agent 意识到由于误差的存在，某些“极限球”是不可打的
        """
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            # --- 注入高斯噪声 ---
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)

            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None: return self._random_action()
        
        # 预处理
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0: my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

        # 生成候选动作
        candidate_actions = self.generate_heuristic_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        
        # MCTS 循环
        for i in range(self.n_simulations):
            # Selection (UCB)
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                # 使用归一化后的 Q 进行计算
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = np.argmax(ucb_values)
            
            # Simulation (带噪声)
            shot = self.simulate_action(balls, table, candidate_actions[idx])

            # Evaluation
            if shot is None:
                raw_reward = -500.0
            else:
                raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
            
            # 映射公式: (val - min) / (max - min)
            normalized_reward = (raw_reward - (-500)) / 600.0
            # 截断一下防止越界
            normalized_reward = np.clip(normalized_reward, 0.0, 1.0)

            # Backpropagation
            N[idx] += 1
            Q[idx] += normalized_reward # 累加归一化后的分数

        # Final Decision
        # 选平均分最高的 (Robust Child)
        avg_rewards = Q / (N + 1e-6)
        best_idx = np.argmax(avg_rewards)
        best_action = candidate_actions[best_idx]
        
        # 简单打印一下当前最好的预测胜率
        print(f"[BasicAgent] Best Avg Score: {avg_rewards[best_idx]:.3f} (Sims: {self.n_simulations})")
        
        return best_action

class NewAgent(Agent):
    """
    NewAgent: 基于鲁棒性蒙特卡洛搜索的 Agent
    
    策略说明：
    1. 候选生成：基于 Ghost Ball 理论生成瞄准动作，并生成不同力度/微调角度的变体。
    2. 快速剪枝：先对候选动作进行无噪声（或低噪声）模拟，剔除明显无效（犯规/不进球）的动作。
    3. 鲁棒性评估（MCTS）：对优选出的动作进行多次带噪声的并行模拟（Rollout）。
       由于环境有随机噪声，一个动作必须在多次模拟中都能稳定得分才会被选中。
    """
    
    def __init__(self):
        super().__init__()
        self.ball_radius = 0.028575
        # 使用稍大于环境噪声的参数进行训练评估，确保安全性
        self.sim_noise = {
            'V0': 0.1, 'phi': 0.12, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        # 搜索参数
        self.num_candidates_per_pocket = 4   # 每个袋口生成的候选动作变体数
        self.max_candidates_total = 60       # 最大总候选数
        self.pruning_simulations = 1         # 剪枝阶段模拟次数
        self.robust_simulations = 10         # 决胜阶段每个动作的模拟次数
        self.top_k_candidates = 5            # 进入决胜阶段的动作数量

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_params(self, cue_pos, obj_pos, pocket_pos):
        """计算瞄准参数：理论角度 phi 和 击球距离 dist"""
        # 目标球到袋口的向量
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        
        if dist_obj_to_pocket == 0:
            return 0, 0, 0
            
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        
        # 幽灵球位置：目标球位置沿袋口反方向回退 2R
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        
        # 白球到幽灵球的向量
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        
        # 计算切球角度（Cut Angle）：白球-目标球连线 与 目标球-袋口连线 的夹角
        vec_cue_to_obj = np.array(obj_pos) - np.array(cue_pos)
        # 防止除0
        norm_c2o = np.linalg.norm(vec_cue_to_obj)
        if norm_c2o > 0:
            cos_angle = np.dot(vec_cue_to_obj / norm_c2o, unit_vec)
            # 夹角 (0度为直球，90度为薄球)
            cut_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        else:
            cut_angle = 90.0

        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        
        return phi, dist_cue_to_ghost, cut_angle

    def generate_candidates(self, balls, my_targets, table):
        """生成候选击球动作列表"""
        actions = []
        cue_ball = balls.get('cue')
        if not cue_ball: return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0]

        # 确定实际目标球（处理黑8逻辑）
        remaining_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        target_ids = remaining_targets if remaining_targets else ['8']
        
        # 策略：如果需要打黑8，且台面较空，可以更谨慎
        is_shooting_8 = (target_ids == ['8'])

        for tid in target_ids:
            obj_ball = balls[tid]
            obj_pos = obj_ball.state.rvw[0]
            
            for pocket_id, pocket in table.pockets.items():
                pocket_pos = pocket.center
                
                # 1. 计算几何参数
                phi_ideal, dist, cut_angle = self._get_ghost_ball_params(cue_pos, obj_pos, pocket_pos)
                
                # 2. 启发式过滤：如果切球角度太大（>80度），极难打进且容易犯规，跳过
                if cut_angle > 80:
                    continue
                
                # 3. 启发式力度计算
                # 基础力度随距离增加，切角越大也需要越大力度来保持线路
                v_base = 1.2 + dist * 1.8 + (cut_angle / 90.0) * 1.0
                
                # 4. 生成变体 (Variations)
                # 变体A: 标准力度，理论角度
                actions.append({
                    'V0': np.clip(v_base, 1.5, 7.0),
                    'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0,
                    'type': 'standard'
                })
                
                # 变体B: 大力出奇迹 (减少变线概率，但容易白球洗袋)
                actions.append({
                    'V0': np.clip(v_base + 2.0, 2.0, 8.0),
                    'phi': phi_ideal, 'theta': 0, 'a': 0, 'b': 0,
                    'type': 'power'
                })
                
                # 变体C/D: 角度微调 (应对噪声，尝试覆盖左右误差)
                # 偏移量：远距离偏移小，近距离偏移大? 其实角度是固定的，但噪声是角度噪声。
                offset = 0.2 # 度
                actions.append({'V0': np.clip(v_base, 1.5, 7.0), 'phi': (phi_ideal + offset)%360, 'theta': 0, 'a': 0, 'b': 0, 'type': 'adj+'})
                actions.append({'V0': np.clip(v_base, 1.5, 7.0), 'phi': (phi_ideal - offset)%360, 'theta': 0, 'a': 0, 'b': 0, 'type': 'adj-'})

        # 如果没有生成任何合理的进攻动作（比如全都被阻挡或角度太死），生成随机防守动作或轻轻击打
        if not actions:
            # 尝试轻轻击打最近的一颗目标球
            closest_tid = None
            min_dist = float('inf')
            for tid in target_ids:
                d = np.linalg.norm(balls[tid].state.rvw[0] - cue_pos)
                if d < min_dist:
                    min_dist = d
                    closest_tid = tid
            
            if closest_tid:
                # 朝着球打，力度很小
                phi, _, _ = self._get_ghost_ball_params(cue_pos, balls[closest_tid].state.rvw[0], balls[closest_tid].state.rvw[0]) 
                # 这里不需要GhostBall，直接对着球心打即可避免 No Hit
                vec = balls[closest_tid].state.rvw[0] - cue_pos
                phi = self._calc_angle_degrees(vec)
                actions.append({'V0': 1.0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0, 'type': 'safety'})
            else:
                return [self._random_action()]

        random.shuffle(actions)
        return actions[:self.max_candidates_total]

    def simulate_shot(self, balls, table, action, enable_noise=True):
        """执行单次物理模拟"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table) # Table 其实通常不变，浅拷贝可能够用，但为了安全深拷贝
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        # 注入噪声
        if enable_noise:
            noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
            noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
            noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
            noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
            noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)
            cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
        else:
            cue.set_state(V0=action['V0'], phi=action['phi'], theta=action['theta'], a=action['a'], b=action['b'])
            
        try:
            pt.simulate(shot, inplace=True)
            return shot
        except Exception:
            return None

    def decision(self, balls=None, my_targets=None, table=None):
        """
        基于 MCTS (Monte Carlo Tree Search) 思想的决策函数
        实际上实现的是：Candidate Generation -> Pruning -> Robustness Rollout
        """
        if balls is None: return self._random_action()
        
        # 0. 准备工作
        # 识别真正的目标列表（如果是空，说明要打黑8）
        real_targets = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not real_targets: 
            real_targets = ["8"]
        
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        
        # 1. Expansion: 生成候选动作
        candidates = self.generate_candidates(balls, my_targets, table)
        
        # 2. Selection / Pruning: 快速筛选
        # 对每个候选动作进行 1 次无噪声（或极低噪声）模拟，快速判断是否有进球潜力
        promising_candidates = []
        
        for action in candidates:
            # 第一次筛选不加噪声，或者加很小的噪声，为了验证理论可行性
            shot = self.simulate_shot(balls, table, action, enable_noise=False)
            
            if shot is None:
                continue
                
            score = analyze_shot_for_reward(shot, last_state, real_targets)
            
            # 如果分数太低（比如犯规、或者没进球），在第一轮就大概率淘汰
            # 但为了防止漏掉“虽然没进球但做了一杆好防守”的情况，这里的阈值不能太高
            # 主要为了剔除直接母球洗袋或严重犯规的球
            if score > -50: 
                promising_candidates.append((action, score))
        
        # 根据初步分数排序，选出 Top K
        promising_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [x[0] for x in promising_candidates[:self.top_k_candidates]]
        
        # 如果所有候选都很差，随机选几个去搏一搏（避免空列表）
        if not top_candidates:
            top_candidates = candidates[:5]

        # 3. Simulation / Evaluation: 鲁棒性测试 (MCTS Rollouts)
        # 对 Top K 动作进行多次带噪声模拟，计算平均期望回报
        best_avg_score = -float('inf')
        best_action = top_candidates[0]
        
        print(f"[NewAgent] Evaluating {len(top_candidates)} candidates with {self.robust_simulations} rollouts each...")

        for action in top_candidates:
            cumulative_score = 0
            valid_sims = 0
            
            for _ in range(self.robust_simulations):
                shot = self.simulate_shot(balls, table, action, enable_noise=True)
                if shot:
                    # 注意：这里传入 real_targets 确保正确判断黑8逻辑
                    r = analyze_shot_for_reward(shot, last_state, real_targets)
                    cumulative_score += r
                    valid_sims += 1
            
            avg_score = cumulative_score / max(valid_sims, 1)
            
            # 简单的调试输出
            # print(f"  Action V0={action['V0']:.1f}, phi={action['phi']:.1f} -> Avg Score: {avg_score:.1f}")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_action = action
        
        print(f"[NewAgent] Selected Action: V0={best_action['V0']:.2f}, phi={best_action['phi']:.2f}, Exp.Reward={best_avg_score:.1f}")
        return best_action