"""
Docstring for Artificial_Intelligence_course_proj.hyperparameter_finding
直接翻到最后面看main怎么用。
有两个TODO
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from matplotlib import pyplot as plt

from agents import cartpole_dqn
from agents import cartpole_ppo
import train
import numpy as np
from typing import Dict, List, Any, Callable
import json
import pandas as pd
import multiprocessing

import traceback

from agents.cartpole_dqn import DQNConfig, DQNSolver
from agents.cartpole_dqn_priority import PDQNConfig, PDQNSolver
from agents.cartpole_ppo import PPOConfig, PPOSolver

class HyperparamTuner:
    def __init__(self, algorithm: str, use_early_stopping: bool = True):
        self.algorithm = algorithm
        self.results = []
        self.use_early_stopping = use_early_stopping  # 控制是否启用早停

    def define_search_space(self) -> Dict[str, tuple]:
        """定义每个超参数的采样范围"""
        spaces = {
            "dqn": {
                "lr": ((1e-4, 1e-2), 'log'),
                "gamma": ((0.9, 0.999), 'uniform'),
                "batch_size": ([16, 32, 64, 128], 'choice'),
                "memory_size": ((10000, 100000), 'int'),
                "target_update": ([100, 500, 1000], 'choice'),
                "eps_start": ((0.9, 1.0), 'uniform'),
                "eps_end": ((0.01, 0.1), 'uniform'),
                "eps_decay": ((0.99, 0.999), 'uniform'),
                "initial_exploration": ((500, 2000), 'int'),
            },
            "ppo": {
                "learning_rate": ((1e-4, 3e-4), 'log'),
                "gamma": ((0.9, 0.999), 'uniform'),
                "value_coef": ((0.25, 1.0), 'uniform'),
                "entropy_coef": ((1e-4, 1e-2), 'log'),
                "lambda_gae": ((0.9, 0.98), 'uniform'),
                "clip_eps": ((0.1, 0.3), 'uniform'),

                # 这三个就是你要联合探索的 rollout 超参数
                # 这里用离散的典型取值，组合出来就会有 (2048, 64, 16) 这一类
                "memory_size": ([1024, 2048, 4096, 8192], 'choice'),
                "minibatch_size": ([32, 64, 128, 256], 'choice'),
                "epoch": ([4, 8, 16, 32], 'choice'),
            },
            "pdqn": {  # 添加 PDQN 搜索空间
                "lr": ((1e-4, 1e-2), 'log'),
                "gamma": ((0.9, 0.999), 'uniform'),
                "batch_size": ([16, 32, 64, 128], 'choice'),
                "memory_size": ((10000, 100000), 'int'),
                "target_update": ([100, 500, 1000], 'choice'),
                "eps_start": ((0.9, 1.0), 'uniform'),
                "eps_end": ((0.01, 0.1), 'uniform'),
                "eps_decay": ((0.99, 0.999), 'uniform'),
                "initial_exploration": ((500, 2000), 'int'),
                # PDQN 特有参数
                "alpha": ((0.5, 0.7), 'uniform'),  # 优先级强度
                "beta": ((0.4, 0.6), 'uniform'),  # 重要性采样初始值
                "beta_increment": ((0.0005, 0.002), 'log'),  # beta 增量
            }
            # TODO: add other parameter you want to adjust
        }

        return spaces.get(self.algorithm, {})

    # def sample_params(self) -> Dict[str, Any]:
    #     """根据搜索空间随机采样一组参数"""
    #     space = self.define_search_space()
    #     params: Dict[str, Any] = {}

    #     while True:
    #         params.clear()
    #         for name, (range_val, sampling) in space.items():

    #             if sampling == 'log':
    #                 params[name] = 10 ** np.random.uniform(
    #                     np.log10(range_val[0]),
    #                     np.log10(range_val[1])
    #                 )
    #             elif sampling == 'uniform':
    #                 params[name] = np.random.uniform(range_val[0], range_val[1])
    #             elif sampling == 'choice':
    #                 params[name] = np.random.choice(range_val)
    #             elif sampling == 'int':
    #                 params[name] = int(np.random.uniform(range_val[0], range_val[1]))

    #         # 对 PPO 做一个简单约束：memory_size 至少是 minibatch_size 的 4 倍，
    #         # 避免 batch 太少，训练不稳定
    #         if self.algorithm == "ppo":
    #             if params["memory_size"] < 4 * params["minibatch_size"]:
    #                 continue  # 不满足约束就重新采样一组
    #         break

    #     return params
    def sample_params(self) -> Dict[str, Any]:
        """根据搜索空间随机采样一组参数"""
        space = self.define_search_space()
        params: Dict[str, Any] = {}

        while True:
            params.clear()
            for name, (range_val, sampling) in space.items():
                if sampling == 'log':
                    value = 10 ** np.random.uniform(
                        np.log10(range_val[0]),
                        np.log10(range_val[1])
                    )
                    params[name] = value.item() if hasattr(value, 'item') else float(value)
                elif sampling == 'uniform':
                    value = np.random.uniform(range_val[0], range_val[1])
                    params[name] = value.item() if hasattr(value, 'item') else float(value)
                elif sampling == 'choice':
                    value = np.random.choice(range_val)
                    params[name] = value.item() if hasattr(value, 'item') else value
                elif sampling == 'int':
                    value = np.random.uniform(range_val[0], range_val[1])
                    params[name] = int(value)

            # 对 PPO 做一个简单约束：memory_size 至少是 minibatch_size 的 4 倍，
            # 避免 batch 太少，训练不稳定
            if self.algorithm == "ppo":
                if params["memory_size"] < 4 * params["minibatch_size"]:
                    continue  # 不满足约束就重新采样一组
            break

        return params

    def create_config(self, params: Dict) -> Any:
        """根据参数创建配置实例"""
        if self.algorithm == "dqn":
            return DQNConfig(
                gamma=params.get('gamma', 0.99),
                lr=params.get('lr', 1e-3),
                batch_size=params.get('batch_size', 32),
                memory_size=params.get('memory_size', 50000),
                target_update=params.get('target_update', 500),
                eps_start=params.get('eps_start', 1.0),
                eps_end=params.get('eps_end', 0.05),
                eps_decay=params.get('eps_decay', 0.995),
                initial_exploration=params.get('initial_exploration', 1000)
            )
        elif self.algorithm == "pdqn":  # 添加 PDQN 配置
            return PDQNConfig(
                gamma=params.get('gamma', 0.99),
                lr=params.get('learning_rate', 1e-3),  # 注意：PDQNConfig 使用 lr 而不是 learning_rate
                batch_size=params.get('batch_size', 32),
                memory_size=params.get('memory_size', 50000),
                target_update=params.get('target_update', 500),
                eps_start=params.get('eps_start', 1.0),
                eps_end=params.get('eps_end', 0.05),
                eps_decay=params.get('eps_decay', 0.995),
                initial_exploration=params.get('initial_exploration', 1000),
                # PDQN 特有参数
                alpha=params.get('alpha', 0.6),
                beta=params.get('beta', 0.4),
                beta_increment=params.get('beta_increment', 0.001)
            )
        # 未来可以添加其他算法的配置创建
        # TODO:
        elif self.algorithm == "ppo":
            # 确保整数参数确实是整数
            
            
            return PPOConfig(   
                gamma=params.get('gamma', 0.99),
                value_coef=params.get('value_coef', 0.5),
                entropy_coef=params.get('entropy_coef', 3e-4),
                learning_rate=params.get('learning_rate', 1e-3),
                lambda_gae=params.get('lambda_gae', 0.95),
                clip_eps=params.get('clip_eps', 0.2),
                memory_size=params.get('memory_size', 8196),
                minibatch_size=params.get('minibatch_size', 512),
                epoch=params.get('epoch', 16),

        )
        # elif self.algorithm == "":
        raise ValueError(f"Unsupported algorithm: {self.algorithm}")


    def run_trial(self, trial_id: int, num_episodes: int = 200) -> Dict:
        """运行一次完整的试验"""
        print("start one trail")
        # 1. 采样参数
        params = self.sample_params()
        print(f"\n[Trial {trial_id}] Parameters: {params}")

        # 2. 创建配置
        config = self.create_config(params)

        # 3. 训练和评估
        try:
            # 使用train.py中的函数
            agent, avg_score, saved_path = train.train_with_config(
                config,
                num_episodes=num_episodes,
                save=False  # 不保存每个试验的模型，节省空间
            )

            # 4. 记录结果
            result = {
                "trial_id": trial_id,
                **params,
                "avg_score": avg_score,
                "success": True
            }

        except Exception as e:
            print(f"[Trial {trial_id}] Failed with error: {e}")
            result = {
                "trial_id": trial_id,
                **params,
                "avg_score": 0,
                "success": False,
                "error": str(e)
            }

        self.results.append(result)
        self.save_progress()

        return result

    def run_search(self, n_trials: int = 30, num_episodes: int = 200) -> pd.DataFrame:
        """运行多次试验"""
        print(f"####run_several trails#####")
        print(f"Starting hyperparameter search for {self.algorithm}")
        print(f"Number of trials: {n_trials}")
        print(f"Episodes per trial: {num_episodes}")

        for i in range(n_trials):
            print(f"\n{'=' * 50}")
            print(f"Starting trial {i + 1}/{n_trials}")
            self.run_trial(i, num_episodes)

        # 保存最终结果
        df = pd.DataFrame(self.results)

        results_file = f"output/param_table/hyperparam_results_{self.algorithm}.csv"
        if not os.path.exists(results_file):  # 如果路径不存在，创建路径
            os.makedirs("output/param_table", exist_ok=True)

        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")

        # 分析结果
        self.analyze_results(df)

        return df

    # def save_progress(self):
    #     """定期保存进度，防止中断"""
    #     progress_file = f"output/param_tuning_progress/tuning_progress_{self.algorithm}.json"
    #
    #     if not os.path.exists(progress_file):  # 如果路径不存在，创建路径
    #
    #         os.makedirs("output/param_tuning_progress", exist_ok=True)
    #
    #     with open(progress_file, 'w') as f:
    #         json.dump(self.results, f, indent=2)

    def save_progress(self):
        """定期保存进度，防止中断 - 修复版本"""
        progress_file = f"output/param_tuning_progress/tuning_progress_{self.algorithm}.json"

        if not os.path.exists(os.path.dirname(progress_file)):
            os.makedirs(os.path.dirname(progress_file))

        try:
            # 转换所有结果为可序列化格式
            serializable_results = []
            for result in self.results:
                serializable_results.append(
                    self._convert_to_serializable(result)
                )

            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            # 可选：同时保存一个备份
            backup_file = f"output/param_tuning_progress/tuning_progress_{self.algorithm}_backup.json"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Failed to save progress: {e}")
            # 尝试保存简化版本
            try:
                simplified_results = []
                for result in self.results:
                    simplified = {
                        'trial_id': int(result.get('trial_id', 0)),
                        'avg_score': float(result.get('avg_score', 0.0)),
                        'success': bool(result.get('success', False))
                    }
                    simplified_results.append(simplified)

                with open(progress_file + '.simple', 'w', encoding='utf-8') as f:
                    json.dump(simplified_results, f, indent=2)
            except:
                pass


    def analyze_results(self, df: pd.DataFrame) -> None:
        """分析并可视化结果"""
        if len(df) == 0:
            print("No results to analyze")
            return

        # 只分析成功的试验
        success_df = df[df['success'] == True] if 'success' in df.columns else df

        if len(success_df) == 0:
            print("No successful trials to analyze")
            return

        print(f"\n{'=' * 50}")
        print("ANALYSIS RESULTS")
        print('=' * 50)

        # 找到最佳参数组合
        if 'avg_score' in success_df.columns:
            best_idx = success_df['avg_score'].idxmax()
            best_result = success_df.loc[best_idx]

            print(f"\nBest trial ID: {int(best_result['trial_id'])}")
            print(f"Best average score: {best_result['avg_score']:.2f}")
            print("\nBest hyperparameters:")
            for key, value in best_result.items():
                if key not in ['trial_id', 'avg_score', 'success', 'error']:
                    print(f"  {key}: {value}")

        # 计算相关性
        if len(success_df) > 1:
            numeric_cols = success_df.select_dtypes(include=[np.number]).columns
            # 移除可能不是超参数的列
            numeric_cols = [col for col in numeric_cols
                            if col not in ['trial_id', 'avg_score', 'success']]

            if 'avg_score' in success_df.columns and len(numeric_cols) > 0:
                correlations = {}
                for col in numeric_cols:
                    corr = success_df[col].corr(success_df['avg_score'])
                    correlations[col] = corr

                print("\nCorrelations with score (absolute values):")
                for col, corr in sorted(correlations.items(),
                                        key=lambda x: abs(x[1]), reverse=True):
                    print(f"  {col}: {corr:.3f}")

        # 绘制图表
        self.plot_results(success_df)

    def plot_results(self, df: pd.DataFrame):
        """绘制超参数对性能的影响"""
        if 'avg_score' not in df.columns:
            return

        time_stamp = time.strftime("%Y%m%d-%H%M%S")

        # 准备绘图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # 绘制不同超参数的散点图
        params_to_plot = ['learning_rate', 'gamma', 'batch_size', 'memory_size']

        for i, param in enumerate(params_to_plot):
            if i >= len(axes):
                break
            if param in df.columns:
                ax = axes[i]
                ax.scatter(df[param], df['avg_score'], alpha=0.6)

                # 如果参数是学习率，使用对数坐标
                if param == 'learning_rate':
                    ax.set_xscale('log')

                ax.set_xlabel(param)
                ax.set_ylabel('Average Score')
                ax.set_title(f'{param} vs Score')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = f"output/figures/plot/hyperparam_analysis_{self.algorithm}_{time_stamp}.png"
        if not os.path.exists(plot_file):
            os.makedirs("output/figures/plot", exist_ok=True)
        plt.savefig(plot_file, dpi=150)
        print(f"\nAnalysis plot saved to: {plot_file}")
        plt.show()

        # 绘制分数分布直方图
        plt.figure(figsize=(8, 6))
        plt.hist(df['avg_score'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Average Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Trial Scores')
        plt.grid(True, alpha=0.3)

        hist_file = f"output/figures/hist/score_distribution_{self.algorithm}_{time_stamp}.png"
        if not os.path.exists(hist_file):
            os.makedirs("output/figures/hist", exist_ok=True)
        plt.savefig(hist_file, dpi=150)
        plt.show()

        # 绘制分数随试验次数的变化
        if 'trial_id' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['trial_id'], df['avg_score'], 'bo-', alpha=0.7)
            plt.xlabel('Trial ID')
            plt.ylabel('Average Score')
            plt.title('Score Progression Over Trials')
            plt.grid(True, alpha=0.3)

            # 添加趋势线
            if len(df) > 1:
                z = np.polyfit(df['trial_id'], df['avg_score'], 1)
                p = np.poly1d(z)
                plt.plot(df['trial_id'], p(df['trial_id']), "r--", alpha=0.8,
                         label=f'Trend (slope={z[0]:.2f})')
                plt.legend()

            trend_file = f"output/figures/score_trend/score_trend_{self.algorithm}_{time_stamp}.png"
            if not os.path.exists(trend_file):
                os.makedirs("output/figures/score_trend", exist_ok=True)
            plt.savefig(trend_file, dpi=150)
            plt.show()

    def run_search_parallel(self, n_trials: int = 30, num_episodes: int = 200,
                            max_workers: int = None, use_gpu: bool = False,
                            early_stop_patience: int = None,
                            early_stop_min_episodes: int = None,
                            early_stop_threshold: float = None,
                            early_stop_window_size: int = None,
                            time_stamp: str|None = None,
                            ) -> pd.DataFrame:
        """
        并行运行超参数搜索

        参数:
            n_trials: 试验总数
            num_episodes: 每个试验的训练回合数
            max_workers: 最大工作进程数（默认使用CPU核心数-1）
            use_gpu: 是否使用GPU（注意：多个进程共享GPU可能造成内存冲突）
            early_stop_patience: 早停耐心值（连续多少个episode没有改进）
            early_stop_min_episodes: 最小训练回合数
            early_stop_threshold: 早停阈值（达到此分数时提前停止）
            early_stop_window_size: 滑动窗口大小
            time_stamp: 时间戳
        """
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)

        print(f"Starting parallel hyperparameter search for {self.algorithm}")
        print(f"Number of trials: {n_trials}, Workers: {max_workers}")
        print(f"Episodes per trial: {num_episodes}")
        print(f"GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")

        # 预先创建所有需要的目录
        self._create_output_dirs()

        # 准备所有试验的参数
        all_trials = []
        
        for i in range(n_trials):
            params = self.sample_params()
            all_trials.append({
                "trial_id": i,
                "params": params,
                "num_episodes": num_episodes,
                "use_gpu": use_gpu,
                "early_stop_patience": early_stop_patience,
                "early_stop_min_episodes": early_stop_min_episodes,
                "early_stop_threshold": early_stop_threshold,
                "early_stop_window_size": early_stop_window_size,
                "use_early_stopping": self.use_early_stopping,
                "agent" : self.algorithm
            })

        results = []
        completed = 0

        # 使用进程池并行执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_trial = {}
            for trial in all_trials:
                future = executor.submit(
                    self._run_single_trial_parallel,
                    trial_id=trial["trial_id"],
                    params=trial["params"],
                    num_episodes=trial["num_episodes"],
                    use_gpu=trial["use_gpu"],
                    early_stop_patience=trial["early_stop_patience"],
                    early_stop_min_episodes=trial["early_stop_min_episodes"],
                    early_stop_threshold=trial["early_stop_threshold"],
                    early_stop_window_size=trial["early_stop_window_size"],
                    use_early_stopping=trial["use_early_stopping"],
                    agent_type=trial["agent"],
                )

                future_to_trial[future] = trial["trial_id"]

            # 处理完成的任务
            for future in as_completed(future_to_trial):
                trial_id = future_to_trial[future]
                try:
                    result = future.result(timeout=3600)  # 1小时超时
                    results.append(result)
                    completed += 1

                    # 显示进度
                    print(f"[Parallel] Trial {trial_id} completed ({completed}/{n_trials}) - "
                          f"Score: {result.get('avg_score', 0):.2f}, "
                          f"Success: {result.get('success', False)}")

                    # 定期保存进度
                    if completed % 5 == 0:
                        self.results = results
                        self.save_progress()

                except Exception as e:
                    print(f"[Parallel] Trial {trial_id} failed: {str(e)[:100]}...")
                    error_result = {
                        "trial_id": trial_id,
                        "avg_score": 0,
                        "success": False,
                        "error": str(e)
                    }
                    results.append(error_result)

        # 保存最终结果
        self.results = results
        df = pd.DataFrame(results)

        results_file = f"output/param_table/hyperparam_results_{self.algorithm}_parallel_{time_stamp}.csv" if time_stamp else f"output/param_table/hyperparam_results_{self.algorithm}_parallel.csv"
        df.to_csv(results_file, index=False)
        print(f"\nParallel results saved to: {results_file}")

        # 分析结果
        self.analyze_results(df)

        return df

    def _run_single_trial_parallel(self,
                                    trial_id: int, params: Dict,
                                    num_episodes: int, use_gpu: bool,
                                    early_stop_patience: int = None,
                                    early_stop_min_episodes: int = None,
                                    early_stop_threshold: float = None,
                                    early_stop_window_size: int = None,
                                    use_early_stopping: bool = False,
                                    agent_type: str|None = None ) -> Dict:
        """
        在独立进程中运行单个试验

        注意：每个进程有独立的Python环境，需要重新导入模块
        """
    
        # 设置进程环境变量
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        # 为每个进程设置不同的随机种子
        import random
        import numpy as np
        seed = trial_id * 1000 + os.getpid()
        random.seed(seed)
        np.random.seed(seed)

        # 必须在函数内部导入，因为每个进程有独立的命名空间
        import torch
        import gymnasium as gym
        from agents.cartpole_dqn import DQNConfig, DQNSolver
        from agents.cartpole_ppo import PPOConfig, PPOSolver
        from agents.cartpole_dqn_priority import PDQNConfig, PDQNSolver

        # 设置设备
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.manual_seed(seed)
        else:
            device = "cpu"
        torch.manual_seed(seed)

        try:
            # 创建配置
            # TODO: 根据agent名称，配置config     
            if not agent_type:
                raise ValueError(...)
            
            env = gym.make("CartPole-v1")
            
            print("enter parallel training and create env successfully")
            
            if agent_type == "dqn":
                config = DQNConfig(
                    gamma=params.get('gamma', 0.99),
                    lr=params.get('lr', 1e-3),
                    batch_size=params.get('batch_size', 32),
                    memory_size=params.get('memory_size', 50000),
                    target_update=params.get('target_update', 500),
                    eps_start=params.get('eps_start', 1.0),
                    eps_end=params.get('eps_end', 0.05),
                    eps_decay=params.get('eps_decay', 0.995),
                    initial_exploration=params.get('initial_exploration', 1000),
                    device=device
                )
            elif agent_type == "ppo":
                
                config = PPOConfig(
                    gamma=params.get('gamma', 0.99),
                    value_coef=params.get('value_coef', 0.5),
                    entropy_coef=params.get('entropy_coef', 3e-4),
                    learning_rate=params.get('learning_rate', 1e-3),
                    lambda_gae=params.get('lambda_gae', 0.95),
                    clip_eps=params.get('clip_eps', 0.2),
                    memory_size=params.get('memory_size', 4096),
                    minibatch_size=params.get('minibatch_size', 512),
                    epoch=params.get('epoch', 16),
                    device=device,
                )

            elif agent_type == "pdqn":  # 添加 PDQN
                config = PDQNConfig(
                    gamma=params.get('gamma', 0.99),
                    lr=params.get('lr', 1e-3),
                    batch_size=params.get('batch_size', 32),
                    memory_size=params.get('memory_size', 50000),
                    target_update=params.get('target_update', 500),
                    eps_start=params.get('eps_start', 1.0),
                    eps_end=params.get('eps_end', 0.05),
                    eps_decay=params.get('eps_decay', 0.995),
                    initial_exploration=params.get('initial_exploration', 1000),
                    alpha=params.get('alpha', 0.6),
                    beta=params.get('beta', 0.4),
                    beta_increment=params.get('beta_increment', 0.001),
                    device=device,
                )
                
            if use_early_stopping and early_stop_patience is not None:
                print("启动早停")
                agent_obj, scores = self.train_with_early_stopping(
                    config=config,
                    env=env,
                    min_episodes=early_stop_min_episodes or 50,
                    patience=early_stop_patience or 20,
                    stop_threshold=early_stop_threshold or 495.0,
                    window_size=early_stop_window_size or 10,
                    num_episodes=num_episodes,
                )
                actual_episodes_trained = len(scores)
                avg_score = float(np.mean(scores[-10:])) if len(scores) > 0 else 0.0
                early_stopped = actual_episodes_trained < num_episodes
                stop_reason = "early_stopping" if early_stopped else "completed"
            else:
                # 原有的训练逻辑（作为备选）
                obs_dim = env.observation_space.shape[0]
                act_dim = env.action_space.n
                if self.algorithm == "dqn":
                    agent = DQNSolver(obs_dim, act_dim, cfg=config)
                elif self.algorithm == "ppo":
                    agent = PPOSolver(obs_dim, act_dim,cfg=config)
                elif self.algorithm == "pdqn":
                    agent = PDQNSolver(obs_dim, act_dim, cfg=config)
                scores = []
                # self.run_trials(agent, scores)
                print("无早停")

                for ep in range(1, num_episodes + 1):
                    state, _ = env.reset(seed=seed + ep)
                    state = np.reshape(state, (1, obs_dim))
                    steps = 0                        
                        
                    while True:
                        action = agent.act(state)
                        next_state_raw, reward, terminated, truncated, info = env.step(action)

                        # 调试代码
                        # print(f"terminated: {terminated}, truncated: {truncated}, reward: {reward}")

                        done = terminated or truncated
                        next_state = np.reshape(next_state_raw, (1, obs_dim))

                        # 根据 agent_type 调用不同的 step 方法
                        if agent_type == "dqn":
                            agent.step(state, action, float(reward), next_state, done)
                        elif agent_type == "ppo":
                            agent.step(state, action, float(reward), done)
                        elif agent_type == "pdqn":
                            agent.step(state, action, float(reward), next_state, done)

                        # 6. Move to next state
                        state = next_state
                        steps += 1
                        # 7. Episode end: log and break
                        if done:
                            # 调试代码
                            # print("done after {} steps".format(steps))

                            scores.append(steps)
                            break
                        
                # train.train_with_config(config=config, num_episodes=num_episodes)
                
                actual_episodes_trained = num_episodes
                avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                early_stopped = False
                stop_reason = "completed"

            env.close()

            # 构建结果字典，确保类型可序列化
            # result = {
            #     "trial_id": int(trial_id),
            #     **params,
            #     "avg_score": float(avg_score),
            #     "final_scores": [int(s) for s in scores[-5:]] if len(scores) >= 5 else [int(s) for s in scores],
            #     "total_episodes": int(actual_episodes_trained),
            #     "max_episodes": int(num_episodes),
            #     "early_stopped": bool(early_stopped),
            #     "stop_reason": str(stop_reason),
            #     "device_used": str(device),
            #     "success": True,
            #     "seed": int(seed)
            # }
            result = {
                "trial_id": int(trial_id),
                **params,
                "avg_score": float(avg_score),
                "final_scores": [int(s) for s in scores[-5:]] if len(scores) >= 5 else [int(s) for s in scores],
                "total_episodes": int(actual_episodes_trained),
                "max_episodes": int(num_episodes),
                "early_stopped": bool(early_stopped),
                "stop_reason": str(stop_reason),
                "device_used": str(device),
                "success": True,
                "seed": int(seed),
            }

            # 额外信息：如果早停，记录停止时的表现趋势
            if early_stopped and len(scores) >= 10:
                last_5 = np.mean(scores[-5:])
                first_5 = np.mean(scores[:5])
                result["improvement_ratio"] = float(last_5 / first_5) if first_5 > 0 else 0.0

            return result

        except Exception as e:
            # 错误结果也要确保可序列化
            error_msg = f"Trial {trial_id} error: {str(e)[:200]}"
            return {
                "trial_id": int(trial_id),
                **{k: (int(v) if isinstance(v, np.integer) else
                       float(v) if isinstance(v, np.floating) else v)
                   for k, v in params.items()},
                "avg_score": 0.0,
                "success": False,
                "error": error_msg,
                "device_used": "unknown"
            }



    def _create_output_dirs(self):
        """创建所有必要的输出目录"""
        dirs = [
            "output/param_table",
            "output/param_tuning_progress",
            "output/figures/plot",
            "output/figures/hist",
            "output/figures/score_trend",
            "output/best_config"
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    # 原有的串行方法，作为备选
    # def run_search(self, n_trials: int = 30, num_episodes: int = 200) -> pd.DataFrame:
    #     """原有的串行搜索方法"""
    #     print(f"Starting sequential hyperparameter search for {self.algorithm}")
    #     return super().run_search(n_trials, num_episodes)


    @staticmethod
    def _convert_to_serializable(obj):
        """
        将对象转换为JSON可序列化的格式
        处理NumPy类型、Pandas类型以及其他常见非序列化类型
        """
        import numpy as np
        import pandas as pd

        # 处理NumPy整数类型
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)

        # 处理NumPy浮点数类型
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)

        # 处理NumPy布尔类型
        if isinstance(obj, np.bool_):
            return bool(obj)

        # 处理NumPy数组
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # 处理Pandas Series和DataFrame
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')

        # 处理字典（递归转换值）
        if isinstance(obj, dict):
            return {k: HyperparamTuner._convert_to_serializable(v) for k, v in obj.items()}

        # 处理列表、元组（递归转换元素）
        if isinstance(obj, (list, tuple, set)):
            return [HyperparamTuner._convert_to_serializable(item) for item in obj]

        # 处理None
        if obj is None:
            return None

        # 对于其他类型，尝试直接返回，如果出错则转为字符串
        try:
            # 测试是否可JSON序列化
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            # 无法序列化则转为字符串表示
            return str(obj)





    def train_with_early_stopping(self, config, env, min_episodes=50, patience=20,
                                stop_threshold=495.0, window_size=10,
                                num_episodes=200, verbose=False):
        """
        带有早停机制的训练函数（优化版）
        """
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        if self.algorithm == "dqn":
            agent = DQNSolver(obs_dim, act_dim, cfg=config)
        elif self.algorithm == "ppo":
            agent = PPOSolver(obs_dim, act_dim,cfg = config)
        elif self.algorithm == "pdqn":
            agent = PDQNSolver(obs_dim, act_dim, cfg = config)

        best_score = -float('inf')
        no_improve = 0
        scores = []
        best_episode = 0

        for episode in range(1, num_episodes + 1):
            # 重置环境
            state, _ = env.reset(seed=episode + config.seed if hasattr(config, 'seed') else episode)
            state = np.reshape(state, (1, obs_dim))
            steps = 0

            # 运行一个episode
            while True:
                if self.algorithm == "dqn":
                    action = agent.act(state)
                    next_state_raw, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_state = np.reshape(next_state_raw, (1, obs_dim))
                
                    agent.step(state, action, reward, next_state, done)
                    #    remembering and learning internally.
                    # agent.step(state, action, reward, done, agent._last_logp, agent._last_value)
                    state = next_state
                    steps += 1

                    if done:
                        scores.append(steps)
                        break
                elif self.algorithm == "ppo":
                    action = agent.act(state)
                    next_state_raw, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_state = np.reshape(next_state_raw, (1, obs_dim))
                
                    agent.step(state, action, reward, done)
                    #    remembering and learning internally.
                    # agent.step(state, action, reward, done, agent._last_logp, agent._last_value)
                    state = next_state
                    steps += 1

                    if done:
                        scores.append(steps)
                        break
                    # TODO: 加入你的模型单独一步训练的代码（本质就是action,env interaction, agent.step,记录分数,看着上面写)

            # 检查早停条件
            if episode >= min_episodes:
                # 使用滑动窗口平均来减少波动影响
                window = min(window_size, len(scores))
                recent_avg = np.mean(scores[-window:])

                if recent_avg > best_score:
                    best_score = recent_avg
                    best_episode = episode
                    no_improve = 0
                    if verbose:
                        print(f"  Episode {episode}: New best avg ({recent_avg:.1f})")
                else:
                    no_improve += 1

                # 早停条件：连续patience个episode没有改善
                if no_improve >= patience:
                    if verbose:
                        print(f"  Early stopping at episode {episode}, "
                            f"best was {best_score:.1f} at episode {best_episode}")
                    break

                # 可选：如果分数已经达到很高，也可以提前停止
                if recent_avg >= stop_threshold and episode >= min_episodes:  # CartPole接近完美
                    if verbose:
                        print(f"  Early stopping: reached threshold {stop_threshold} ({recent_avg:.1f})")
                    break

        return agent, scores


def analyze_results_from_file(csv_path: str):
    """从CSV文件分析结果"""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 基本统计
    print(f"Total trials: {len(df)}")
    if 'success' in df.columns:
        success_rate = df['success'].mean() * 100
        print(f"Success rate: {success_rate:.1f}%")

    if 'avg_score' in df.columns:
        print(f"Average score: {df['avg_score'].mean():.2f}")
        print(f"Best score: {df['avg_score'].max():.2f}")
        print(f"Worst score: {df['avg_score'].min():.2f}")

    # 找到最佳参数组合
    if 'avg_score' in df.columns:
        best_idx = df['avg_score'].idxmax()
        best_row = df.loc[best_idx]

        print(f"\nBest trial ID: {int(best_row['trial_id'])}")
        print(f"Best score: {best_row['avg_score']:.2f}")

        # 输出最佳参数
        print("\nBest hyperparameters:")
        param_cols = [col for col in df.columns
                      if col not in ['trial_id', 'avg_score', 'success', 'error']]
        for col in param_cols:
            if col in best_row:
                print(f"  {col}: {best_row[col]}")

        score_over_475 = df[df['avg_score'] >= 475]
        score_over_475.to_csv(csv_path.replace('.csv', '_over_475.csv'))


def main():
    """主函数：支持并行和串行模式"""
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter tuning for RL agents')
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn','ppo','pdqn'], help='RL algorithm to tune')
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of hyperparameter trials')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of training episodes per trial')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel execution')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU cores - 1)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Enable GPU acceleration in parallel mode')

    # 添加早停相关参数
    parser.add_argument('--early-stop', action='store_true',
                        help='Enable early stopping during training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Patience for early stopping (number of episodes without improvement)')
    parser.add_argument('--min-episodes', type=int, default=50,
                        help='Minimum episodes before early stopping can be triggered')
    parser.add_argument('--stop-threshold', type=float, default=495.0,
                        help='Early stop threshold (stop if average score reaches this value)')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Window size for moving average in early stopping')
    
    args = parser.parse_args()

    # 创建调优器
    tuner = HyperparamTuner(args.algorithm, use_early_stopping=args.early_stop)

    TS = (str(time.localtime().tm_mon) + "m"
          + str(time.localtime().tm_mday) + "d"
          + str(time.localtime().tm_hour) + "h"
          + str(time.localtime().tm_min) + "min")

    # 显示早停配置信息
    if args.early_stop:
        print("\n" + "=" * 60)
        print("EARLY STOPPING CONFIGURATION")
        print("=" * 60)
        print(f"Early stopping: ENABLED")
        print(f"  - Patience: {args.patience} episodes")
        print(f"  - Minimum episodes: {args.min_episodes}")
        print(f"  - Stop threshold: {args.stop_threshold}")
        print(f"  - Window size: {args.window_size}")
        print("=" * 60 + "\n")

    if args.parallel:
        # 并行搜索
        print("enable parallel")
        results_df = tuner.run_search_parallel(
            n_trials=args.trials,
            num_episodes=args.episodes,
            max_workers=args.workers,
            use_gpu=args.use_gpu,
            early_stop_patience=args.patience if args.early_stop else None,
            early_stop_min_episodes=args.min_episodes if args.early_stop else None,
            early_stop_threshold=args.stop_threshold if args.early_stop else None,
            early_stop_window_size=args.window_size if args.early_stop else None,
            time_stamp=TS,
        )
    else:
        # 串行搜索
        print("Note: Early stopping is currently only supported in parallel mode.")
        if args.early_stop:  # 如果传入了早停参数
            print("Switching to parallel mode to enable early stopping...")
            results_df = tuner.run_search_parallel(
                n_trials=args.trials,
                num_episodes=args.episodes,
                max_workers=args.workers,
                use_gpu=args.use_gpu,
                early_stop_patience=args.patience,
                early_stop_min_episodes=args.min_episodes,
                time_stamp=TS,
                
            )
        else:
            print("run_search")
            results_df = tuner.run_search(
                n_trials=args.trials,
                num_episodes=args.episodes
            )

    # 保存最佳配置
    if len(results_df) > 0 and 'avg_score' in results_df.columns:
        success_df = results_df[results_df['success'] == True] if 'success' in results_df.columns else results_df

        if len(success_df) > 0:
            best_idx = success_df['avg_score'].idxmax()
            best_params = success_df.loc[best_idx].to_dict()

            mode = "parallel" if args.parallel or args.early_stop else "sequential"
            early_stop_str = "_earlystop" if args.early_stop else ""
            best_config_file = f"output/best_config/best_config_{args.algorithm}_{mode}{early_stop_str}_{TS}.json"
            
            # 添加早停参数到最佳配置中
            if args.early_stop:
                best_params['early_stopping_patience'] = args.patience
                best_params['early_stopping_min_episodes'] = args.min_episodes
                best_params['early_stopping_threshold'] = args.stop_threshold
                best_params['early_stopping_window'] = args.window_size

            with open(best_config_file, 'w') as f:
                json.dump(best_params, f, indent=2)

            print(f"\nBest configuration saved to: {best_config_file}")

            # 打印最佳配置摘要
            print("\n" + "=" * 60)
            print("BEST CONFIGURATION SUMMARY")
            print("=" * 60)
            print(f"Algorithm: {args.algorithm}")
            print(f"Mode: {mode}")
            print(f"Early Stopping: {'ENABLED' if args.early_stop else 'DISABLED'}")
            print(f"Best Score: {best_params['avg_score']:.2f}")
            print("\nKey Hyperparameters:")
            key_params = ['learning_rate', 'gamma', 'batch_size', 'memory_size']
            for param in key_params:
                if param in best_params:
                    print(f"  {param}: {best_params[param]}")

            # 如果启用了早停，显示相关信息
            if args.early_stop and 'early_stopped' in best_params:
                early_stopped = best_params.get('early_stopped', False)
                total_episodes = best_params.get('total_episodes', args.episodes)
                if early_stopped:
                    print(f"\nEarly Stopping Info:")
                    print(f"  - Stopped early at episode: {total_episodes}")
                    print(f"  - Saved episodes: {args.episodes - total_episodes}")
                    if 'improvement_ratio' in best_params:
                        ratio = best_params['improvement_ratio']
                        print(f"  - Improvement ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
    # use python hyperparameter_finding.py --algorithm dqn --trials 500 --episodes 300 --parallel --use-gpu
    # python hyperparameter_finding.py --algorithm dqn --trials 300 --episodes 500 --parallel --early-stop --use-gpu
    # 电脑有显卡就加 --use-gpu，没有就不加。电脑是多核处理器就加 --parallel，不是就不加。
    # python hyperparameter_finding.py --algorithm ppo --trials 11 --episodes 500 
    # python hyperparameter_finding.py --algorithm ppo --trials 11 --episodes 500 --parallel --early-stop --use-gpu