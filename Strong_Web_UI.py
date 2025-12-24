# webui_real.py
import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import subprocess
import json
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import torch
import io
import contextlib
from typing import List, Dict, Any, Tuple
import tempfile
import base64
from PIL import Image
import networkx as nx
from io import BytesIO

# 将当前目录添加到Python路径
sys.path.append('.')
sys.path.append('./agents')
sys.path.append('./scores')

# 导入现有的模块
try:
    # 尝试导入训练模块
    from train import (
        train_dqn, train_ppo, train_pdqn, train_with_config,
        evaluate_agent, plot_training_progress
    )

    # 导入智能体模块
    from agents.cartpole_dqn import DQNSolver, DQNConfig
    from agents.cartpole_ppo import PPOSolver, PPOConfig
    from agents.cartpole_dqn_priority import PDQNSolver, PDQNConfig
    from agents.cartpole_physics import PhysicsAgent, PhysicsConfig
    from agents.cartpole_ac import ACSolver, ACConfig

    # 导入超参数调优
    from hyperparameter_finding import HyperparamTuner, analyze_results_from_file

    # 导入其他模块
    from pretrain_student import pretrain_student
    from test_physics import main as test_physics
    from generate_data import generate_expert_dataset
    from train_student import train_student_agent

    import_success = True
except ImportError as e:
    st.error(f"导入模块失败: {e}")
    import_success = False

# 设置页面配置
st.set_page_config(
    page_title="CartPole RL Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置matplotlib中文字体
try:
    # Windows系统
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # MacOS系统备选
    # matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
except:
    pass

# 设置正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin: 0.5rem 0;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
    .layer-node {
        stroke: #333;
        stroke-width: 2px;
    }
    .layer-label {
        font-family: Arial, sans-serif;
        font-weight: bold;
        text-anchor: middle;
    }
</style>
""", unsafe_allow_html=True)


class NeuralNetworkDesigner:
    """神经网络设计器"""

    def __init__(self):
        self.default_layers = [
            {"type": "Linear", "units": 128, "activation": "ReLU", "dropout": 0.0},
            {"type": "Linear", "units": 64, "activation": "ReLU", "dropout": 0.0},
            {"type": "Linear", "units": 32, "activation": "ReLU", "dropout": 0.0}
        ]

        self.activation_functions = {
            "None": None,
            "ReLU": "ReLU",
            "LeakyReLU": "LeakyReLU",
            "Tanh": "Tanh",
            "Sigmoid": "Sigmoid",
            "GELU": "GELU",
            "ELU": "ELU",
            "Softplus": "Softplus"
        }

        self.layer_types = {
            "Linear": "全连接层",
            "Conv1d": "1D卷积层",
            "Conv2d": "2D卷积层",
            "LSTM": "LSTM层",
            "GRU": "GRU层"
        }

    def create_design_interface(self, default_input_dim=4, default_output_dim=2):
        """创建神经网络设计界面"""

        st.subheader("🧠 神经网络设计器")

        # 基本参数
        col1, col2 = st.columns(2)
        with col1:
            input_dim = st.number_input("输入维度", min_value=1, max_value=1000, value=default_input_dim)
        with col2:
            output_dim = st.number_input("输出维度", min_value=1, max_value=1000, value=default_output_dim)

        # 初始化层配置
        if 'nn_layers' not in st.session_state:
            st.session_state.nn_layers = self.default_layers.copy()
            st.session_state.input_dim = input_dim
            st.session_state.output_dim = output_dim

        # 层配置
        st.subheader("📐 网络层配置")

        # 显示当前所有层
        self.display_layers_ui()

        # 添加新层
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_layer_type = st.selectbox("层类型", list(self.layer_types.keys()))
        with col2:
            new_layer_units = st.number_input("神经元/单元数", min_value=1, max_value=1024, value=64)
        with col3:
            st.write("")
            st.write("")
            if st.button("➕ 添加层"):
                st.session_state.nn_layers.append({
                    "type": new_layer_type,
                    "units": new_layer_units,
                    "activation": "ReLU",
                    "dropout": 0.0
                })
                st.rerun()

        # 批量操作
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重置为默认", use_container_width=True):
                st.session_state.nn_layers = self.default_layers.copy()
                st.rerun()
        with col2:
            if st.button("🗑️ 清空所有层", use_container_width=True, type="secondary"):
                st.session_state.nn_layers = []
                st.rerun()

        # 可视化
        st.subheader("📊 网络结构可视化")
        self.visualize_network(input_dim, output_dim)

        # 参数统计
        self.display_network_stats(input_dim, output_dim)

        # 生成配置
        config = self.generate_config(input_dim, output_dim)

        # 代码生成
        st.subheader("💻 生成代码")
        if st.button("生成PyTorch代码"):
            code = self.generate_pytorch_code(input_dim, output_dim)
            st.code(code, language='python')

            # 下载按钮
            b64 = base64.b64encode(code.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="network.py">📥 下载代码</a>'
            st.markdown(href, unsafe_allow_html=True)

        return config

    def display_layers_ui(self):
        """显示层配置UI"""
        layers_to_remove = []

        for i, layer in enumerate(st.session_state.nn_layers):
            with st.expander(f"层 {i + 1}: {self.layer_types[layer['type']]}", expanded=i == 0):
                cols = st.columns([2, 2, 2, 1, 1])

                with cols[0]:
                    layer_type = st.selectbox(
                        "类型",
                        list(self.layer_types.keys()),
                        index=list(self.layer_types.keys()).index(layer['type']),
                        key=f"type_{i}"
                    )
                    st.session_state.nn_layers[i]['type'] = layer_type

                with cols[1]:
                    units = st.number_input(
                        "神经元数",
                        min_value=1,
                        max_value=4096,
                        value=layer['units'],
                        key=f"units_{i}"
                    )
                    st.session_state.nn_layers[i]['units'] = units

                with cols[2]:
                    activation = st.selectbox(
                        "激活函数",
                        list(self.activation_functions.keys()),
                        index=list(self.activation_functions.keys()).index(layer['activation']),
                        key=f"activation_{i}"
                    )
                    st.session_state.nn_layers[i]['activation'] = activation

                with cols[3]:
                    dropout = st.slider(
                        "Dropout",
                        min_value=0.0,
                        max_value=0.5,
                        value=layer['dropout'],
                        step=0.05,
                        key=f"dropout_{i}"
                    )
                    st.session_state.nn_layers[i]['dropout'] = dropout

                with cols[4]:
                    if st.button("❌", key=f"remove_{i}"):
                        layers_to_remove.append(i)

        # 删除层
        for i in sorted(layers_to_remove, reverse=True):
            st.session_state.nn_layers.pop(i)
        if layers_to_remove:
            st.rerun()

    def visualize_network(self, input_dim, output_dim):
        """可视化网络结构"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 设置中文字体
        try:
            # Windows系统
            plt.rcParams['font.sans-serif'] = ['SimHei']
            # MacOS系统备选
            # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        except:
            pass

        # 设置正常显示负号
        plt.rcParams['axes.unicode_minus'] = False

        # 计算布局
        layers = [{"type": "Input", "units": input_dim}] + \
                 st.session_state.nn_layers + \
                 [{"type": "Output", "units": output_dim}]

        n_layers = len(layers)
        max_units = max(layer['units'] for layer in layers)

        # 绘制每个层
        for i, layer in enumerate(layers):
            x_pos = i / (n_layers - 1) if n_layers > 1 else 0.5
            y_center = 0.5
            y_height = layer['units'] / max_units * 0.8

            # 绘制层框
            rect = plt.Rectangle(
                (x_pos - 0.04, y_center - y_height / 2),
                0.08, y_height,
                facecolor=self.get_layer_color(layer['type']),
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)

            # 添加层标签
            layer_name = layer['type']
            if layer_name in ["Linear", "Conv1d", "Conv2d", "LSTM", "GRU"]:
                layer_name = f"{layer_name}\n{layer['units']}"

            ax.text(
                x_pos, y_center,
                layer_name,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                color='white'
            )

            # 绘制神经元点（仅对前几层）
            if layer['units'] <= 20:  # 只在神经元较少时绘制点
                for j in range(layer['units']):
                    y_pos = y_center - y_height / 2 + (j + 0.5) * y_height / layer['units']
                    ax.plot(x_pos, y_pos, 'o', color='white', markersize=8)

            # 绘制连接线
            if i < n_layers - 1:
                next_layer = layers[i + 1]
                if layer['units'] <= 10 and next_layer['units'] <= 10:  # 只在神经元较少时绘制连接
                    for j in range(min(5, layer['units'])):
                        for k in range(min(5, next_layer['units'])):
                            y1 = y_center - y_height / 2 + (j + 0.5) * y_height / layer['units']
                            y2_height = next_layer['units'] / max_units * 0.8
                            y2 = y_center - y2_height / 2 + (k + 0.5) * y2_height / next_layer['units']
                            ax.plot(
                                [x_pos + 0.04, x_pos + 1 / (n_layers - 1) - 0.04],
                                [y1, y2],
                                color='#444444',  # 改为深灰色
                                alpha=0.6,  # 增加透明度
                                linewidth=1.0  # 增加线宽
                            )

        # 设置图形属性
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("神经网络结构图", fontsize=14, fontweight='bold')

        # 添加图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#2E86AB', label='输入层'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#A23B72', label='隐藏层'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#F18F01', label='输出层')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

        st.pyplot(fig)
        plt.close(fig)

        # 文本描述
        st.markdown("**结构描述:**")
        desc = f"Input({input_dim})"
        for layer in st.session_state.nn_layers:
            activation_str = f" + {layer['activation']}" if layer['activation'] != "None" else ""
            dropout_str = f" + Dropout({layer['dropout']})" if layer['dropout'] > 0 else ""
            desc += f" → {layer['type']}({layer['units']}){activation_str}{dropout_str}"
        desc += f" → Output({output_dim})"
        st.code(desc)

    def get_layer_color(self, layer_type):
        """获取层的颜色"""
        color_map = {
            "Input": "#2E86AB",  # 蓝色
            "Linear": "#A23B72",  # 紫色
            "Conv1d": "#F18F01",  # 橙色
            "Conv2d": "#C73E1D",  # 红色
            "LSTM": "#6A994E",  # 绿色
            "GRU": "#457B9D",  # 深蓝
            "Output": "#F18F01"  # 橙色
        }
        return color_map.get(layer_type, "#CCCCCC")

    def display_network_stats(self, input_dim, output_dim):
        """显示网络统计信息"""
        total_params = self.calculate_parameters(input_dim, output_dim)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("总层数", len(st.session_state.nn_layers) + 2)
        with col2:
            total_neurons = sum(layer['units'] for layer in st.session_state.nn_layers)
            st.metric("总神经元数", total_neurons)
        with col3:
            st.metric("总参数数", f"{total_params:,}")
        with col4:
            # 复杂度评级
            if total_params < 10000:
                complexity = "🟢 简单"
            elif total_params < 100000:
                complexity = "🟡 中等"
            elif total_params < 1000000:
                complexity = "🟠 复杂"
            else:
                complexity = "🔴 极复杂"
            st.metric("复杂度", complexity)

    def calculate_parameters(self, input_dim, output_dim):
        """计算总参数数量"""
        total_params = 0
        prev_units = input_dim

        for layer in st.session_state.nn_layers:
            if layer['type'] == "Linear":
                # 权重参数 + 偏置参数
                total_params += prev_units * layer['units'] + layer['units']
            elif layer['type'] in ["LSTM", "GRU"]:
                # 简化计算：4 * (input_size + hidden_size) * hidden_size
                total_params += 4 * (prev_units + layer['units']) * layer['units']

            prev_units = layer['units']

        # 输出层
        total_params += prev_units * output_dim + output_dim

        return total_params

    def generate_config(self, input_dim, output_dim):
        """生成网络配置"""
        return {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "layers": st.session_state.nn_layers.copy(),
            "total_params": self.calculate_parameters(input_dim, output_dim)
        }

    def generate_pytorch_code(self, input_dim, output_dim):
        """生成PyTorch网络代码"""
        code_lines = []
        code_lines.append("import torch")
        code_lines.append("import torch.nn as nn")
        code_lines.append("import torch.nn.functional as F")
        code_lines.append("")
        code_lines.append("")
        code_lines.append("class CustomNetwork(nn.Module):")
        code_lines.append(f"    def __init__(self, input_dim={input_dim}, output_dim={output_dim}):")
        code_lines.append("        super().__init__()")
        code_lines.append("        ")
        code_lines.append("        # 构建网络层")

        # 添加层
        prev_dim = input_dim
        for i, layer in enumerate(st.session_state.nn_layers):
            if layer['type'] == "Linear":
                code_lines.append(f"        self.fc{i + 1} = nn.Linear({prev_dim}, {layer['units']})")
                prev_dim = layer['units']
            elif layer['type'] == "LSTM":
                code_lines.append(f"        self.lstm{i + 1} = nn.LSTM({prev_dim}, {layer['units']}, batch_first=True)")
                prev_dim = layer['units']
            elif layer['type'] == "GRU":
                code_lines.append(f"        self.gru{i + 1} = nn.GRU({prev_dim}, {layer['units']}, batch_first=True)")
                prev_dim = layer['units']

        # 输出层
        code_lines.append(f"        self.output = nn.Linear({prev_dim}, output_dim)")
        code_lines.append("        ")

        # Dropout层
        has_dropout = any(layer['dropout'] > 0 for layer in st.session_state.nn_layers)
        if has_dropout:
            for i, layer in enumerate(st.session_state.nn_layers):
                if layer['dropout'] > 0:
                    code_lines.append(f"        self.dropout{i + 1} = nn.Dropout({layer['dropout']})")
            code_lines.append("        ")

        code_lines.append("    def forward(self, x):")

        # 前向传播
        for i, layer in enumerate(st.session_state.nn_layers):
            if layer['type'] == "Linear":
                code_lines.append(f"        x = self.fc{i + 1}(x)")
                if layer['activation'] != "None":
                    if layer['activation'] == "LeakyReLU":
                        code_lines.append(f"        x = F.leaky_relu(x)")
                    else:
                        code_lines.append(f"        x = F.{layer['activation'].lower()}(x)")
                if layer['dropout'] > 0:
                    code_lines.append(f"        x = self.dropout{i + 1}(x)")
            elif layer['type'] in ["LSTM", "GRU"]:
                layer_type = layer['type'].lower()
                code_lines.append(f"        x, _ = self.{layer_type}{i + 1}(x)")
                if layer['dropout'] > 0:
                    code_lines.append(f"        x = self.dropout{i + 1}(x)")

        code_lines.append("        x = self.output(x)")
        code_lines.append("        return x")
        code_lines.append("")
        code_lines.append("")
        code_lines.append("# 使用示例:")
        code_lines.append("# network = CustomNetwork()")
        code_lines.append("# print(network)")
        code_lines.append("# print(f'总参数: {sum(p.numel() for p in network.parameters()):,}')")

        return "\n".join(code_lines)


class RealCartPoleWebUI:
    def __init__(self):
        self.models_dir = "models"
        self.data_dir = "data"
        self.output_dir = "output"
        self.scores_dir = "scores"
        self.configs_dir = "configs"
        self.nn_designer = NeuralNetworkDesigner()
        self.initialize_directories()

    def initialize_directories(self):
        """确保所有必要的目录都存在"""
        for directory in [self.models_dir, self.data_dir,
                          self.output_dir, self.scores_dir, self.configs_dir]:
            os.makedirs(directory, exist_ok=True)

    def get_available_models(self):
        """获取所有可用的模型文件"""
        models = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith(".torch") or file.endswith(".json"):
                    models.append(file)
        return sorted(models)

    def get_training_history(self):
        """获取训练历史（从CSV文件）"""
        history_files = []
        if os.path.exists(self.scores_dir):
            for file in os.listdir(self.scores_dir):
                if file.endswith(".csv"):
                    history_files.append(file)
        return history_files

    def run_capture_output(self, func, *args, **kwargs):
        """运行函数并捕获输出"""
        output = io.StringIO()
        result = None

        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = func(*args, **kwargs)
            return True, output.getvalue(), result
        except Exception as e:
            error_msg = f"错误: {str(e)}\n\n{output.getvalue()}"
            return False, error_msg, None


def main():
    if not import_success:
        st.error("无法导入必要的模块。请确保所有依赖已安装且文件结构正确。")
        st.code("""
        请运行以下命令安装依赖：
        pip install streamlit plotly pandas numpy matplotlib torch gymnasium networkx pillow
        """)
        return

    # 创建UI实例
    ui = RealCartPoleWebUI()

    # 标题
    st.markdown("<h1 class='main-header'>🤖 CartPole 强化学习智能体系统</h1>",
                unsafe_allow_html=True)

    # 侧边栏 - 主菜单
    st.sidebar.title("🎮 导航")
    menu = st.sidebar.radio(
        "选择功能",
        ["🏠 仪表盘", "🚀 训练智能体", "📊 评估模型",
         "⚙️ 超参数调优", "📁 模型管理", "📈 训练历史", "🎓 模仿学习", "🧠 神经网络设计器"]
    )

    # 仪表盘
    if menu == "🏠 仪表盘":
        show_dashboard(ui)

    # 训练智能体
    elif menu == "🚀 训练智能体":
        show_training_interface(ui)

    # 评估模型
    elif menu == "📊 评估模型":
        show_evaluation_interface(ui)

    # 超参数调优
    elif menu == "⚙️ 超参数调优":
        show_hyperparameter_tuning(ui)

    # 模型管理
    elif menu == "📁 模型管理":
        show_model_management(ui)

    # 训练历史
    elif menu == "📈 训练历史":
        show_training_history(ui)

    # 模仿学习
    elif menu == "🎓 模仿学习":
        show_imitation_learning_interface(ui)

    # 神经网络设计器
    elif menu == "🧠 神经网络设计器":
        show_neural_network_designer(ui)


def show_dashboard(ui):
    """显示仪表盘"""
    st.subheader("📊 系统概览")

    col1, col2, col3 = st.columns(3)

    with col1:
        models = ui.get_available_models()
        st.metric("可用模型数量", len(models))
        if models:
            st.caption(f"最新模型: {models[-1] if models else '无'}")

    with col2:
        history_files = ui.get_training_history()
        st.metric("训练记录数", len(history_files))

    with col3:
        data_exists = os.path.exists("data/expert_data.pt")
        st.metric("专家数据", "✅ 已存在" if data_exists else "❌ 未生成")

    # 快速操作
    st.subheader("⚡ 快速操作")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 生成专家数据", use_container_width=True):
            with st.spinner("正在生成专家数据..."):
                success, output, _ = ui.run_capture_output(generate_expert_dataset)
                if success:
                    st.success("✅ 专家数据生成完成！")
                    st.code(output[:500])
                else:
                    st.error(f"❌ 生成失败: {output}")

    with col2:
        if st.button("🎯 测试物理控制器", use_container_width=True):
            with st.spinner("正在测试物理控制器..."):
                success, output, _ = ui.run_capture_output(test_physics)
                if success:
                    st.success("✅ 物理控制器测试完成！")
                    st.code(output[:500])
                else:
                    st.error(f"❌ 测试失败: {output}")

    with col3:
        if st.button("🧠 预训练学生模型", use_container_width=True):
            with st.spinner("正在预训练学生模型..."):
                success, output, _ = ui.run_capture_output(pretrain_student)
                if success:
                    st.success("✅ 预训练完成！")
                    st.code(output[:500])
                else:
                    st.error(f"❌ 预训练失败: {output}")

    with col4:
        if st.button("📋 查看系统信息", use_container_width=True):
            st.info("系统状态信息")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Python版本:", sys.version.split()[0])
                st.write("PyTorch版本:", torch.__version__)
                st.write("CUDA可用:", torch.cuda.is_available())
            with col_b:
                st.write("模型目录:", ui.models_dir)
                st.write("数据目录:", ui.data_dir)
                st.write("输出目录:", ui.output_dir)

    # 最近模型
    st.subheader("📁 最近模型")
    models = ui.get_available_models()[-10:]  # 显示最近10个
    if models:
        for model in reversed(models):
            model_path = os.path.join(ui.models_dir, model)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.write(f"📄 {model}")
            with col2:
                st.write(f"大小: {size_mb:.2f} MB")
            with col3:
                if st.button(f"评估", key=f"eval_{model}", use_container_width=True):
                    st.session_state['eval_model'] = model
                    st.rerun()
    else:
        st.info("暂无模型，请先训练一个模型")


def show_training_interface(ui):
    """显示训练界面"""
    st.header("🚀 训练智能体")

    # 算法选择
    algorithm = st.selectbox(
        "选择算法",
        ["dqn", "pdqn", "ppo", "ac", "physics"],
        format_func=lambda x: {
            "dqn": "DQN",
            "pdqn": "PDQN (优先级DQN)",
            "ppo": "PPO",
            "ac": "Actor-Critic (AC)",
            "physics": "Physics (教师)"
        }[x],
        index=0
    )

    # 训练参数
    col1, col2 = st.columns(2)
    with col1:
        episodes = st.number_input("训练回合数", min_value=1, max_value=10000, value=200)
    with col2:
        render = st.checkbox("显示训练画面", value=False,
                             help="注意：显示画面会显著降低训练速度，建议仅在调试时使用")

    # 神经网络配置（除了physics算法）
    if algorithm != "physics":
        st.subheader("🧠 神经网络配置")
        nn_config = ui.nn_designer.create_design_interface(default_input_dim=4, default_output_dim=2)

    # 算法特定参数
    st.subheader("⚙️ 算法参数")

    if algorithm == "dqn":
        config = configure_dqn_params()
    elif algorithm == "pdqn":
        config = configure_pdqn_params()
    elif algorithm == "ppo":
        config = configure_ppo_params()
    elif algorithm == "ac":
        config = configure_ac_params()
    elif algorithm == "physics":
        config = configure_physics_params()

    # 开始训练按钮
    if st.button("🎬 开始训练", type="primary", use_container_width=True):
        if algorithm != "physics":
            st.info(f"使用自定义神经网络结构训练{algorithm.upper()}...")
            # 将神经网络配置合并到训练配置中
            if hasattr(config, '__dict__'):
                config.__dict__['network_config'] = nn_config
            elif isinstance(config, dict):
                config['network_config'] = nn_config
        start_training(algorithm, episodes, config, render, ui=ui)


def show_evaluation_interface(ui):
    """显示评估界面"""
    st.header("📊 评估模型")

    # 选择模型
    models = ui.get_available_models()

    if not models:
        st.warning("⚠️ 没有找到模型文件。请先训练一个模型。")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        selected_model = st.selectbox("选择要评估的模型", models)

    with col2:
        # 自动检测算法类型
        if "dqn" in selected_model.lower():
            default_algo = "dqn"
        elif "ppo" in selected_model.lower():
            default_algo = "ppo"
        elif "pdqn" in selected_model.lower():
            default_algo = "pdqn"
        elif "physics" in selected_model.lower():
            default_algo = "physics"
        else:
            default_algo = "dqn"

        algorithm = st.selectbox(
            "算法类型",
            ["dqn", "pdqn", "ppo", "physics"],
            index=["dqn", "pdqn", "ppo", "physics"].index(default_algo)
        )

    # 评估参数
    st.subheader("📋 评估参数")

    col1, col2, col3 = st.columns(3)

    with col1:
        render_eval = st.checkbox("显示评估画面", value=False)

        # 根据是否显示评估画面，动态调整评估回合数限制
        if render_eval:
            max_episodes = 5  # 显示画面时限制为5次
            default_episodes = 3  # 默认3次，避免时间太长
            warning_msg = "⚠️ 显示评估画面时，建议评估回合数不超过5次，以免动画时间太长"
            st.warning(warning_msg)
        else:
            max_episodes = 1000  # 不显示画面时可以更多
            default_episodes = 50

        eval_episodes = st.number_input(
            "评估回合数",
            min_value=1,
            max_value=max_episodes,
            value=default_episodes,
            help=f"最大评估回合数: {max_episodes}次"
        )

    with col2:
        fps = st.slider("帧率 (FPS)", 1, 120, 60, 5)

    with col3:
        use_agent_directly = st.checkbox("直接使用agent实例", value=False)

    # 实时更新信息
    if render_eval:
        estimated_time = eval_episodes * 10  # 假设每次评估大约10秒
        if estimated_time > 30:
            st.error(f"⚠️ 警告：评估预计需要约{estimated_time}秒，可能会很慢！")
        else:
            st.info(f"评估预计需要约{estimated_time}秒")

    # 开始评估按钮
    if st.button("🔍 开始评估", type="primary", use_container_width=True):
        model_path = os.path.join(ui.models_dir, selected_model)

        # 如果选择了显示画面但回合数太多，再次确认
        if render_eval and eval_episodes > 5:
            eval_episodes = min(eval_episodes, 5)
            st.info(f"已自动将评估次数调整为5次")

        with st.spinner("正在评估模型..."):
            try:
                # 创建进度显示
                progress_bar = st.progress(0)
                status_text = st.empty()

                # 模拟进度更新（在实际评估中，我们可以通过回调更新进度）
                if render_eval:
                    status_text.text("正在评估（显示画面中）...")
                    # 显示画面时，每回合更新进度
                    for i in range(eval_episodes):
                        progress = int((i + 1) / eval_episodes * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"正在评估第 {i + 1}/{eval_episodes} 回合...")
                        time.sleep(0.5)  # 模拟评估时间
                else:
                    status_text.text("正在评估（不显示画面）...")

                # 实际执行评估
                if use_agent_directly and algorithm == "physics":
                    # 对于Physics Agent，直接创建实例
                    config = PhysicsConfig()
                    agent = PhysicsAgent(4, 2, cfg=config)
                    agent.load(model_path)

                    scores, avg_score = evaluate_agent(
                        algorithm=algorithm,
                        episodes=eval_episodes,
                        render=render_eval,
                        fps=fps,
                        if_agent=True,
                        agent=agent
                    )
                else:
                    scores, avg_score = evaluate_agent(
                        model_path=model_path,
                        algorithm=algorithm,
                        episodes=eval_episodes,
                        render=render_eval,
                        fps=fps
                    )

                progress_bar.progress(100)
                status_text.text("评估完成！")

                # 显示结果
                st.success(f"✅ 评估完成！平均得分: {avg_score:.2f}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均得分", f"{avg_score:.2f}")
                with col2:
                    st.metric("最高得分", max(scores))
                with col3:
                    st.metric("最低得分", min(scores))
                with col4:
                    st.metric("标准差", f"{np.std(scores):.2f}")

                # 显示得分分布图
                tab1, tab2 = st.tabs(["得分趋势", "分布统计"])

                with tab1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(scores))),
                        y=scores,
                        mode='lines+markers',
                        name='每回合得分',
                        line=dict(color='#1E88E5')
                    ))
                    fig.add_hline(y=avg_score, line_dash="dash",
                                  line_color="red", annotation_text=f"平均: {avg_score:.1f}")
                    fig.update_layout(
                        title="得分趋势图",
                        xaxis_title="回合",
                        yaxis_title="得分"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    col1, col2 = st.columns(2)

                    with col1:
                        fig = px.histogram(x=scores, nbins=min(10, len(scores)),
                                           title="得分分布直方图")
                        fig.update_layout(xaxis_title="得分", yaxis_title="频次")
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Box(y=scores, name='得分分布'))
                        fig.update_layout(title="得分箱线图", yaxis_title="得分")
                        st.plotly_chart(fig, use_container_width=True)

                # 显示详细得分
                with st.expander("📋 查看详细得分"):
                    score_df = pd.DataFrame({
                        "回合": range(1, len(scores) + 1),
                        "得分": scores
                    })
                    st.dataframe(score_df, use_container_width=True)

                    # 计算统计数据
                    stats_df = pd.DataFrame({
                        "统计项": ["平均分", "中位数", "标准差", "最大值", "最小值", "成功率"],
                        "数值": [
                            f"{avg_score:.2f}",
                            f"{np.median(scores):.2f}",
                            f"{np.std(scores):.2f}",
                            f"{max(scores)}",
                            f"{min(scores)}",
                            f"{(np.array(scores) >= 475).sum() / len(scores) * 100:.1f}%"  # 假设475分以上算成功
                        ]
                    })
                    st.dataframe(stats_df, use_container_width=True)

            except Exception as e:
                st.error(f"❌ 评估失败: {str(e)}")
                import traceback
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())


def show_hyperparameter_tuning(ui):
    """显示超参数调优界面"""
    st.header("⚙️ 超参数调优")

    col1, col2 = st.columns(2)

    with col1:
        algorithm = st.selectbox(
            "选择要调优的算法",
            ["dqn", "ppo", "pdqn"],
            index=0
        )

    with col2:
        tuning_mode = st.radio(
            "调优模式",
            ["串行搜索", "并行搜索"],
            horizontal=True
        )

    # 调优参数
    st.subheader("🔧 调优参数")

    trials = st.slider("试验次数", 10, 500, 30, 10)
    episodes_per_trial = st.slider("每试验回合数", 50, 500, 200, 50)

    # 高级选项
    with st.expander("高级选项"):
        col1, col2 = st.columns(2)

        with col1:
            early_stop = st.checkbox("启用早停", value=False)
            use_gpu = st.checkbox("使用GPU加速", value=torch.cuda.is_available())

        with col2:
            if early_stop:
                patience = st.number_input("早停耐心值", min_value=5, max_value=100, value=20)
                min_episodes = st.number_input("最小回合数", min_value=10, max_value=200, value=50)

    # 开始调优按钮
    if st.button("🔬 开始超参数调优", type="primary", use_container_width=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with st.spinner(f"正在进行超参数调优 ({trials}次试验)..."):
            try:
                # 创建调优器
                tuner = HyperparamTuner(algorithm, use_early_stopping=early_stop)

                progress_bar = st.progress(0)
                status_text = st.empty()

                # 执行调优
                if tuning_mode == "并行搜索":
                    results_df = tuner.run_search_parallel(
                        n_trials=trials,
                        num_episodes=episodes_per_trial,
                        use_gpu=use_gpu,
                        early_stop_patience=patience if early_stop else None,
                        early_stop_min_episodes=min_episodes if early_stop else None,
                        time_stamp=timestamp,
                    )
                else:
                    results_df = tuner.run_search(
                        n_trials=trials,
                        num_episodes=episodes_per_trial
                    )

                progress_bar.progress(100)
                status_text.text("调优完成！")

                # 显示结果
                st.success(f"✅ 超参数调优完成！共进行了 {trials} 次试验")

                # 找到最佳参数
                if 'avg_score' in results_df.columns:
                    success_df = results_df[
                        results_df['success'] == True] if 'success' in results_df.columns else results_df

                    if len(success_df) > 0:
                        best_idx = success_df['avg_score'].idxmax()
                        best_result = success_df.loc[best_idx].to_dict()

                        st.subheader("🏆 最佳参数组合")

                        # 显示最佳参数表格
                        best_params_df = pd.DataFrame([{k: v for k, v in best_result.items()
                                                        if k not in ['trial_id', 'success', 'error']}])
                        st.dataframe(best_params_df, use_container_width=True)

                        # 保存最佳配置
                        config_path = f"configs/best_{algorithm}_config_{timestamp}.json"
                        with open(config_path, "w") as f:
                            json.dump(best_result, f, indent=2)

                        st.info(f"最佳配置已保存到: {config_path}")

                        # 显示分数分布
                        fig = px.histogram(success_df, x='avg_score',
                                           title="试验得分分布")
                        st.plotly_chart(fig, use_container_width=True)

                        # 显示分数趋势
                        if 'trial_id' in success_df.columns:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=success_df['trial_id'],
                                y=success_df['avg_score'],
                                mode='lines+markers',
                                name='试验得分'
                            ))
                            fig.update_layout(
                                title="试验得分趋势",
                                xaxis_title="试验ID",
                                yaxis_title="平均得分"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # 提供下载链接
                results_file = f"output/param_table/hyperparam_results_{algorithm}"
                if tuning_mode == "并行搜索":
                    results_file += f"_parallel_{timestamp}.csv"
                else:
                    results_file += ".csv"

                if os.path.exists(results_file):
                    with open(results_file, "rb") as f:
                        st.download_button(
                            label="📥 下载完整结果CSV",
                            data=f,
                            file_name=os.path.basename(results_file),
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"❌ 超参数调优失败: {str(e)}")


def show_model_management(ui):
    """显示模型管理界面"""
    st.header("📁 模型管理")

    # 获取模型列表
    models = ui.get_available_models()

    if not models:
        st.warning("没有找到模型文件")
        return

    # 模型列表
    st.subheader("📋 模型列表")

    # 创建数据框
    model_data = []
    for model in models:
        model_path = os.path.join(ui.models_dir, model)
        size_kb = os.path.getsize(model_path) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))

        # 识别算法类型
        if "dqn" in model.lower():
            algo = "DQN"
        elif "ppo" in model.lower():
            algo = "PPO"
        elif "pdqn" in model.lower():
            algo = "PDQN"
        elif "physics" in model.lower():
            algo = "Physics"
        elif "student" in model.lower():
            algo = "Student"
        elif "ac" in model.lower():
            algo = "AC"
        else:
            algo = "Unknown"

        model_data.append({
            "文件名": model,
            "算法": algo,
            "大小 (KB)": f"{size_kb:.1f}",
            "修改时间": mtime.strftime("%Y-%m-%d %H:%M"),
            "操作": model  # 用于操作按钮
        })

    # 显示表格
    df = pd.DataFrame(model_data)
    edited_df = st.data_editor(
        df,
        column_config={
            "操作": st.column_config.Column(
                "操作",
                width="medium",
                help="选择操作",
            )
        },
        disabled=["文件名", "算法", "大小 (KB)", "修改时间"],
        hide_index=True,
        use_container_width=True
    )

    # 批量操作
    st.subheader("🛠️ 批量操作")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_for_delete = st.multiselect("选择要删除的模型", models)
        if st.button("🗑️ 批量删除", type="secondary"):
            for model in selected_for_delete:
                model_path = os.path.join(ui.models_dir, model)
                try:
                    os.remove(model_path)
                    st.success(f"已删除: {model}")
                except Exception as e:
                    st.error(f"删除失败 {model}: {e}")
            st.rerun()

    with col2:
        if st.button("📥 打包下载所有模型", type="secondary"):
            import zipfile
            zip_path = f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for model in models:
                    model_path = os.path.join(ui.models_dir, model)
                    zipf.write(model_path, model)

            with open(zip_path, "rb") as f:
                st.download_button(
                    label="点击下载ZIP文件",
                    data=f,
                    file_name=os.path.basename(zip_path),
                    mime="application/zip"
                )

    with col3:
        if st.button("🔄 刷新列表", type="secondary"):
            st.rerun()


def show_training_history(ui):
    """显示训练历史"""
    st.header("📈 训练历史")

    # 获取历史文件
    history_files = ui.get_training_history()

    if not history_files:
        st.warning("没有找到训练历史文件")
        return

    # 选择历史文件
    selected_file = st.selectbox("选择训练记录", history_files)

    if selected_file:
        file_path = os.path.join(ui.scores_dir, selected_file)

        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 显示基本信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总回合数", len(df))
            with col2:
                st.metric("平均得分", f"{df['Score'].mean():.2f}")
            with col3:
                st.metric("最高得分", df['Score'].max())
            with col4:
                st.metric("最后得分", df['Score'].iloc[-1] if len(df) > 0 else 0)

            # 显示数据表
            with st.expander("📊 查看详细数据"):
                st.dataframe(df, use_container_width=True)

            # 绘制图表
            st.subheader("📈 训练曲线")

            tab1, tab2, tab3 = st.tabs(["原始曲线", "移动平均", "统计分析"])

            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Episode'],
                    y=df['Score'],
                    mode='lines',
                    name='原始得分',
                    line=dict(color='#1E88E5')
                ))
                fig.update_layout(
                    title="训练得分原始曲线",
                    xaxis_title="训练回合",
                    yaxis_title="得分"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                window_size = st.slider("移动平均窗口", 5, 100, 20, 5)
                df['Moving_Avg'] = df['Score'].rolling(window=window_size).mean()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Episode'],
                    y=df['Score'],
                    mode='lines',
                    name='原始得分',
                    line=dict(color='lightblue', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df['Episode'],
                    y=df['Moving_Avg'],
                    mode='lines',
                    name=f'{window_size}回合移动平均',
                    line=dict(color='red', width=2)
                ))
                fig.update_layout(
                    title="移动平均曲线",
                    xaxis_title="训练回合",
                    yaxis_title="得分",
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                col1, col2 = st.columns(2)

                with col1:
                    # 直方图
                    fig = px.histogram(df, x='Score', nbins=30,
                                       title="得分分布直方图")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # 箱线图
                    fig = go.Figure()
                    fig.add_trace(go.Box(y=df['Score'], name='得分分布'))
                    fig.update_layout(title="得分箱线图", yaxis_title="得分")
                    st.plotly_chart(fig, use_container_width=True)

            # 导出选项
            st.subheader("💾 数据导出")
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 下载CSV数据",
                data=csv,
                file_name=selected_file,
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"读取文件失败: {e}")


def show_imitation_learning_interface(ui):
    """显示模仿学习界面"""
    st.header("🎓 模仿学习")

    st.markdown("""
    **模仿学习流程:**
    1. 🎯 首先需要专家数据（物理老师的数据）
    2. 🧠 用专家数据预训练学生模型
    3. 🚀 微调预训练的学生模型
    4. 📊 评估学生模型性能
    """)

    # 步骤选择
    step = st.radio(
        "选择步骤",
        ["1. 生成专家数据", "2. 预训练学生", "3. 微调学生", "4. 评估学生"],
        horizontal=True
    )

    if step == "1. 生成专家数据":
        st.subheader("🎯 生成专家数据")

        col1, col2 = st.columns(2)

        with col1:
            num_samples = st.number_input("样本数量", min_value=100, max_value=100000,
                                          value=10000, step=1000)
            save_path = st.text_input("保存路径", value="data/expert_data.pt")

        with col2:
            # 物理老师参数
            theta_coef = st.slider("角度系数", 0.0, 2.0, 1.0, 0.1)
            omega_coef = st.slider("角速度系数", 0.0, 2.0, 1.0, 0.1)
            pos_coef = st.slider("位置系数", 0.0, 1.0, 0.1, 0.05)
            vel_coef = st.slider("速度系数", 0.0, 1.0, 0.1, 0.05)

        if st.button("🚀 开始生成专家数据", type="primary"):
            with st.spinner("正在生成专家数据..."):
                try:
                    # 创建临时配置文件
                    config = PhysicsConfig(
                        theta_coef=theta_coef,
                        omega_coef=omega_coef,
                        pos_coef=pos_coef,
                        vel_coef=vel_coef
                    )

                    # 这里需要调用生成数据的函数
                    # 由于generate_expert_dataset函数需要PhysicsConfig，我们直接调用
                    success, output, _ = ui.run_capture_output(
                        generate_expert_dataset,
                        num_samples=num_samples,
                        save_path=save_path
                    )

                    if success:
                        st.success("✅ 专家数据生成完成！")
                        st.code(output[:500])

                        # 显示数据统计
                        if os.path.exists(save_path):
                            data = torch.load(save_path)
                            st.info(f"""
                            **数据统计:**
                            - 状态数据形状: {data['states'].shape}
                            - 动作数据形状: {data['actions'].shape}
                            - 样本数量: {len(data['states'])}
                            """)
                    else:
                        st.error(f"❌ 生成失败: {output}")

                except Exception as e:
                    st.error(f"❌ 错误: {str(e)}")

    elif step == "2. 预训练学生":
        st.subheader("🧠 预训练学生模型")

        col1, col2 = st.columns(2)

        with col1:
            epochs = st.number_input("训练轮数", min_value=10, max_value=1000,
                                     value=50, step=10)
            batch_size = st.number_input("批次大小", min_value=16, max_value=512,
                                         value=64)

        with col2:
            lr = st.number_input("学习率", min_value=1e-5, max_value=1e-1,
                                 value=0.001, format="%.5f")
            data_path = st.text_input("专家数据路径", value="data/expert_data.pt")

        if not os.path.exists(data_path):
            st.warning(f"⚠️ 找不到专家数据文件: {data_path}")
            st.info("请先完成第1步生成专家数据")

        if st.button("🧠 开始预训练", type="primary"):
            with st.spinner("正在预训练学生模型..."):
                try:
                    success, output, _ = ui.run_capture_output(
                        pretrain_student,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr
                    )

                    if success:
                        st.success("✅ 预训练完成！")
                        st.code(output[:500])

                        if os.path.exists("models/pretrained_dqn.torch"):
                            st.info("预训练模型已保存为: models/pretrained_dqn.torch")
                    else:
                        st.error(f"❌ 预训练失败: {output}")

                except Exception as e:
                    st.error(f"❌ 错误: {str(e)}")

    elif step == "3. 微调学生":
        st.subheader("🚀 微调学生模型")

        col1, col2 = st.columns(2)

        with col1:
            episodes = st.number_input("微调回合数", min_value=10, max_value=1000,
                                       value=20, step=5)

        with col2:
            pretrained_path = st.text_input("预训练模型路径",
                                            value="models/pretrained_dqn.torch")

        if not os.path.exists(pretrained_path):
            st.warning(f"⚠️ 找不到预训练模型: {pretrained_path}")
            st.info("请先完成第2步预训练学生模型")

        if st.button("🚀 开始微调", type="primary"):
            with st.spinner("正在微调学生模型..."):
                try:
                    success, output, agent = ui.run_capture_output(
                        train_student_agent,
                        num_episodes=episodes
                    )

                    if success and agent:
                        st.success("✅ 微调完成！")
                        st.code(output[:500])
                        st.info("学生模型已保存为: models/student_final.torch")
                    else:
                        st.error(f"❌ 微调失败: {output}")

                except Exception as e:
                    st.error(f"❌ 错误: {str(e)}")

    elif step == "4. 评估学生":
        st.subheader("📊 评估学生模型")

        model_path = st.text_input("学生模型路径", value="models/student_final.torch")

        if not os.path.exists(model_path):
            st.warning(f"⚠️ 找不到学生模型: {model_path}")
            st.info("请先完成第3步微调学生模型")

        col1, col2 = st.columns(2)

        with col1:
            render = st.checkbox("显示评估画面", value=False)
            # 根据是否显示画面调整评估次数
            if render:
                max_episodes = 5
                default_episodes = 3
                st.warning("⚠️ 显示评估画面时，建议评估回合数不超过5次")
            else:
                max_episodes = 50
                default_episodes = 10
            eval_episodes = st.number_input(
                "评估回合数",
                min_value=1,
                max_value=max_episodes,
                value=default_episodes
            )

        if st.button("🔍 开始评估", type="primary"):
            # 如果选择了显示画面但回合数太多，自动调整
            if render and eval_episodes > 5:
                eval_episodes = 5
                st.info("已自动将评估次数调整为5次（显示画面时不宜过多）")

            with st.spinner("正在评估学生模型..."):
                try:
                    scores, avg_score = evaluate_agent(
                        model_path=model_path,
                        algorithm="dqn",
                        episodes=eval_episodes,
                        render=render,
                        fps=60
                    )

                    st.success(f"✅ 评估完成！平均得分: {avg_score:.2f}")

                    # 显示结果
                    col1, col2 = st.columns(2)

                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(len(scores))),
                            y=scores,
                            mode='lines+markers',
                            name='学生得分'
                        ))
                        fig.update_layout(
                            title="学生模型评估结果",
                            xaxis_title="回合",
                            yaxis_title="得分"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.metric("平均得分", f"{avg_score:.2f}")
                        st.metric("最高得分", max(scores))
                        st.metric("最低得分", min(scores))
                        st.metric("稳定性", f"{np.std(scores):.2f}")

                except Exception as e:
                    st.error(f"❌ 评估失败: {str(e)}")


def show_neural_network_designer(ui):
    """显示独立的神经网络设计器"""
    st.header("🧠 神经网络设计器")

    st.markdown("""
    ### 设计你自己的神经网络

    在这里你可以：
    1. **添加/删除网络层**
    2. **配置每层的神经元数量和类型**
    3. **选择激活函数和Dropout率**
    4. **可视化网络结构**
    5. **生成PyTorch代码**

    设计好的网络可以用于训练DQN、PPO等强化学习算法。
    """)

    # 创建设计器界面
    nn_config = ui.nn_designer.create_design_interface()

    # 保存配置选项
    st.subheader("💾 保存配置")

    col1, col2 = st.columns(2)

    with col1:
        config_name = st.text_input("配置名称", value="custom_network")

    with col2:
        if st.button("💾 保存配置", use_container_width=True):
            config_path = f"configs/{config_name}.json"
            with open(config_path, "w") as f:
                json.dump(nn_config, f, indent=2)
            st.success(f"配置已保存到: {config_path}")

    # 加载配置选项
    st.subheader("📂 加载配置")

    # 获取现有配置
    config_files = []
    if os.path.exists("configs"):
        config_files = [f for f in os.listdir("configs") if f.endswith(".json")]

    if config_files:
        selected_config = st.selectbox("选择已有配置", config_files)

        if st.button("📂 加载配置", use_container_width=True):
            config_path = os.path.join("configs", selected_config)
            with open(config_path, "r") as f:
                loaded_config = json.load(f)

            # 更新session_state中的层配置
            if "layers" in loaded_config:
                st.session_state.nn_layers = loaded_config["layers"]
                st.success("配置已加载！")
                st.rerun()
    else:
        st.info("暂无保存的配置")


def configure_dqn_params():
    """配置DQN参数"""
    col1, col2 = st.columns(2)

    with col1:
        lr = st.number_input("学习率", min_value=1e-5, max_value=1e-1,
                             value=0.0005, format="%.5f")
        gamma = st.slider("折扣因子 (γ)", 0.8, 0.999, 0.9985, 0.0001)
        batch_size = st.number_input("批次大小", min_value=16, max_value=512, value=128)
        memory_size = st.number_input("记忆容量", min_value=1000, max_value=200000,
                                      value=61600, step=1000)

    with col2:
        target_update = st.number_input("目标网络更新间隔", min_value=10, max_value=5000,
                                        value=500, step=10)
        eps_start = st.slider("探索率起始值", 0.1, 1.0, 0.957, 0.001)
        eps_end = st.slider("探索率结束值", 0.01, 0.3, 0.0723, 0.001)
        eps_decay = st.slider("探索率衰减", 0.9, 0.9999, 0.995, 0.0001)

    return DQNConfig(
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        memory_size=memory_size,
        target_update=target_update,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay
    )


def configure_pdqn_params():
    """配置PDQN参数"""
    col1, col2 = st.columns(2)

    with col1:
        lr = st.number_input("学习率", min_value=1e-5, max_value=1e-1,
                             value=0.0005, format="%.5f")
        gamma = st.slider("折扣因子 (γ)", 0.8, 0.999, 0.99, 0.001)
        batch_size = st.number_input("批次大小", min_value=16, max_value=512, value=32)
        memory_size = st.number_input("记忆容量", min_value=1000, max_value=200000,
                                      value=50000, step=1000)

    with col2:
        alpha = st.slider("优先级强度 (α)", 0.0, 1.0, 0.6, 0.05)
        beta = st.slider("重要性采样 (β)", 0.0, 1.0, 0.4, 0.05)
        beta_increment = st.number_input("β增量", min_value=0.0001, max_value=0.01,
                                         value=0.001, format="%.4f")

    return PDQNConfig(
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        memory_size=memory_size,
        alpha=alpha,
        beta=beta,
        beta_increment=beta_increment
    )


def configure_ppo_params():
    """配置PPO参数"""
    col1, col2, col3 = st.columns(3)

    with col1:
        learning_rate = st.number_input("学习率", min_value=1e-5, max_value=1e-1,
                                        value=0.00015, format="%.5f")
        gamma = st.slider("折扣因子 (γ)", 0.8, 0.999, 0.99, 0.001)
        value_coef = st.number_input("价值系数", min_value=0.1, max_value=2.0,
                                     value=0.54, step=0.1)

    with col2:
        entropy_coef = st.number_input("熵系数", min_value=1e-5, max_value=0.1,
                                       value=0.002, format="%.5f")
        lambda_gae = st.slider("GAE λ", 0.8, 1.0, 0.95, 0.01)
        clip_eps = st.slider("Clip参数 (ε)", 0.1, 0.4, 0.2, 0.05)

    with col3:
        memory_size = st.number_input("记忆容量", min_value=256, max_value=10000,
                                      value=1024, step=256)
        minibatch_size = st.number_input("小批次大小", min_value=32, max_value=512,
                                         value=64, step=32)
        epoch = st.number_input("训练轮数", min_value=1, max_value=100, value=16)

    return PPOConfig(
        learning_rate=learning_rate,
        gamma=gamma,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        lambda_gae=lambda_gae,
        clip_eps=clip_eps,
        memory_size=memory_size,
        minibatch_size=minibatch_size,
        epoch=epoch
    )


def configure_ac_params():
    """配置AC参数"""
    col1, col2 = st.columns(2)

    with col1:
        lr = st.number_input("学习率", min_value=1e-5, max_value=1e-1,
                             value=0.001, format="%.5f")
        gamma = st.slider("折扣因子 (γ)", 0.8, 0.999, 0.9, 0.001)
        batch_size = st.number_input("批次大小", min_value=16, max_value=512, value=32)

    with col2:
        memory_size = st.number_input("记忆容量", min_value=1000, max_value=100000,
                                      value=5000, step=1000)
        value_coef = st.number_input("价值系数", min_value=0.1, max_value=1.0,
                                     value=0.5, step=0.1)
        entropy_coef = st.number_input("熵系数", min_value=1e-5, max_value=0.1,
                                       value=0.001, format="%.5f")

    return ACConfig(
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        memory_size=memory_size,
        value_coef=value_coef,
        entropy_coef=entropy_coef
    )


def configure_physics_params():
    """配置Physics参数"""
    col1, col2 = st.columns(2)

    with col1:
        theta_coef = st.slider("角度系数", 0.0, 2.0, 1.0, 0.1)
        omega_coef = st.slider("角速度系数", 0.0, 2.0, 1.0, 0.1)

    with col2:
        pos_coef = st.slider("位置系数", 0.0, 1.0, 0.1, 0.05)
        vel_coef = st.slider("速度系数", 0.0, 1.0, 0.1, 0.05)

    return PhysicsConfig(
        theta_coef=theta_coef,
        omega_coef=omega_coef,
        pos_coef=pos_coef,
        vel_coef=vel_coef
    )


def start_training(algorithm, episodes, config, render, ui):
    """开始训练"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"cartpole_{algorithm}_{timestamp}"

    # 创建进度显示
    progress_bar = st.progress(0)
    status_text = st.empty()
    output_container = st.empty()

    # 训练逻辑
    try:
        if algorithm == "dqn":
            status_text.text("正在训练DQN...")
            success, output, agent = ui.run_capture_output(
                train_dqn,
                num_episodes=episodes,
                terminal_penalty=True,
                save_path=f"models/{model_name}.torch",
                saved=True
            )

        elif algorithm == "pdqn":
            status_text.text("正在训练PDQN...")
            success, output, agent = ui.run_capture_output(
                train_pdqn,
                num_episodes=episodes,
                terminal_penalty=True,
                save_path=f"models/{model_name}.torch",
                saved=True,
                config_path=None
            )

        elif algorithm == "ppo":
            status_text.text("正在训练PPO...")
            success, output, agent = ui.run_capture_output(
                train_ppo,
                num_episodes=episodes,
                terminal_penalty=True,
                save_path=f"models/{model_name}.torch",
                saved=True,
                config_path=None
            )

        elif algorithm == "ac":
            status_text.text("正在训练AC...")
            # 注意：AC需要另外实现train_ac函数
            status_text.text("AC训练功能待实现...")
            success, output = False, "AC训练功能待实现"
            agent = None

        elif algorithm == "physics":
            status_text.text("正在创建Physics Agent...")
            success, output, _ = ui.run_capture_output(
                create_physics_agent, config, model_name
            )
            agent = None

        # 更新进度条
        progress_bar.progress(100)

        if success:
            st.success(f"✅ {algorithm.upper()} 训练完成！")

            # 显示训练输出
            with st.expander("查看训练日志"):
                st.code(output[:2000])

            # 如果是PPO，显示训练进度图
            if algorithm == "ppo" and agent and hasattr(agent, 'step_record'):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(agent.step_record)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Steps")
                ax.set_title("PPO Training Progress")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            # 立即评估（限制评估次数，特别是如果显示画面）
            if algorithm != "physics":
                # 如果选择了显示画面，限制评估次数
                eval_episodes = 3 if render else 10

                with st.spinner(f"正在评估训练好的模型 ({eval_episodes}次)..."):
                    try:
                        scores, avg_score = evaluate_agent(
                            model_path=f"models/{model_name}.torch",
                            algorithm=algorithm,
                            episodes=eval_episodes,
                            render=render,
                            fps=60
                        )

                        st.metric("平均得分", f"{avg_score:.2f}")

                        # 显示评估结果
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(len(scores))),
                                y=scores,
                                mode='lines+markers',
                                name='得分'
                            ))
                            fig.update_layout(
                                title="评估结果",
                                xaxis_title="回合",
                                yaxis_title="得分"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # 简单统计
                            stats = pd.DataFrame({
                                "指标": ["平均分", "最高分", "最低分"],
                                "数值": [f"{avg_score:.2f}", f"{max(scores)}", f"{min(scores)}"]
                            })
                            st.dataframe(stats, use_container_width=True)

                    except Exception as e:
                        st.warning(f"评估失败: {str(e)}")

    except Exception as e:
        st.error(f"❌ 训练过程中发生错误: {str(e)}")


def create_physics_agent(config, model_name):
    """创建Physics Agent"""
    agent = PhysicsAgent(4, 2, cfg=config)
    agent.save(f"models/{model_name}.json")
    return agent


if __name__ == "__main__":
    main()