import re
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def parse_single_log_loss(log_content, model_name):
    """
    从单个日志内容中解析loss数据
    """
    epochs = []
    losses = []
    train_accuracies = []
    
    # 正则表达式匹配
    epoch_pattern = r'Epoch\(train\)\s+\[(\d+)\]'
    loss_pattern = r'loss: (\d+\.\d+)'
    acc_pattern = r'top1_acc: (\d+\.\d+)'
    
    lines = log_content.split('\n')
    current_epoch = 0
    
    for line in lines:
        # 匹配epoch
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
            
        # 匹配loss
        loss_match = re.search(loss_pattern, line)
        if loss_match and 'loss_cls' not in line and 'Epoch(train)' in line:
            loss_value = float(loss_match.group(1))
            epochs.append(current_epoch)
            losses.append(loss_value)
        
        # 匹配训练准确率（可选）
        acc_match = re.search(acc_pattern, line)
        if acc_match and 'Epoch(train)' in line:
            acc_value = float(acc_match.group(1))
            train_accuracies.append(acc_value)
    
    return {
        'epochs': epochs, 
        'losses': losses, 
        'train_accuracies': train_accuracies,
        'model': model_name
    }

def plot_single_model_analysis(data, output_dir, model_name):
    """
    绘制单个模型的详细分析图表
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Loss曲线
    ax1.plot(data['epochs'], data['losses'], 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_name} - Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 如果loss变化范围大，使用对数尺度
    if max(data['losses']) > 10 * min(data['losses']):
        ax1.set_yscale('log')
        ax1.set_ylabel('Training Loss (log scale)', fontsize=12, fontweight='bold')
    
    # 2. Loss下降百分比（相对第一个epoch）
    initial_loss = data['losses'][0]
    loss_reduction = [(initial_loss - loss) / initial_loss * 100 for loss in data['losses']]
    ax2.plot(data['epochs'], loss_reduction, 'r-', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss Reduction (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_name} - Loss Reduction Percentage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练准确率（如果有数据）
    if data['train_accuracies'] and len(data['train_accuracies']) == len(data['epochs']):
        ax3.plot(data['epochs'], data['train_accuracies'], 'g-', linewidth=2, marker='^', markersize=3)
        ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Training Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title(f'{model_name} - Training Accuracy', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)  # 准确率范围0-1.1
    
    # 4. 移动平均loss（平滑曲线）
    window_size = min(5, len(data['losses']) // 10)  # 动态窗口大小
    if window_size > 1:
        moving_avg = pd.Series(data['losses']).rolling(window=window_size).mean()
        ax4.plot(data['epochs'], data['losses'], 'b-', alpha=0.5, label='Raw Loss')
        ax4.plot(data['epochs'], moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
        ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax4.set_title(f'{model_name} - Smoothed Loss', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = output_path / f"{model_name}_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_loss_report(data, output_dir, model_name):
    """
    生成详细的loss分析报告
    """
    output_path = Path(output_dir)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'Epoch': data['epochs'],
        'Loss': data['losses']
    })
    
    if data['train_accuracies'] and len(data['train_accuracies']) == len(data['epochs']):
        df['Training_Accuracy'] = data['train_accuracies']
    
    # 计算统计信息
    initial_loss = data['losses'][0]
    final_loss = data['losses'][-1]
    min_loss = min(data['losses'])
    max_loss = max(data['losses'])
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    # 生成报告文本
    report_text = f"""
=== {model_name} Training Loss Analysis Report ===

Basic Statistics:
- Total Epochs: {len(data['epochs'])}
- Initial Loss: {initial_loss:.6f}
- Final Loss: {final_loss:.6f}
- Minimum Loss: {min_loss:.6f} (Epoch {data['epochs'][data['losses'].index(min_loss)]})
- Maximum Loss: {max_loss:.6f}
- Loss Reduction: {loss_reduction:.2f}%

Convergence Analysis:
- Final loss is {final_loss/min_loss:.2f}x the minimum loss
- Loss reduced by {loss_reduction:.1f}% from initial value

Training Progress:
"""
    
    # 添加关键epoch的信息
    key_epochs = [1, len(data['epochs'])//4, len(data['epochs'])//2, 
                 3*len(data['epochs'])//4, len(data['epochs'])]
    
    for epoch in sorted(set(key_epochs)):
        if epoch <= len(data['epochs']):
            idx = epoch - 1
            report_text += f"- Epoch {epoch}: Loss = {data['losses'][idx]:.6f}\n"
    
    # 保存报告
    report_file = output_path / f"{model_name}_loss_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 保存CSV数据
    csv_file = output_path / f"{model_name}_loss_data.csv"
    df.to_csv(csv_file, index=False)
    
    return report_file, csv_file

def analyze_single_log(input_file, output_dir, model_name=None):
    """
    主函数：分析单个日志文件
    """
    # 如果没有指定模型名，使用文件名
    if model_name is None:
        model_name = Path(input_file).stem
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 读取日志文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    print(f"开始分析 {model_name} 的日志文件...")
    
    # 解析数据
    data = parse_single_log_loss(log_content, model_name)
    
    if not data['epochs']:
        print("警告: 没有解析到有效的训练数据")
        return
    
    print(f"解析成功: {len(data['epochs'])} 个epoch的数据")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成图表
    chart_file = plot_single_model_analysis(data, output_dir, model_name)
    print(f"分析图表已保存: {chart_file}")
    
    # 生成报告
    report_file, csv_file = generate_loss_report(data, output_dir, model_name)
    print(f"分析报告已保存: {report_file}")
    print(f"详细数据已保存: {csv_file}")
    
    # 显示关键信息
    print(f"\n{model_name} 训练摘要:")
    print(f"  - 总epoch数: {len(data['epochs'])}")
    print(f"  - 初始loss: {data['losses'][0]:.6f}")
    print(f"  - 最终loss: {data['losses'][-1]:.6f}")
    print(f"  - loss下降: {(data['losses'][0] - data['losses'][-1]) / data['losses'][0] * 100:.1f}%")

# 使用示例
if __name__ == "__main__":
    # 在这里指定您的输入文件和输出目录
    input_file = "path/to/your/training_log.txt"  # 替换为您的日志文件路径
    output_dir = "analysis_results"  # 输出目录
    model_name = "I3D"  # 可选：指定模型名称
    
    # 运行分析
    analyze_single_log(input_file, output_dir, model_name)