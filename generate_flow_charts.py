import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_main_flow_chart(output_path='output/主业务流程图.png'):
    fig, ax = plt.subplots(figsize=(14, 18), dpi=150)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 180)
    ax.axis('off')

    box_width = 40
    box_height = 12
    start_y = 165
    vertical_gap = 16

    def draw_box(x, y, text, subtext=None, color='#4A90D9', text_color='white'):
        rect = FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                            boxstyle="round,pad=0.3", facecolor=color, edgecolor='#2E5C8A', linewidth=2)
        ax.add_patch(rect)
        
        if subtext:
            lines = text.split('\n')
            main_text = '\n'.join(lines[:-1]) if len(lines) > 1 else text
            sub_text = lines[-1] if len(lines) > 1 else ''
            
            ax.text(x, y + 2, main_text, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
            ax.text(x, y - 2, sub_text, ha='center', va='center', fontsize=9, color=text_color)
        else:
            ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
        
        return y - box_height/2

    def draw_diamond(x, y, text, size=15):
        diamond = patches.RegularPolygon((x, y), 4, size, orientation=np.pi/4,
                                        facecolor='#F39C12', edgecolor='#D68910', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        return y - size

    def draw_arrow(x_start, y_start, x_end, y_end, label=None):
        arrow = patches.FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                        arrowstyle='->', mutation_scale=15,
                                        linewidth=2, color='#34495E')
        ax.add_patch(arrow)
        if label:
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2
            ax.text(mid_x + 3, mid_y, label, fontsize=8, color='#7F8C8D')

    ax.text(70, 175, '海运运价预测系统 - 主业务流程图', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='#2E5C8A')

    y = start_y
    ax.text(70, y, '开始', ha='center', va='center', fontsize=12, fontweight='bold', color='#2E5C8A')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    left_box_y = draw_box(35, y, '读取SCFI数据', None, '#3498DB')
    right_box_y = draw_box(105, y, '读取韩国出口指数', None, '#3498DB')
    y -= vertical_gap + 8
    
    draw_arrow(35, left_box_y, 70, y + 8)
    draw_arrow(105, right_box_y, 70, y + 8)

    y = draw_box(70, y, '数据融合合并\n按日期对齐', None, '#27AE60')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    y = draw_box(70, y, '数据清洗\n缺失值处理', None, '#27AE60')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    y = draw_box(70, y, '滞后特征构造\nlag1~lag4', None, '#27AE60')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    y = draw_box(70, y, '数据集划分\n训练集80%\n验证集20%', None, '#27AE60')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    y = draw_box(70, y, '模型训练\n随机森林', None, '#9B59B6')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    y = draw_box(70, y, '模型预测\n验证集预测', None, '#9B59B6')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    y = draw_box(70, y, '结果分析\nR²/MAE/RMSE\n特征重要性', None, '#E74C3C')
    y -= vertical_gap
    draw_arrow(70, y + 8, 70, y - 8)

    y = draw_box(70, y, '结果导出\n图表+报表+日志', None, '#E74C3C')
    y -= vertical_gap + 8
    draw_arrow(70, y + 8, 70, y - 8)

    ax.text(70, y - 8, '结束', ha='center', va='center', fontsize=12, fontweight='bold', color='#2E5C8A')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] 主业务流程图已保存至: {output_path}')

def draw_data_flow_chart(output_path='output/数据流图.png'):
    fig, ax = plt.subplots(figsize=(16, 14), dpi=150)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 140)
    ax.axis('off')

    box_width = 50
    box_height = 18
    start_y = 120

    def draw_process_box(x, y, title, subtitle, color='#3498DB'):
        rect = FancyBboxPatch((x - box_width/2, y - box_height/2), box_width, box_height,
                            boxstyle="round,pad=0.3", facecolor=color, edgecolor='#2E5C8A', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 3, title, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(x, y - 3, subtitle, ha='center', va='center', fontsize=9, color='white')
        return y - box_height/2

    def draw_data_box(x, y, text, color='#F39C12'):
        rect = patches.Rectangle((x - 30, y - 10), 60, 20, 
                                facecolor=color, edgecolor='#D68910', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        return y - 10

    def draw_output_box(x, y, text, color='#E74C3C'):
        rect = patches.Rectangle((x - 20, y - 8), 40, 16, 
                                facecolor=color, edgecolor='#C0392B', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    def draw_arrow(x_start, y_start, x_end, y_end, label=None):
        arrow = patches.FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                        arrowstyle='->', mutation_scale=15,
                                        linewidth=2, color='#34495E')
        ax.add_patch(arrow)
        if label:
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2
            ax.text(mid_x, mid_y - 5, label, ha='center', fontsize=8, color='#7F8C8D')

    ax.text(80, 130, '海运运价预测系统 - 数据流图', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='#2E5C8A')

    y = start_y
    left_y = draw_data_box(35, y, 'SCFI Excel文件')
    right_y = draw_data_box(125, y, '韩国出口指数Excel')
    y -= 25

    draw_arrow(35, left_y, 80, y + box_height/2)
    draw_arrow(125, right_y, 80, y + box_height/2)

    y = draw_process_box(80, y, 'DataReader', '(双数据源读取，格式转换)', '#3498DB')
    y -= 25

    draw_arrow(80, y + 8, 80, y - 8)
    ax.text(80, y - 2, 'SCFI DataFrame + 韩国出口指数 DataFrame', ha='center', fontsize=9, color='#34495E')

    y -= 15
    y = draw_process_box(80, y, 'DataFusionPreprocessor', '(日期对齐、滞后特征构造)', '#27AE60')
    y -= 25

    draw_arrow(80, y + 8, 80, y - 8)
    ax.text(80, y - 2, '融合后的特征矩阵X + 目标Y', ha='center', fontsize=9, color='#34495E')

    y -= 15
    left_split_y = y
    right_split_y = y
    
    draw_arrow(80, y + 8, 45, y + 8)
    draw_arrow(80, y + 8, 115, y + 8)
    
    draw_data_box(45, y, '训练集 80%', '#F39C12')
    draw_data_box(115, y, '验证集 20%', '#F39C12')
    
    y -= 25
    draw_arrow(45, y + 18, 80, y + box_height/2)
    draw_arrow(115, y + 18, 80, y + box_height/2)

    y = draw_process_box(80, y, 'SCFIForecastModel', '(模型训练、保存加载、预测)', '#9B59B6')
    y -= 25

    draw_arrow(80, y + 8, 80, y - 8)
    ax.text(80, y - 2, '训练完成的模型 + 预测结果', ha='center', fontsize=9, color='#34495E')

    y -= 15
    y = draw_process_box(80, y, 'ResultAnalyzer', '(精度评估、相关性分析)', '#E67E22')
    y -= 25

    draw_arrow(80, y + 8, 80, y - 8)
    ax.text(80, y - 2, '评估指标 + 特征重要性', ha='center', fontsize=9, color='#34495E')

    y -= 15
    y = draw_process_box(80, y, 'ResultExporter', '(可视化图表、报表导出)', '#E74C3C')
    
    y -= 20
    draw_arrow(80, y + 8, 40, y - 8)
    draw_arrow(80, y + 8, 80, y - 8)
    draw_arrow(80, y + 8, 120, y - 8)

    y -= 15
    draw_output_box(40, y, 'PNG图表')
    draw_output_box(80, y, 'Excel报表')
    draw_output_box(120, y, 'TXT日志')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] 数据流图已保存至: {output_path}')

if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)
    
    draw_main_flow_chart()
    draw_data_flow_chart()
    
    print('\n所有流程图已生成完毕！')
    print('输出文件:')
    print('  - output/主业务流程图.png')
    print('  - output/数据流图.png')