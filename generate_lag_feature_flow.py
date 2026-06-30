import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_lag_feature_flow(output_path='output/滞后特征构造算法流程图.png'):
    fig, ax = plt.subplots(figsize=(18, 22), dpi=150)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 220)
    ax.axis('off')

    box_width = 60
    box_height = 14
    small_box_width = 45
    small_box_height = 10
    start_y = 200

    def draw_box(x, y, text, color='#3498DB', text_color='white', width=box_width, height=box_height):
        rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.3", facecolor=color, edgecolor='#2E5C8A', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
        return y - height/2

    def draw_parallelogram(x, y, text, color='#F39C12'):
        vertices = np.array([
            [x - 30, y - 12],
            [x - 10, y + 12],
            [x + 30, y + 12],
            [x + 10, y - 12]
        ])
        para = patches.Polygon(vertices, facecolor=color, edgecolor='#D68910', linewidth=2)
        ax.add_patch(para)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        return y - 12

    def draw_diamond(x, y, text, size=12):
        diamond = patches.RegularPolygon((x, y), 4, size, orientation=np.pi/4,
                                        facecolor='#9B59B6', edgecolor='#8E44AD', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        return y - size

    def draw_loop_box(x, y, text, color='#27AE60'):
        rect = patches.Rectangle((x - 55, y - 35), 110, 70,
                                facecolor=color, edgecolor='#1E8449', linewidth=2, fill=False, linestyle='--')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#1E8449')
        return y - 35

    def draw_arrow(x_start, y_start, x_end, y_end, label=None, direction='down'):
        if direction == 'down':
            arrow = patches.FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                            arrowstyle='->', mutation_scale=15,
                                            linewidth=2, color='#34495E')
            ax.add_patch(arrow)
            if label:
                ax.text(x_start + 3, (y_start + y_end) / 2, label, fontsize=8, color='#7F8C8D')
        elif direction == 'right':
            arrow = patches.FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                            arrowstyle='->', mutation_scale=15,
                                            linewidth=2, color='#34495E')
            ax.add_patch(arrow)
            if label:
                ax.text((x_start + x_end) / 2, y_start + 3, label, fontsize=8, color='#7F8C8D')

    def draw_text(x, y, text, fontsize=10, color='#34495E'):
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=color)

    ax.text(90, 210, '滞后特征构造算法流程图', ha='center', va='center', 
            fontsize=20, fontweight='bold', color='#2E5C8A')

    y = start_y
    y = draw_parallelogram(90, y, '输入：融合清洗后DataFrame\nlag_weeks=4')
    y -= 22
    draw_arrow(90, y + 8, 90, y - 8)

    y = draw_box(90, y, '步骤1：筛选全部数值特征\n(SCFI航线、韩国出口各指标)', '#3498DB')
    y -= 22
    draw_arrow(90, y + 8, 90, y - 8)

    draw_text(90, y - 2, 'SCFI航线: 综合指数、欧洲航线、美西航线...', fontsize=9)
    draw_text(90, y - 8, '韩国出口: 综合景气指数、对中国出口指数、先行指数...', fontsize=9)
    y -= 15

    y = draw_box(90, y, '步骤2：循环生成滞后特征', '#27AE60')
    y -= 8

    loop_start_y = y
    draw_loop_box(90, y, 'FOR EACH 特征 IN 数值特征:')
    
    y -= 15
    draw_text(90, y, 'FOR lag = 1 TO 4:', fontsize=10, color='#1E8449')
    
    y -= 12
    draw_box(90, y, 'df[特征_lagN] = df[特征].shift(N)', '#5DADE2', width=small_box_width, height=small_box_height)
    
    y -= 15
    draw_text(90, y, '生成列名示例:', fontsize=9, color='#7F8C8D')
    draw_text(90, y - 5, '综合指数_lag1, 综合指数_lag2, 综合指数_lag3, 综合指数_lag4', fontsize=8)
    draw_text(90, y - 10, '韩国出口先行指数_lag1, 韩国出口先行指数_lag2...', fontsize=8)
    
    y -= 25
    
    draw_arrow(90, y + 8, 90, y + 80, direction='right')
    draw_arrow(90, y + 80, 90, y + 88, direction='down')

    y = loop_start_y - 75
    y -= 20
    draw_arrow(90, y + 8, 90, y - 8)

    y = draw_box(90, y, '步骤3：构造预测标签\nscfi_next = 综合指数.shift(-1)\n(下周运价)', '#9B59B6')
    y -= 22
    draw_arrow(90, y + 8, 90, y - 8)

    y = draw_box(90, y, '步骤4：删除边界空行', '#E67E22')
    y -= 22
    draw_arrow(90, y + 8, 90, y - 8)

    y = draw_parallelogram(90, y, '输出：带全部滞后特征数据集')
    y -= 15

    draw_text(90, y - 5, '数据行数: N - lag_weeks', fontsize=9, color='#7F8C8D')
    draw_text(90, y - 10, '特征列数: 原始列数 × (1 + lag_weeks)', fontsize=9, color='#7F8C8D')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] 滞后特征构造算法流程图已保存至: {output_path}')

if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)
    
    draw_lag_feature_flow()
    
    print('\n流程图已生成完毕！')
    print('输出文件:')
    print('  - output/滞后特征构造算法流程图.png')