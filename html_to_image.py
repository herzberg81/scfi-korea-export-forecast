"""
将HTML流程图转换为PNG图片
"""
import os
import subprocess

def html_to_image_simple():
    """
    使用matplotlib重新绘制流程图（因为HTML转PNG需要额外依赖）
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch, Polygon
    import numpy as np
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(14, 20), dpi=150)
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 200)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    
    # 标题
    ax.text(70, 190, '滞后特征构造算法流程图', 
            ha='center', va='center', fontsize=18, fontweight='bold', color='#2E5C8A')
    
    def draw_parallelogram(x, y, width, height, text_lines, color='#F39C12'):
        """绘制平行四边形（输入/输出框）"""
        offset = 15
        vertices = [
            (x - width/2 + offset, y - height/2),
            (x - width/2, y + height/2),
            (x + width/2, y + height/2),
            (x + width/2 - offset, y - height/2)
        ]
        para = Polygon(vertices, facecolor=color, edgecolor='#D68910', linewidth=2)
        ax.add_patch(para)
        
        line_height = height / (len(text_lines) + 1)
        for i, line in enumerate(text_lines):
            ax.text(x, y - height/2 + line_height * (i + 1), line,
                   ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        return y - height/2
    
    def draw_rounded_box(x, y, width, height, text_lines, color='#3498DB', step_num=None):
        """绘制圆角矩形（处理步骤框）"""
        rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.1", facecolor=color, 
                             edgecolor='#2E5C8A', linewidth=2)
        ax.add_patch(rect)
        
        # 步骤编号
        if step_num:
            circle = plt.Circle((x - width/2 - 15, y), 10, color='#34495E', zorder=10)
            ax.add_patch(circle)
            ax.text(x - width/2 - 15, y, str(step_num), ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white', zorder=11)
        
        line_height = height / (len(text_lines) + 1)
        for i, line in enumerate(text_lines):
            ax.text(x, y - height/2 + line_height * (i + 1), line,
                   ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        return y - height/2
    
    def draw_loop_box(x, y, width, height, color='#27AE60'):
        """绘制虚线边框（循环结构）"""
        rect = patches.Rectangle((x - width/2, y - height/2), width, height,
                                 facecolor='#E8F6F3', edgecolor=color, 
                                 linewidth=2, linestyle='--')
        ax.add_patch(rect)
        return y - height/2
    
    def draw_arrow_down(x, y_start, y_end):
        """绘制向下箭头"""
        ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                   arrowprops=dict(arrowstyle='->', color='#34495E', lw=2))
    
    box_width = 80
    box_height = 18
    y = 175
    
    # 输入
    y = draw_parallelogram(70, y, box_width, 25, 
                          ['输入', '融合清洗后DataFrame', 'lag_weeks = 4'], '#F39C12')
    y -= 15
    draw_arrow_down(70, y + 8, y - 8)
    
    # 步骤1
    y -= 5
    y = draw_rounded_box(70, y, box_width, 22, 
                        ['筛选全部数值特征', 'SCFI航线、韩国出口各指标'], '#3498DB', 1)
    
    # 特征列表说明
    ax.text(70, y - 5, 'SCFI航线: 综合指数、欧洲航线、美西航线...', 
           ha='center', fontsize=9, color='#7F8C8D')
    ax.text(70, y - 10, '韩国出口: 综合景气指数、对中国出口指数、先行指数...', 
           ha='center', fontsize=9, color='#7F8C8D')
    y -= 20
    draw_arrow_down(70, y + 8, y - 8)
    
    # 步骤2
    y -= 5
    y = draw_rounded_box(70, y, box_width, 18, ['循环生成滞后特征'], '#27AE60', 2)
    y -= 15
    
    # 循环框
    loop_y = draw_loop_box(70, y, box_width + 20, 50, '#27AE60')
    ax.text(70, y + 15, 'FOR EACH 特征 IN 数值特征:', 
           ha='center', fontsize=10, fontweight='bold', color='#1E8449')
    ax.text(70, y + 5, 'FOR lag = 1 TO 4:', 
           ha='center', fontsize=10, color='#27AE60')
    
    # 代码行
    code_y = y - 10
    code_rect = patches.Rectangle((70 - 50, code_y - 8), 100, 16,
                                  facecolor='#2C3E50', edgecolor='#1E8449', linewidth=1)
    ax.add_patch(code_rect)
    ax.text(70, code_y, 'df[特征_lagN] = df[特征].shift(N)', 
           ha='center', va='center', fontsize=10, color='#ECF0F1')
    
    # 列名示例
    ax.text(70, y - 25, '生成列名: 综合指数_lag1~lag4, 韩国出口先行指数_lag1~lag4', 
           ha='center', fontsize=8, color='#7F8C8D')
    
    y = loop_y - 5
    draw_arrow_down(70, y + 8, y - 8)
    
    # 步骤3
    y -= 5
    y = draw_rounded_box(70, y, box_width, 28, 
                        ['构造预测标签', 'scfi_next = 综合指数.shift(-1)', '(下周运价)'], '#9B59B6', 3)
    y -= 15
    draw_arrow_down(70, y + 8, y - 8)
    
    # 步骤4
    y -= 5
    y = draw_rounded_box(70, y, box_width, 22, 
                        ['删除边界空行', '移除shift产生的缺失值'], '#E67E22', 4)
    y -= 15
    draw_arrow_down(70, y + 8, y - 8)
    
    # 输出
    y -= 5
    y = draw_parallelogram(70, y, box_width, 25, 
                          ['输出', '带全部滞后特征数据集'], '#F39C12')
    
    # 输出信息
    ax.text(70, y - 5, '数据行数: N - lag_weeks', 
           ha='center', fontsize=9, color='#7F8C8D')
    ax.text(70, y - 10, '特征列数: 原始列数 × (1 + lag_weeks)', 
           ha='center', fontsize=9, color='#7F8C8D')
    
    # 底部说明
    ax.text(70, 10, '海运运价预测系统 · 核心算法流程图', 
           ha='center', fontsize=12, color='#7F8C8D')
    
    plt.tight_layout()
    output_path = 'output/滞后特征构造算法流程图_HTML版.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    
    print(f'[OK] 流程图已保存至: {output_path}')
    return output_path

if __name__ == '__main__':
    os.makedirs('output', exist_ok=True)
    result = html_to_image_simple()
    print(f'\n生成完成: {result}')