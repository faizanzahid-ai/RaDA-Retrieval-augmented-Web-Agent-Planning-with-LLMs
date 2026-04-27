"""
Comprehensive plotting of all evaluation metrics and tables
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load evaluation results
results_file = Path("evaluation_results/mind2web_mind2web_evaluation_20260427_203248.json")
with open(results_file, 'r') as f:
    data = json.load(f)

aggregate = data['aggregate']

# Create figure with subplots
fig = plt.figure(figsize=(20, 14))
fig.suptitle('RaDA Evaluation Results - Comprehensive Report', fontsize=18, fontweight='bold', y=0.995)

# Color scheme
colors_primary = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
colors_success = ['#2ecc71']
colors_fail = ['#e74c3c']

# 1. Primary Metrics (Horizontal Bar Chart)
ax1 = plt.subplot(3, 4, 1)
metrics = ['Task Success Rate', 'Step Success Rate', 'Action F1', 'Element Accuracy', 'Operation F1']
values = [
    aggregate['task_success_rate'] * 100,
    aggregate['step_success_rate'] * 100,
    aggregate['action_f1'] * 100,
    aggregate['element_accuracy'] * 100,
    aggregate['operation_f1'] * 100
]
colors = ['#2ecc71' if v > 50 else '#e74c3c' if v > 0 else '#f39c12' for v in values]
bars = ax1.barh(metrics, values, color=colors)
ax1.set_xlabel('Percentage (%)')
ax1.set_title('Primary Metrics (Paper-Level)', fontweight='bold', fontsize=11)
ax1.set_xlim(0, 110)
for bar, val in zip(bars, values):
    ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
             va='center', fontweight='bold')

# 2. Task Execution Time (Vertical Bar)
ax2 = plt.subplot(3, 4, 2)
task_ids = [t['task_id'] for t in aggregate['task_metrics']]
exec_times = [t['execution_time'] for t in aggregate['task_metrics']]
colors_time = ['#3498db', '#e67e22']
bars = ax2.bar(task_ids, exec_times, color=colors_time, alpha=0.8)
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Task Execution Time', fontweight='bold', fontsize=11)
for bar, val in zip(bars, exec_times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}s', 
             ha='center', va='bottom', fontweight='bold')

# 3. Actions per Task
ax3 = plt.subplot(3, 4, 3)
actions_count = [t['total_actions'] for t in aggregate['task_metrics']]
bars = ax3.bar(task_ids, actions_count, color=colors_time, alpha=0.8)
ax3.set_ylabel('Number of Actions')
ax3.set_title('Actions per Task', fontweight='bold', fontsize=11)
for bar, val in zip(bars, actions_count):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val}', 
             ha='center', va='bottom', fontweight='bold')

# 4. Subtasks Completion (Grouped Bar)
ax4 = plt.subplot(3, 4, 4)
subtasks_completed = [t['subtasks_completed'] for t in aggregate['task_metrics']]
subtasks_total = [t['subtasks_total'] for t in aggregate['task_metrics']]
x = np.arange(len(task_ids))
width = 0.35
rects1 = ax4.bar(x - width/2, subtasks_completed, width, label='Completed', color='#2ecc71', alpha=0.8)
rects2 = ax4.bar(x + width/2, subtasks_total, width, label='Total', color='#95a5a6', alpha=0.8)
ax4.set_ylabel('Count')
ax4.set_title('Subtasks Completion', fontweight='bold', fontsize=11)
ax4.set_xticks(x)
ax4.set_xticklabels(task_ids)
ax4.legend(loc='upper right')
for rect in rects1:
    height = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2, height, f'{int(height)}', 
             ha='center', va='bottom', fontweight='bold')
for rect in rects2:
    height = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2, height, f'{int(height)}', 
             ha='center', va='bottom', fontweight='bold')

# 5. Task Success Distribution (Pie)
ax5 = plt.subplot(3, 4, 5)
successful = sum(1 for t in aggregate['task_metrics'] if t['task_success'])
failed = len(aggregate['task_metrics']) - successful
sizes = [successful, failed]
labels = [f'Successful ({successful})', f'Failed ({failed})']
colors_pie = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                                     startangle=90, textprops={'fontweight': 'bold'})
ax5.set_title('Task Success Distribution', fontweight='bold', fontsize=11)

# 6. Step Success Rate per Task (Bar)
ax6 = plt.subplot(3, 4, 6)
step_success = [t['step_success_rate'] * 100 for t in aggregate['task_metrics']]
bars = ax6.bar(task_ids, step_success, color=colors_time, alpha=0.8)
ax6.set_ylabel('Step Success Rate (%)')
ax6.set_title('Step Success Rate per Task', fontweight='bold', fontsize=11)
ax6.set_ylim(0, 110)
for bar, val in zip(bars, step_success):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.0f}%', 
             ha='center', va='bottom', fontweight='bold')

# 7. Performance by Difficulty (Table as plot)
ax7 = plt.subplot(3, 4, 7)
ax7.axis('off')
difficulty_data = [
    ['Difficulty', 'Tasks', 'Success Rate', 'Avg Time (s)'],
    ['Easy', '2', '100.00%', '58.57']
]
table = ax7.table(cellText=difficulty_data, cellLoc='center', loc='center',
                 colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)
for i in range(len(difficulty_data)):
    for j in range(len(difficulty_data[0])):
        if i == 0:
            table[(i, j)].set_facecolor('#34495e')
            table[(i, j)].set_text_props(color='white', weight='bold')
        else:
            table[(i, j)].set_facecolor('#ecf0f1')
ax7.set_title('Performance by Difficulty', fontweight='bold', fontsize=11, pad=20)

# 8. Overall Performance Summary (Table)
ax8 = plt.subplot(3, 4, 8)
ax8.axis('off')
summary_data = [
    ['Metric', 'Value'],
    ['Total Tasks', f"{aggregate['total_tasks']}"],
    ['Successful Tasks', f"{sum(1 for t in aggregate['task_metrics'] if t['task_success'])}"],
    ['Task Success Rate', f"{aggregate['task_success_rate']*100:.1f}%"],
    ['Step Success Rate', f"{aggregate['step_success_rate']*100:.1f}%"],
    ['Avg Actions/Task', f"{aggregate['avg_actions_per_task']:.1f}"],
    ['Avg Execution Time', f"{aggregate['avg_execution_time']:.1f}s"],
    ['Avg Subtasks Completed', f"{aggregate['avg_subtasks_completed']:.1f}"]
]
table = ax8.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(summary_data)):
    for j in range(len(summary_data[0])):
        if i == 0:
            table[(i, j)].set_facecolor('#34495e')
            table[(i, j)].set_text_props(color='white', weight='bold')
        else:
            table[(i, j)].set_facecolor('#ecf0f1')
ax8.set_title('Overall Performance Summary', fontweight='bold', fontsize=11, pad=20)

# 9. Per-Task Breakdown (Table)
ax9 = plt.subplot(3, 4, 9)
ax9.axis('off')
task_data = [
    ['Task ID', 'Success', 'Actions', 'Time (s)']
]
for t in aggregate['task_metrics']:
    task_data.append([
        t['task_id'],
        str(t['task_success']),
        str(t['total_actions']),
        f"{t['execution_time']:.1f}"
    ])
table = ax9.table(cellText=task_data, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.2, 0.2, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(task_data)):
    for j in range(len(task_data[0])):
        if i == 0:
            table[(i, j)].set_facecolor('#34495e')
            table[(i, j)].set_text_props(color='white', weight='bold')
        else:
            table[(i, j)].set_facecolor('#ecf0f1')
ax9.set_title('Per-Task Breakdown', fontweight='bold', fontsize=11, pad=20)

# 10. Radar Chart for Metrics
ax10 = plt.subplot(3, 4, 10, projection='polar')
categories = ['Task Success', 'Step Success', 'Action F1', 'Element Acc', 'Operation F1']
values_radar = [
    aggregate['task_success_rate'],
    aggregate['step_success_rate'],
    aggregate['action_f1'],
    aggregate['element_accuracy'],
    aggregate['operation_f1']
]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
values_radar += values_radar[:1]
angles += angles[:1]
ax10.plot(angles, values_radar, 'o-', linewidth=2, color='#3498db')
ax10.fill(angles, values_radar, alpha=0.25, color='#3498db')
ax10.set_xticks(angles[:-1])
ax10.set_xticklabels(categories, size=9)
ax10.set_ylim(0, 1)
ax10.set_title('Metrics Radar Chart', fontweight='bold', fontsize=11, pad=20)
ax10.grid(True)

# 11. Execution Time vs Actions (Scatter)
ax11 = plt.subplot(3, 4, 11)
actions = [t['total_actions'] for t in aggregate['task_metrics']]
times = [t['execution_time'] for t in aggregate['task_metrics']]
colors_scatter = ['#2ecc71' if t['task_success'] else '#e74c3c' for t in aggregate['task_metrics']]
ax11.scatter(actions, times, s=200, c=colors_scatter, alpha=0.7, edgecolors='black', linewidth=2)
for i, tid in enumerate(task_ids):
    ax11.annotate(tid, (actions[i], times[i]), xytext=(5, 5), textcoords='offset points',
                 fontsize=9, fontweight='bold')
ax11.set_xlabel('Number of Actions')
ax11.set_ylabel('Execution Time (s)')
ax11.set_title('Actions vs Execution Time', fontweight='bold', fontsize=11)
ax11.grid(True, alpha=0.3)

# 12. Performance Gauge (Task Success Rate)
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
success_rate = aggregate['task_success_rate'] * 100
theta = np.linspace(0, np.pi, 100)
r = 0.8
x = r * np.cos(theta)
y = r * np.sin(theta)
ax12.plot(x, y, color='#ecf0f1', linewidth=3)
ax12.fill_between(x, y, alpha=0.1, color='#ecf0f1')
theta_fill = np.linspace(0, np.pi * (success_rate / 100), 100)
x_fill = r * np.cos(theta_fill)
y_fill = r * np.sin(theta_fill)
ax12.plot(x_fill, y_fill, color='#2ecc71', linewidth=3)
ax12.fill_between(x_fill, y_fill, alpha=0.3, color='#2ecc71')
ax12.text(0, -0.3, f'{success_rate:.1f}%', ha='center', va='center', 
         fontsize=24, fontweight='bold', color='#2ecc71')
ax12.text(0, -0.5, 'Task Success Rate', ha='center', va='center', 
         fontsize=12, fontweight='bold', color='#34495e')
ax12.set_xlim(-1, 1)
ax12.set_ylim(-0.6, 1)
ax12.set_title('Overall Performance Gauge', fontweight='bold', fontsize=11, pad=20)

plt.tight_layout()
plt.savefig('evaluation_results/comprehensive_metrics_plots.png', dpi=300, bbox_inches='tight')
print("Comprehensive plots saved to evaluation_results/comprehensive_metrics_plots.png")
plt.show()

# Also save a summary text report
with open('evaluation_results/evaluation_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("MIND2WEB EVALUATION REPORT\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Dataset: mind2web\n")
    f.write(f"Total Tasks: {aggregate['total_tasks']}\n")
    f.write(f"Timestamp: {aggregate['timestamp']}\n\n")
    
    f.write("Primary Metrics (Paper-Level)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Task Success Rate:     {aggregate['task_success_rate']*100:.2f}%\n")
    f.write(f"Step Success Rate:     {aggregate['step_success_rate']*100:.2f}%\n")
    f.write(f"Action F1:             {aggregate['action_f1']*100:.2f}%\n")
    f.write(f"Element Accuracy:      {aggregate['element_accuracy']*100:.2f}%\n")
    f.write(f"Operation F1:          {aggregate['operation_f1']*100:.2f}%\n\n")
    
    f.write("Performance by Difficulty\n")
    f.write("-" * 70 + "\n")
    f.write(f"Difficulty: Easy, Tasks: 2, Success Rate: 100.00%, Avg Time: 58.57s\n\n")
    
    f.write("Overall Performance Metrics\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Tasks:                     {aggregate['total_tasks']}\n")
    f.write(f"Successful Tasks:                {sum(1 for t in aggregate['task_metrics'] if t['task_success'])}\n")
    f.write(f"Avg Actions per Task:            {aggregate['avg_actions_per_task']:.2f}\n")
    f.write(f"Avg Execution Time (s):          {aggregate['avg_execution_time']:.2f}\n")
    f.write(f"Avg Subtasks Completed:          {aggregate['avg_subtasks_completed']:.2f}\n\n")
    
    f.write("Per-Task Breakdown\n")
    f.write("-" * 70 + "\n")
    for t in aggregate['task_metrics']:
        f.write(f"Task ID: {t['task_id']}\n")
        f.write(f"  Success: {t['task_success']}\n")
        f.write(f"  Actions: {t['total_actions']}\n")
        f.write(f"  Time: {t['execution_time']:.2f}s\n")
        f.write(f"  Subtasks: {t['subtasks_completed']}/{t['subtasks_total']}\n\n")

print("Summary report saved to evaluation_results/evaluation_summary.txt")
