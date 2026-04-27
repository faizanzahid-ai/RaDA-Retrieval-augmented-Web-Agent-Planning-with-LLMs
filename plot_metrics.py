"""
Plot evaluation metrics from RaDA evaluation results
"""

import json
import matplotlib.pyplot as plt
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
fig = plt.figure(figsize=(15, 10))
fig.suptitle('RaDA Evaluation Results - Mind2Web', fontsize=16, fontweight='bold')

# 1. Primary Metrics (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
metrics = ['Task Success Rate', 'Step Success Rate', 'Action F1', 'Element Accuracy', 'Operation F1']
values = [
    aggregate['task_success_rate'] * 100,
    aggregate['step_success_rate'] * 100,
    aggregate['action_f1'] * 100,
    aggregate['element_accuracy'] * 100,
    aggregate['operation_f1'] * 100
]
colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]
bars = ax1.bar(metrics, values, color=colors)
ax1.set_ylabel('Percentage (%)')
ax1.set_title('Primary Metrics (Paper-Level)', fontweight='bold')
ax1.set_ylim(0, 110)
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}%', 
             ha='center', va='bottom', fontweight='bold')
plt.xticks(rotation=45, ha='right')

# 2. Task Execution Time
ax2 = plt.subplot(2, 3, 2)
task_ids = [t['task_id'] for t in aggregate['task_metrics']]
exec_times = [t['execution_time'] for t in aggregate['task_metrics']]
colors_time = ['#3498db', '#e67e22']
ax2.bar(task_ids, exec_times, color=colors_time)
ax2.set_ylabel('Time (seconds)')
ax2.set_title('Task Execution Time', fontweight='bold')
for i, (tid, time) in enumerate(zip(task_ids, exec_times)):
    ax2.text(i, time + 2, f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

# 3. Actions per Task
ax3 = plt.subplot(2, 3, 3)
actions_count = [t['total_actions'] for t in aggregate['task_metrics']]
ax3.bar(task_ids, actions_count, color=colors_time)
ax3.set_ylabel('Number of Actions')
ax3.set_title('Actions per Task', fontweight='bold')
for i, (tid, actions) in enumerate(zip(task_ids, actions_count)):
    ax3.text(i, actions + 0.3, f'{actions}', ha='center', va='bottom', fontweight='bold')

# 4. Subtasks Completion
ax4 = plt.subplot(2, 3, 4)
subtasks_completed = [t['subtasks_completed'] for t in aggregate['task_metrics']]
subtasks_total = [t['subtasks_total'] for t in aggregate['task_metrics']]
x = np.arange(len(task_ids))
width = 0.35
rects1 = ax4.bar(x - width/2, subtasks_completed, width, label='Completed', color='#2ecc71')
rects2 = ax4.bar(x + width/2, subtasks_total, width, label='Total', color='#95a5a6')
ax4.set_ylabel('Count')
ax4.set_title('Subtasks Completion', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(task_ids)
ax4.legend()
for rect in rects1:
    height = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2, height, f'{int(height)}', 
             ha='center', va='bottom', fontweight='bold')
for rect in rects2:
    height = rect.get_height()
    ax4.text(rect.get_x() + rect.get_width()/2, height, f'{int(height)}', 
             ha='center', va='bottom', fontweight='bold')

# 5. Overall Summary (Pie Chart)
ax5 = plt.subplot(2, 3, 5)
successful = sum(1 for t in aggregate['task_metrics'] if t['task_success'])
failed = len(aggregate['task_metrics']) - successful
sizes = [successful, failed]
labels = [f'Successful ({successful})', f'Failed ({failed})']
colors_pie = ['#2ecc71', '#e74c3c']
ax5.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax5.set_title('Task Success Distribution', fontweight='bold')

# 6. Metrics Summary Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
table_data = [
    ['Total Tasks', f"{aggregate['total_tasks']}"],
    ['Task Success Rate', f"{aggregate['task_success_rate']*100:.1f}%"],
    ['Step Success Rate', f"{aggregate['step_success_rate']*100:.1f}%"],
    ['Avg Actions/Task', f"{aggregate['avg_actions_per_task']:.1f}"],
    ['Avg Execution Time', f"{aggregate['avg_execution_time']:.1f}s"],
    ['Avg Subtasks Completed', f"{aggregate['avg_subtasks_completed']:.1f}"]
]
table = ax6.table(cellText=table_data, cellLoc='left', loc='center', 
                 colWidths=[0.7, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)
for i in range(len(table_data)):
    table[(i, 0)].set_facecolor('#ecf0f1')
    table[(i, 1)].set_facecolor('#ffffff')
ax6.set_title('Overall Summary', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('evaluation_results/metrics_plots.png', dpi=300, bbox_inches='tight')
print("Plots saved to evaluation_results/metrics_plots.png")
plt.show()
