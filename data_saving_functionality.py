import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ==================== CONFIGURATION ====================
S = 2.0          # KB
alpha = 1.15
R = 4
p_poor = 0.82
p_good = 0.02

# Visual customization (modify as desired)
COLOR_POOR = '#ffcccc'      # light red
COLOR_GOOD = '#ccffcc'      # light green
COLOR_SAVED_LINE = '#2ca02c'  # green line
COLOR_NAIVE_LINE = '#d62728'  # red line (optional)
FIG_SIZE = (12, 8)
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12

# ==================== COST FUNCTION ====================
def expected_cost(p):
    """Expected KB transmitted for one notification with failure prob p."""
    if p >= 1.0:
        return float('inf')
    return S * alpha * (1 - p**(R+1)) / (1 - p)

C_poor = expected_cost(p_poor)
C_good = expected_cost(p_good)
delta = C_poor - C_good   # savings per deferred message

# ==================== SIMULATION SETUP ====================
# Define the route: list of zone types ('poor' or 'good')
# You can change this to any pattern of your choice for testing.
route = ['poor', 'poor', 'good', 'good', 'poor', 'good', 
         'poor', 'poor', 'good', 'good', 'poor', 'good']

num_steps = len(route)

# Notification generation: list of step indices where non‑urgent notifications appear.
# (Alternatively, you can generate randomly or periodically.)
notification_steps = [0, 1, 4, 6, 7, 10]   # example

# ==================== RUN SIMULATION ====================
queue = []               # pending non‑urgent notifications
cumulative_saved = []    # list of cumulative savings after each step
delivery_log = []        # (step, zone, naive_cost, smart_cost, saved)
saved_so_far = 0.0

# For baseline comparison: track what would have been spent if sending immediately
naive_total_cost = 0.0

for step in range(num_steps):
    zone = route[step]
    
    # 1. Generate new non‑urgent notification if scheduled
    if step in notification_steps:
        # For simulation, we record its arrival step and hypothetical cost if sent immediately
        arrival_cost = expected_cost(p_poor) if route[step] == 'poor' else expected_cost(p_good)
        queue.append({
            'arrival_step': step,
            'arrival_zone': route[step],
            'arrival_cost': arrival_cost
        })
        naive_total_cost += arrival_cost
    
    # 2. Deliver queued notifications if in good coverage
    if zone == 'good' and queue:
        num_delivered = len(queue)
        for notif in queue:
            saved_so_far += delta
            # Log each delivery (could be batched, but one row per notif for detail)
            delivery_log.append([
                step,
                zone,
                f"{notif['arrival_cost']:.2f} KB",
                f"{C_good:.2f} KB",
                f"+{delta:.2f} KB"
            ])
        queue.clear()
    
    # 3. Record cumulative savings after this step
    cumulative_saved.append(saved_so_far)

# After simulation, any remaining queued messages? (not delivered)
# They do not contribute to savings.

# ==================== PLOTTING ====================
fig = plt.figure(figsize=FIG_SIZE)

# ----- Top subplot: Cumulative Savings -----
ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
steps = np.arange(num_steps)

# Plot cumulative savings as step function (post-step values)
ax1.step(steps, cumulative_saved, where='post', 
         color=COLOR_SAVED_LINE, linewidth=2.5, label='Cumulative Data Saved')
ax1.set_ylabel('Data Saved (KB)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Geo-Deferred Notification Savings Over Trip', fontsize=FONT_SIZE_TITLE)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left')

# Add a shaded background indicating coverage zones
for i, zone in enumerate(route):
    color = COLOR_GOOD if zone == 'good' else COLOR_POOR
    ax1.axvspan(i, i+1, facecolor=color, alpha=0.3)

# Create custom legend for zones
poor_patch = mpatches.Patch(color=COLOR_POOR, alpha=0.5, label='Poor Coverage')
good_patch = mpatches.Patch(color=COLOR_GOOD, alpha=0.5, label='Good Coverage')
ax1.legend(handles=[poor_patch, good_patch, plt.Line2D([0], [0], color=COLOR_SAVED_LINE, lw=2.5)],
           labels=['Poor Coverage', 'Good Coverage', 'Cumulative Saved'],
           loc='upper left')

# Annotate savings values at each step (optional)
for step, val in enumerate(cumulative_saved):
    ax1.annotate(f'{val:.1f}', (step, val), textcoords="offset points", 
                 xytext=(0,5), ha='center', fontsize=9)

# ----- Bottom subplot: Delivery Log Table -----
ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)
ax2.axis('tight')
ax2.axis('off')

# Prepare table data
if delivery_log:
    columns = ['Step', 'Zone', 'Naive Cost', 'Smart Cost', 'Saved']
    # Add a total row at the bottom
    total_saved = saved_so_far
    table_data = delivery_log + [['', '', '', 'TOTAL SAVED:', f'{total_saved:.2f} KB']]
    
    table = ax2.table(cellText=table_data, colLabels=columns,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color header row
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i == len(table_data):  # total row
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(fontweight='bold')
        # Color zone column based on coverage
        if j == 1 and i > 0 and i <= len(delivery_log):
            zone_val = delivery_log[i-1][1]
            cell.set_facecolor(COLOR_GOOD if zone_val == 'good' else COLOR_POOR)
else:
    ax2.text(0.5, 0.5, 'No deferred notifications delivered yet.', 
             ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()

# ==================== PRINT SUMMARY ====================
print(f"Total naive cost (immediate send): {naive_total_cost:.2f} KB")
print(f"Total deferred cost: {(naive_total_cost - saved_so_far):.2f} KB")
print(f"Total data saved: {saved_so_far:.2f} KB")
print(f"Percentage saved: {(saved_so_far / naive_total_cost * 100):.1f}%")