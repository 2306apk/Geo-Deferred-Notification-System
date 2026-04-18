"""
Geo-Deferred Notification Simulator with Dynamic Visualization
================================================================
This script simulates a vehicle moving along a predefined route with
alternating good and poor network coverage zones. Non‑urgent notifications
are queued and delivered only when the vehicle enters a good‑coverage
segment, thereby saving data by avoiding costly retransmissions.

The mathematical model calculates expected data consumption using a
finite‑retry geometric series, yielding a fixed saving per deferred
message. The simulation tracks cumulative savings step‑by‑step and
visualises the results dynamically.

Author: [Your Name]
Date: [Current Date]
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

# ==================== CONFIGURATION ====================
# ---------- Notification / Protocol Parameters ----------
S = 2.0                 # Payload size per notification (KB)
alpha = 1.15            # Protocol overhead multiplier (headers, ACKs, etc.)
R = 4                   # Maximum retry attempts after initial failure
p_poor = 0.82           # Probability of transmission failure in poor coverage
p_good = 0.02           # Probability of transmission failure in good coverage

# ---------- Visual Customisation (UI) ----------
COLOR_POOR = '#ffcccc'          # Light red background for poor coverage
COLOR_GOOD = '#ccffcc'          # Light green background for good coverage
COLOR_SAVED_LINE = '#2ca02c'    # Green step line for cumulative savings
COLOR_NAIVE_LINE = '#d62728'    # Red dashed line for naive cost (optional)
COLOR_CAR = 'red'               # Car marker colour
FIG_SIZE = (12, 8)
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
ANIMATION_INTERVAL = 800        # Milliseconds between frames

# ==================== MATHEMATICAL MODEL ====================
def expected_cost(p: float) -> float:
    """
    Expected total data transmitted (KB) for one notification given a
    per‑attempt failure probability p.
    
    The function models the retransmission behaviour:
        - Initial send always occurs.
        - Up to R retries are attempted if failures persist.
        - Each attempt consumes S * alpha KB.
    
    The expected number of attempts is the sum of probabilities:
        1 (first attempt) + p (first retry) + p^2 (second retry) + ... + p^R
    which equals (1 - p^(R+1)) / (1 - p) for p < 1.
    
    Args:
        p (float): Probability that a single transmission fails.
    
    Returns:
        float: Expected data transmitted in KB.
    """
    if p >= 1.0:   # Complete failure – cost is infinite (message never delivered)
        return float('inf')
    # Geometric series sum: 1 + p + p^2 + ... + p^R
    attempts = (1 - p**(R+1)) / (1 - p)
    return S * alpha * attempts

# Precompute costs for poor and good coverage (using the fixed parameters)
C_poor = expected_cost(p_poor)   # Expected KB if sent in poor zone
C_good = expected_cost(p_good)   # Expected KB if sent in good zone
delta = C_poor - C_good          # Data saved per deferred message (KB)

# ==================== SIMULATION SETUP ====================
# ---------- Define the Route ----------
# List of zone types along the trip. You can modify this sequence.
route = ['poor', 'poor', 'good', 'good', 'poor', 'good',
         'poor', 'poor', 'good', 'good', 'poor', 'good']
num_steps = len(route)

# ---------- Notification Generation Schedule ----------
# Step indices at which a non‑urgent notification is generated.
# In a real system this could be random or triggered by events.
notification_steps = [0, 1, 4, 6, 7, 10]

# ==================== SIMULATION STATE ====================
queue = []                # Pending non‑urgent notifications (list of dicts)
saved_so_far = 0.0        # Cumulative data saved (KB)
naive_total_cost = 0.0    # Total data that would have been used without deferral

# These lists store the history for plotting
cumulative_saved_history = []   # savings value after each step
delivery_log = []               # details of each delivered message (step, zone, ...)

# ==================== RUN FULL SIMULATION (for static data) ====================
# We run the simulation once to collect all data needed for the dynamic plot.
# The animation will simply replay these precomputed states.

# Temporary variables for the full run
temp_queue = []
temp_saved = 0.0
temp_naive = 0.0

for step in range(num_steps):
    zone = route[step]
    
    # 1. Generate new non‑urgent notification if scheduled
    if step in notification_steps:
        # Determine expected cost if sent immediately from this zone
        arrival_cost = expected_cost(p_poor) if zone == 'poor' else expected_cost(p_good)
        temp_queue.append({
            'arrival_step': step,
            'arrival_zone': zone,
            'arrival_cost': arrival_cost
        })
        temp_naive += arrival_cost
    
    # 2. Deliver queued notifications if currently in good coverage
    if zone == 'good' and temp_queue:
        for notif in temp_queue:
            temp_saved += delta
            delivery_log.append([
                step,
                zone,
                f"{notif['arrival_cost']:.2f} KB",
                f"{C_good:.2f} KB",
                f"+{delta:.2f} KB"
            ])
        temp_queue.clear()
    
    # 3. Record cumulative savings after this step
    cumulative_saved_history.append(temp_saved)

# After loop, store final values for summary printing
final_saved = temp_saved
final_naive = temp_naive

# ==================== DYNAMIC ANIMATION ====================
# Set up the figure and axes
fig = plt.figure(figsize=FIG_SIZE)

# Top subplot: cumulative savings over time
ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
steps = np.arange(num_steps)

# Static background elements (coverage zones, grid, labels)
for i, zone in enumerate(route):
    color = COLOR_GOOD if zone == 'good' else COLOR_POOR
    ax1.axvspan(i, i+1, facecolor=color, alpha=0.3)

ax1.set_xlim(-0.5, num_steps - 0.5)
ax1.set_ylim(0, max(cumulative_saved_history) * 1.1 if cumulative_saved_history else 10)
ax1.set_ylabel('Data Saved (KB)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Geo‑Deferred Notification Savings Over Trip', fontsize=FONT_SIZE_TITLE)
ax1.grid(True, linestyle='--', alpha=0.6)

# Legend patches
poor_patch = mpatches.Patch(color=COLOR_POOR, alpha=0.5, label='Poor Coverage')
good_patch = mpatches.Patch(color=COLOR_GOOD, alpha=0.5, label='Good Coverage')
ax1.legend(handles=[poor_patch, good_patch], loc='upper left')

# Dynamic elements: savings line (initially empty) and car marker
savings_line, = ax1.step([], [], where='post', color=COLOR_SAVED_LINE, linewidth=2.5)
car_marker, = ax1.plot([], [], 'o', color=COLOR_CAR, markersize=12, zorder=5)

# Annotation for current savings value
savings_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Bottom subplot: delivery log table (static, shows all deliveries at end)
ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=2)
ax2.axis('off')

# Prepare table data (will be displayed fully after animation ends,
# but we can also update it dynamically if desired)
if delivery_log:
    columns = ['Step', 'Zone', 'Naive Cost', 'Smart Cost', 'Saved']
    table_data = delivery_log + [['', '', '', 'TOTAL SAVED:', f'{final_saved:.2f} KB']]
    table = ax2.table(cellText=table_data, colLabels=columns,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # header
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', fontweight='bold')
        elif i == len(table_data):  # total row
            cell.set_facecolor('#f0f0f0')
            cell.set_text_props(fontweight='bold')
        if j == 1 and 0 < i <= len(delivery_log):
            zone_val = delivery_log[i-1][1]
            cell.set_facecolor(COLOR_GOOD if zone_val == 'good' else COLOR_POOR)
else:
    ax2.text(0.5, 0.5, 'No deferred notifications delivered yet.',
             ha='center', va='center', fontsize=12)

plt.tight_layout()

# ==================== ANIMATION FUNCTIONS ====================
def init():
    """Initialisation function for FuncAnimation."""
    savings_line.set_data([], [])
    car_marker.set_data([], [])
    savings_text.set_text('')
    return savings_line, car_marker, savings_text

def animate(frame):
    """
    Update the plot for the given frame (step index).
    frame runs from 0 to num_steps - 1.
    """
    current_step = frame
    # Update savings line: show data up to current_step
    x_data = np.arange(current_step + 1)
    y_data = cumulative_saved_history[:current_step + 1]
    savings_line.set_data(x_data, y_data)
    
    # Update car marker: position at current step (centred in zone)
    car_marker.set_data([current_step + 0.5], [0])  # place at bottom of plot for visibility
    
    # Update text annotation
    current_saved = cumulative_saved_history[current_step] if current_step < len(cumulative_saved_history) else 0
    zone = route[current_step]
    savings_text.set_text(f'Step: {current_step}  |  Zone: {zone.upper()}  |  Saved: {current_saved:.2f} KB')
    
    return savings_line, car_marker, savings_text

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init,
                              frames=num_steps, interval=ANIMATION_INTERVAL,
                              blit=True, repeat=False)

# To save the animation as a GIF or video, uncomment one of the following lines:
# ani.save('geo_deferred_simulation.gif', writer='pillow', fps=1)
# ani.save('geo_deferred_simulation.mp4', writer='ffmpeg', fps=1)

plt.show()

# ==================== PRINT SUMMARY ====================
print("========== SIMULATION SUMMARY ==========")
print(f"Route length: {num_steps} steps")
print(f"Non‑urgent notifications generated: {len(notification_steps)}")
print(f"Total naive cost (immediate send): {final_naive:.2f} KB")
print(f"Total deferred cost: {(final_naive - final_saved):.2f} KB")
print(f"Total data saved: {final_saved:.2f} KB")
if final_naive > 0:
    print(f"Percentage saved: {(final_saved / final_naive * 100):.1f}%")
else:
    print("Percentage saved: N/A (no notifications generated)")
print("=========================================")