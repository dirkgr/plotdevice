from plotdevice import *
from plotdevice.plot_with_plotly import plot
import plotly.io as pio
import logging
import numpy as np
import csv

logging.basicConfig(level=logging.INFO)

pio.renderers.default = "browser"
import plotly.graph_objects as go

# Override plotly's default line width and background globally
go.Scatter.__init__ = (lambda original_init:
    lambda self, *args, **kwargs: (
        kwargs.setdefault('line', {}).update({'width': 2}),
        original_init(self, *args, **kwargs)
    )[1]
)(go.Scatter.__init__)

pio.templates.default = "plotly"
pio.templates["plotly"].layout.plot_bgcolor = "lightgray"

# Define the runs with updated variable names
swafix = WandbRun("ai2-llm", "olmo3", "OLMo3-7B-swafix", name="OLMo 3")
idm_dis = WandbRun("ai2-llm", "olmo3", "OLMo3-7B-swafix-from229000-idm-disabled", name="idm disabled")
runs = [swafix, idm_dis]

# Get time series data
swafix_data = swafix.get_time_series("train/CE loss")
idm_dis_data = idm_dis.get_time_series("train/CE loss")

# Plot original curves
plot(
    [swafix_data, idm_dis_data],
    ylim=(1.7, 2.5),
)

# Plot smoothed curves
plot(
    [swafix_data, idm_dis_data],
    moving_average_smoothing=100,
    ylim=(1.7, 2.5),
)

# Create dictionaries to map step -> loss value
swafix_step_to_loss = {step: loss for step, loss in zip(swafix_data.xs, swafix_data.ys)}
idm_dis_step_to_loss = {step: loss for step, loss in zip(idm_dis_data.xs, idm_dis_data.ys)}

# Calculate difference to show drift over time
common_xs = []
differences = []

print("First 10 step comparisons:")
print("step_x | swafix_loss | idm_dis_loss | difference")
print("-" * 50)

for step in idm_dis_data.xs:
    if step in swafix_step_to_loss:
        swafix_val = swafix_step_to_loss[step]
        idm_dis_val = idm_dis_step_to_loss[step]
        diff = swafix_val - idm_dis_val
        
         ## Print first 10 values
         #if len(common_xs) < 10:
         #   print(f"{step:6d} | {swafix_val:11.6f} | {idm_dis_val:12.6f} | {diff:10.6f}")
        
        common_xs.append(step)
        differences.append(diff)

difference_data = TimeSeries(
    xs=np.array(common_xs),
    ys=np.array(differences),
    name="Loss Difference (OLMo 3 Loss - IDM Disabled Loss)"
)

# Plot raw difference (shows drift over time)
plot(
    [difference_data],
)

# Plot smoothed difference
plot(
    [difference_data],
    moving_average_smoothing=100,
)

# Save CSV with losses and raw differences
# with open('loss_analysis.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['step', 'olmo3_loss', 'idm_disabled_loss', 'loss_diff'])
#     
#     for i, step in enumerate(common_xs):
#         swafix_idx = list(swafix_data.xs).index(step)
#         idm_dis_idx = list(idm_dis_data.xs).index(step)
#         
#         writer.writerow([
#             step,
#             swafix_data.ys[swafix_idx],
#             idm_dis_data.ys[idm_dis_idx],
#             differences[i]
#         ])

# print(f"Saved loss analysis to loss_analysis.csv")
