#!/usr/bin/env python3
"""
Script to plot training loss curves from Wandb runs using plotdevice.
"""

from plotdevice import *
from plotdevice.plot_with_plotly import plot
import logging
import plotly.io as pio
import pandas as pd
import plotly.graph_objects as go
import numpy as np


pio.renderers.default = "browser"

# Override plotly's default line width and background globally
go.Scatter.__init__ = (lambda original_init: 
    lambda self, *args, **kwargs: (
        kwargs.setdefault('line', {}).update({'width': 3}),
        original_init(self, *args, **kwargs)
    )[1]
)(go.Scatter.__init__)

pio.templates.default = "plotly"
pio.templates["plotly"].layout.plot_bgcolor = "lightgray"

logging.basicConfig(level=logging.INFO)

def main():
    # Runs we want to compare (OLMo3 swafix, 4k (short) context)
    swafix = WandbRun("ai2-llm", "olmo3", "OLMo3-7B-swafix", name="OLMo 3")
    short = WandbRun("ai2-llm", "olmo3", "OLMo3-7B-swafix-from229000-4k-context2", name="short context")
    
    swafix_data = swafix.get_time_series("train/CE loss")
    short_data = short.get_time_series("train/CE loss")
    
    swafix_data.name = swafix.name   # set name to OLMo 3.. keeps getting overwritten
    
    # Plot 1: Original curves
    plot(
        [swafix_data, short_data],
        ylim=(1.7, 2.5),
    )
    
    # Plot 2: Raw difference over time (no smoothing, no offset)
    common_xs = []
    differences = []
    
    # Use the shorter series' x values as reference (short context)
    for x in short_data.xs:
        # Find closest x in swafix data 
        # (steps aren't sampled exactly the same, it seems)
        swafix_distances = [abs(sx - x) for sx in swafix_data.xs]
        closest_swafix_idx = swafix_distances.index(min(swafix_distances))
        
        if min(swafix_distances) <= 100:
            short_idx = list(short_data.xs).index(x)
            
            common_xs.append(x)
            swafix_val = swafix_data.ys[closest_swafix_idx]
            short_val = short_data.ys[short_idx]
            differences.append(swafix_val - short_val)  # (OLMo 3) - (short context)
    
    # New timeseries
    difference_data = TimeSeries(
        xs=np.array(common_xs),
        ys=np.array(differences),
        name="Loss Difference (OLMo 3 - short context)"
    )
    
    plot(
        [difference_data],
        ylim=None,  # Let it auto-scale
    )

if __name__ == "__main__":
    main()
