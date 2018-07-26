### https://plot.ly/python/box-plots/

import plotly
import plotly.graph_objs as go
import numpy as np

y0 = np.random.randn(50)
y1 = np.random.randn(50)+1

trace0 = go.Box(
    y=y0,
    name = r'Sample A',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = r'Sample B',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
plotly.offline.plot(data)