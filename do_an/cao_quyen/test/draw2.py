# https://plot.ly/python/box-plots/

import numpy as np
import plotly
plotly.tools.set_credentials_file(username='nguyenthieu95', api_key='aK01EuuXXjRmMo8XuSUR')
plotly.tools.set_config_file(world_readable=True, sharing='public')

y0 = np.random.randn(50)
y1 = np.random.randn(50)+1

trace0 = plotly.graph_objs.Box(
    y=y0,
    name = r'$p_c$',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = plotly.graph_objs.Box(
    y=y1,
    name = r'$p_s$',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
data = [trace0, trace1]
plotly.plotly.plot(data)