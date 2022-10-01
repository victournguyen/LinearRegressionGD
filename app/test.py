import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv('default.csv')

print(df.to_dict('records'))