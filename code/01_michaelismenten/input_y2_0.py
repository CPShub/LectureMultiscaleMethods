w_y_2_0=interactive(inp,y_2_0=widgets.FloatSlider(min=0, max=2.51, step=0.5, value=1))
display(w_y_2_0)

def run_add(ev):
    display(Javascript("Jupyter.notebook.execute_cells([8,7])"))
button = widgets.Button(description="Add")
button.on_click(run_add)
display(button)

def run_clear(ev):
    display(Javascript("Jupyter.notebook.execute_cells([9,7])"))
button = widgets.Button(description="Clear")
button.on_click(run_clear)
display(button)
