w_max=interactive(inp,shear=widgets.FloatSlider(min=0.01, max=0.5, step=0.01, value=1.01),
                   tension=widgets.FloatSlider(min=1.01, max=2, step=0.01, value=1.01))


display(w_max)

refresh.run([9])
