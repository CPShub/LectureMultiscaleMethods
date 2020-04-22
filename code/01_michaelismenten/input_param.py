w_param=interactive(inp,y_1_0=widgets.FloatSlider(min=0, max=2.51, step=0.5, value=1),
              y_2_0=widgets.FloatSlider(min=0, max=2.51, step=0.5, value=1),
              kap=widgets.FloatSlider(min=0.1, max=2.51, step=0.1, value=1),
            lam=widgets.FloatSlider(min=0.1, max=2.51, step=0.1, value=2),
              eps=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.1))
			  
display(w_param)

refresh.run([3,9,7])