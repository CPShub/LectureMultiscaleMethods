#	Interactive Input of the deformation gradient for plane strain

w_F=interactive(inp,F11=widgets.FloatSlider(min=0.5, max=2, step=0.1, value=1),
                    F22=widgets.FloatSlider(min=0.5, max=2, step=0.1, value=1),
                    F12=widgets.FloatSlider(min=-0.60001, max=0.60001, step=0.1, value=0),
                    F21=widgets.FloatSlider(min=-0.60001, max=0.60001, step=0.1, value=0))

display(w_F)

refresh.run([4,7,12,13])