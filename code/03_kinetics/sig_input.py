w_sig=widgets.interactive(inp,sig_11=widgets.FloatSlider(min=0, max=2, step=0.1, value=1),
                    sig_22=widgets.FloatSlider(min=0, max=2, step=0.1, value=0),
                    sig_33=widgets.FloatSlider(min=0, max=2, step=0.1, value=0),
                    sig_12=widgets.FloatSlider(min=0, max=2, step=0.1, value=1),
					sig_13=widgets.FloatSlider(min=0, max=2, step=0.1, value=0),
					sig_23=widgets.FloatSlider(min=0, max=2, step=0.1, value=0))

display(w_sig)

refresh.run([13,14])