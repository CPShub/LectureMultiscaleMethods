param=list(w_param.result.values())
solution=ode.solve(param)
ode.plot_y1_y2(solution,param)