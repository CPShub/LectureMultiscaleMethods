w_Neo_Hooke=widgets.Checkbox(
    value=True,
    description='Neo-Hooke',
    disabled=False,
    indent=False
)

w_Ogden_1=widgets.Checkbox(
    value=False,
    description='Ogden (n=1)',
    disabled=False,
    indent=False
)

w_Ogden_2=widgets.Checkbox(
    value=False,
    description='Ogden (n=2)',
    disabled=False,
    indent=False
)

w_Ogden_3=widgets.Checkbox(
    value=False,
    description='Ogden (n=3)',
    disabled=False,
    indent=False
)

w_Mooney_Rivlin=widgets.Checkbox(
    value=False,
    description='Mooney-Rivlin',
    disabled=False,
    indent=False
)

display(w_Neo_Hooke)
display(w_Mooney_Rivlin)
display(w_Ogden_1)
display(w_Ogden_2)
display(w_Ogden_3)

refresh.run([16])