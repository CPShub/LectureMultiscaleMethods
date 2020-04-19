# Create Refresh-Button
# Run cells which are defined in "ind"

from IPython.display import Javascript
import ipywidgets as widgets

def run(ind):
    """
    :ind: indices of cells to be executed

    Execution of selected cells when pressing the "Refresh" button
    """

    def run_all(ev):
        for cell_index in ind:
            display(Javascript("Jupyter.notebook.execute_cells([" + str(cell_index) + "])"))
    button = widgets.Button(description="Refresh")
    button.on_click(run_all)
    display(button)