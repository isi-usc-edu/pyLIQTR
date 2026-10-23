import IPython.display
from qualtran.drawing import GraphvizCallGraph
from qualtran import Bloq

class BasicGraph(GraphvizCallGraph):

    def get_node_title(self, b: Bloq):
        try:
            return b.pretty_name()
        except AttributeError:
            return b.__class__.__name__

def show_call_graph(call_graph):
    IPython.display.display(BasicGraph(call_graph).get_svg())