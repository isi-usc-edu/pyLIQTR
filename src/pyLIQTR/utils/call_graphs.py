import IPython.display
from qualtran.drawing.bloq_counts_graph import GraphvizCounts
from qualtran import Bloq

class BasicGraph(GraphvizCounts):

    def get_node_title(self, b: Bloq):
        try:
            return b.pretty_name()
        except AttributeError:
            return b.__class__.__name__

def show_call_graph(call_graph):
    IPython.display.display(BasicGraph(call_graph).get_svg())