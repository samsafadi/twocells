from twocells.nn import Network
from twocells.tensor import Tensor, FunctionBase, Base
from twocells.loss import LossBase

import graphviz

def vizualize_network(network: Network, output: Tensor, filename: str = 'viz-output/network.gv'):
    dot = graphviz.Digraph(type(network).__name__)
    
    _vizualize_node(output, dot)
    dot.render(filename, view=False)

def _vizualize_node(x: Base, dot: graphviz.Digraph, depth=0):
    if isinstance(x, Tensor):
        if x.parent is not None:
            name = str(x) + str(depth)
            dot.node(name)
            dot.edge(str(x.parent), name)
            _vizualize_node(x.parent, dot, depth=depth+1)
    elif isinstance(x, FunctionBase):
        for parent in x.saved_vars.values():
            dot.node(str(x), label=type(x).__name__)
            dot.edge(str(parent) + str(depth+1), str(x))
            _vizualize_node(parent, dot, depth=depth+1)
    elif isinstance(x, LossBase):
        dot.node(str(x), label=type(x).__name__)
        dot.edge(str(x.x) + str(depth+1), str(x))
        dot.edge(str(x.y) + str(depth+1), str(x))
        _vizualize_node(x.x, dot, depth=depth+1)
        _vizualize_node(x.y, dot, depth=depth+1)
        