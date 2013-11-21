
import numpy
import random

import networkx

g = networkx.read_edgelist('www.dat', create_using=networkx.Graph(),
                           nodetype=int)

#networkx.write_edgelist(g, 'www-normalized.dat')


def remove_fraction(g, f, style='random'):
    """Remove a fraction of the nodes of the graph.  Return subgraph.

    This recalculates _everything_ on each call.  Some of this (random
    ordering, subgraphs, etc) could be cached.
    """
    nodes = g.nodes()
    if style == 'random':
        random.shuffle(nodes)
    elif style == 'degree_highest':
        #nodes = g.nodes()
        nodes_degrees = networkx.degree(g)
        # nodes_degrees is a mapping node -> degree
        degrees = [nodes_degrees[n] for n in nodes]
        dn = zip(degrees, nodes)
        dn.sort(reverse=True)
        degrees, nodes = zip(*dn)
        # nodes and degrees are now lists of nodes and degrees, sorted
        # by degrees.  Highest degree is first.
    else:
        raise ValueError("Unknown style %s"%style)

    removed_nodes   = nodes[:int(len(nodes)*f)]
    remaining_nodes = nodes[int(len(nodes)*f):]


    subg = g.subgraph(remaining_nodes)
    return subg

def lcs(g):
    """Largest component size.

    Breaks a graph into all components.  Returns the size of the
    largest component.
    """
    components = networkx.connected_components(g)
    # component is a list of lists of nodes.  First element is the
    # largest component.
    largest_component_nodes = components[0]
    largest_component_size = len(largest_component_nodes)
    return largest_component_size

def compute_S(g, f, style):
    """Compute fraction of graph connected, after removing `f` nodes.

    
    """
    if f == 1.0:
        raise ValueError("Removing all nodes doesn't work!")
    N = len(g)
    new_graph = remove_fraction(g, f, style=style)
    size = lcs(new_graph)
    S = size / float(len(new_graph))#float(N)
    return S


def run_range(f, style='random', trials=10):
    Ss = [ ]
    for f in fs:
        vals = [ ]
        for i in range(trials):
            print f, i
            vals.append(compute_S(g, f, style=style))
            print "   ", vals[-1]
        Ss.append(numpy.mean(vals))
    return Ss

#fs = numpy.arange(0, .2, .05)
fs = numpy.arange(0, 1.0, .05)
trials = 25

#import pyplot
import pylab
Ss = run_range(fs, style='random', trials=trials)
pylab.plot(fs, Ss, label="random")

# Removing highest degree nodes is deterministic (ignoring ordering of
# same-degree nodes, so we only need one trial here.  To solve that,
# we could randomize node order, then sort it by degree.).
Ss = run_range(fs, style='degree_highest', trials=1)
pylab.plot(fs, Ss, label="degree_highest")

pylab.yscale('log')
pylab.title("ND webgraph - robustness")
pylab.ylabel("largest component fraction")
pylab.xlabel("fraction of nodes removed")
pylab.legend(loc='lower left')
pylab.savefig('webgraph-S.png')
pylab.savefig('webgraph-S.pdf')
