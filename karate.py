
import copy
import networkx
import numpy.linalg

def laplacian_matrix(g, nodes):
    # We should use networkx.laplacian_matrix, but for instructional
    # value we will do this manually.
    L = numpy.zeros(shape=(len(nodes), len(nodes)))
    for n1 in nodes:
        # For each node, for each neighbor, place a -1 at each element
        # that is diagonal.
        for n2 in g.neighbors(n1):
            L[n1, n2] = -1.
        # Diagonal is node degrees.
        L[n1, n1] = g.degree(n1)
    return L


# Get karate club graph from networkx.
g = networkx.karate_club_graph()

nodes = g.nodes()
L = laplacian_matrix(g, nodes)

ev, evec = numpy.linalg.eigh(L)

# Sort by eigenvalue
ev_ranks = numpy.argsort(ev)

print "should be zero:", ev[ev_ranks[0]]
print "next eigenvalue:", ev[ev_ranks[1]]
evec1 = evec[:,ev_ranks[1]]

group1 = [ nodes[i] for i in range(len(evec1)) if evec1[i] >= 0 ]
group2 = [ nodes[i] for i in range(len(evec1)) if evec1[i] < 0 ]
print "Group sizes: group1, group2:", len(group1), len(group2)

# Do a check to ensure that all nodes were partitioned
assert set(group1) | set(group2) == set(g.nodes())

boundary_edges = networkx.edge_boundary(g, group1)
# This should be reversible
assert len(boundary_edges) == len(networkx.edge_boundary(g, group2))


print "Cut size is", len(boundary_edges)



import pcd.graphs
g = pcd.graphs.karate_club()
#g = pcd.graphs.dolphins()
#g = pcd.graphs.football()
nodes = g.nodes()
#from fitz import interactnow

def shuffled(l):
    l = list(l)
    import random
    random.shuffle(l)
    return l


# Communities:

def comm_neighbors(g, c):
    """Return all communities connected to c"""
    neighbors = set()
    for n in comms[c]:
        neighbor_nodes = g.neighbors(n)
        neighbor_comms = [ node_comms[n] for n in neighbor_nodes
                           if node_comms[n] != c ]
        neighbors.update(neighbor_comms)
    return neighbors
def comm_degree(g, c):
    """Total degree of community"""
    degree = 0
    for n in comms[c]:
        degree += g.degree(n)
    return degree
def deltaQ(g, c1, c2):
    """Change of modularity if c1 and c2 were merged"""
    # The formula for modulary change in the paper is deceiving.
    # Derive it yourself:
    #  deltaQ = E12/M - 2*K1*K2/(2M)^2
    M = float(g.number_of_edges())
    E12 = len(networkx.edge_boundary(g, comms[c1], comms[c2]))
    #E1 = g.subgraph(comms[c1]).number_of_edges()
    #E2 = g.subgraph(comms[c2]).number_of_edges()
    K1 = sum(g.degree(n) for n in comms[c1])
    K2 = sum(g.degree(n) for n in comms[c2])
    #dQ = 2*(eij - ai*aj)
    dQ = E12/M - 2*K1*K2/(2*M)**2
    return dQ

    # Newman method:
    e12 = E12/M
    a1 = (K1 - E1)/M
    a2 = (K2 - E2)/M
    return e12 - a1*a2


def modularity(g, comms):
    """Comput modularity: Community-centric version."""
    Q = 0.0
    M = float(g.number_of_edges())
    for c, nodes in comms.iteritems():
        E_in = len(networkx.edge_boundary(g, nodes, nodes))/2
        assert E_in/2 == E_in//2
        K_in = sum(g.degree(n) for n in nodes)
        Q += E_in/(M*1) - (K_in/(2*M))**2
    return Q
def modularity2(g, comms):
    """Compute modularity: node-centric version"""
    Q = 0.0
    M = float(g.number_of_edges())
    for n1 in g.nodes_iter():
        for n2 in g.nodes_iter():
            c1 = node_comms[n1]
            c2 = node_comms[n2]
            if c1 != c2:
                continue
            if g.has_edge(n1, n2):
                Q += 1
            #if n1 == n2: continue
            Q += - g.degree(n1)*g.degree(n2) / (2.*M)
    Q = Q / (2.*M)
    return Q
#modularity = modularity2

# Make initial communities - each node in one community.
comms = dict()
node_comms = dict()
for i, n in enumerate(nodes):
    comms[i] = set((n, ))
    node_comms[n] = i


last_Q = modularity(g, comms)
best_Q = last_Q
best_comms = copy.deepcopy(comms)
while True:
    best_dQ = -1e9
    best_comm = None
    best_merge_with = None
    for c1 in comms.keys():
        for c_neighbor in comm_neighbors(g, c1):
            assert c1 != c_neighbor
            # No need to test merge every pair twice.
            if c1 > c_neighbor:
                continue
            dQ = deltaQ(g, c1, c_neighbor)
            if dQ > best_dQ:
                best_dQ = dQ
                best_comm = c1
                best_merge_with = c_neighbor
    # This will break once there is nothing more to merge.
    if best_comm is None:
        break
    #if dQ < 0:
    #    break
    # If we got here, we accept this
    c1 = best_comm
    c2 = best_merge_with
    #print "  joining", comms[c1], comms[c2]
    comms[c1].update(comms[c2])
    for n in comms[c2]:
        node_comms[n] = c1
    del comms[c2]


    Q = modularity(g, comms)
    # To show that we are computing modularity correctly, require that
    # the two different modularity computations match to within
    # floating point precision.
    Q2 = modularity2(g, comms)
    assert abs(Q-Q2) < 1e-10
    # Let's do another test.  Ensure that our dQ is equal to the
    # actual computed modularity change.
    assert abs(Q-last_Q-best_dQ) < 1e-10

    print Q, best_dQ, len(comms)
    if Q > best_Q:
        best_Q = Q
        best_comms = copy.deepcopy(comms)

    if len(comms) == 2:
        comms_2 = copy.deepcopy(comms)

    last_Q = Q

print "Optimal modularity:", best_Q
print "Optimal number of communities:", len(best_comms)
print "Optimal communities:", best_comms

group1, group2 = comms_2.values()
#print group1
#print group2

print comms_2
print modularity(g, comms_2)

boundary_edges = networkx.edge_boundary(g, group1)
print "Cut, Newman algorithm: ", len(boundary_edges)
