
import networkx
import numpy
import pylab

from pcd.support import algorithms

#g = networkx.karate_club_graph()
g = networkx.read_edgelist('www.dat', create_using=networkx.Graph(),
                           nodetype=int)
#print "number of nodes:", len(g)
#print "number of edges:", g.number_of_edges()
#print "number of self-loops:", g.number_of_selfloops()
for n1, n2 in g.selfloop_edges():
    g.remove_edge(n1, n2)


#import plfit
import plfit

def run_method(methname):
    sizes = [ ]
    for i in range(1):
        cda = getattr(algorithms, methname)
        cd = cda(g, trials=1)
        sizes.extend(cd.cmtys.cmtysizes().itervalues())

    sizes.sort()
    import pickle
    pickle.dump(sizes, open('sizes.pickle', 'w'), -1)
    #import pickle
    #sizes = pickle.load(open('sizes.pickle'))
    #print 'loaded'

    sizes = [x for x in sizes if x>=10]
    print min(sizes), max(sizes)
    Ncmty = len(sizes)

    #alpha, xmin, L = plfit.plfit(sizes)
    fitter = plfit.plfit(sizes)
    xmin, alpha = fitter.plfit(xmin=10, verbose=True, discrete=False)
    L = fitter._likelihood

    print "Fitted to powerlaw with alpha=%s, xmin=%s, L=%s"%(alpha,xmin,L)
    bins = bins=numpy.logspace(0,4, num=(4-0)*11+1, base=10)


    pylab.plot(sizes, [(Ncmty-i)/float(Ncmty) for i in range(Ncmty)],
               label="cCDF")
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.title("ND webgraph - complimentary CDF")
    pylab.suptitle("%s - alpha=%s, xmin=%s, L=%s"%(methname,alpha,xmin,L))
    pylab.ylabel("1-CDF(size)")
    pylab.xlabel("size")
    pylab.legend()#loc='lower left')
    pylab.savefig('webgraph-%s-sizes-cCDF.png'%methname)
    pylab.savefig('webgraph-%s-sizes-cCDF.pdf'%methname)
    pylab.plot(bins, bins**(-alpha+1)/(alpha-1), label="CDF(x^-%s)"%alpha)


    pylab.cla()
    pylab.hist(sizes,
               #bins=100,
               bins=bins,
               label="PDF", normed=True, histtype='step',
               log=True)
    pylab.plot(bins, bins**(-alpha), label="x^-%s"%alpha)

    pylab.xscale('log')
    pylab.yscale('log')
    pylab.title("ND webgraph - community sizes")
    pylab.suptitle("%s - alpha=%s, xmin=%s, L=%s"%(methname,alpha,xmin,L))
    pylab.ylabel("P(size)")
    pylab.xlabel("size")
    pylab.legend()#loc='lower left')
    pylab.savefig('webgraph-%s-sizes-PDF.png'%methname)
    pylab.savefig('webgraph-%s-sizes-PDF.pdf'%methname)


run_method("Infomap")
