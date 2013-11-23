
import networkx
import numpy
import pylab

from pcd.support import algorithms

##g = networkx.karate_club_graph()
g = networkx.read_edgelist('www.dat', create_using=networkx.Graph(),
                           nodetype=int)
##print "number of nodes:", len(g)
##print "number of edges:", g.number_of_edges()
##print "number of self-loops:", g.number_of_selfloops()
for n1, n2 in g.selfloop_edges():
    g.remove_edge(n1, n2)


#import plfit
import plfit

def pl_fit(data, xmin):
    from math import log
    #alpha = 1 + n / sum(map(lambda X: log(float(X)/xmin),z))

    n = len([x for x in data if x >= xmin])
    alpha = 1 + n / float(sum(log(float(x)/xmin) for x in data
                          if x>=xmin) )
    return alpha

def run_method(methname):
    sizes = [ ]
    for i in range(10):
        cda = getattr(algorithms, methname)
        cd = cda(g)
        sizes.extend(cd.cmtys.cmtysizes().itervalues())
    sizes.sort()
    import pickle
    pickle.dump(sizes, open('sizes.%s.pickle'%methname, 'w'), -1)
    #import pickle
    #sizes = pickle.load(open('sizes.%s.pickle'%methname))
    #print 'loaded'
    #sizes.sort()

    #sizes = sizes[:-50]
    #sizes = [x for x in sizes if x>=10]
    #sizes = [x for x in sizes if x<=500]
    Ncmty = len(sizes)
    print Ncmty, min(sizes), max(sizes)

    # My method
    xmin = 10
    alpha = pl_fit(sizes, xmin=10)
    L = 0
    # code #1
    #import plfit_m
    #alpha, xmin, L = plfit_m.plfit(sizes)
    #plfit_m: Fitted to powerlaw with alpha=2.47, xmin=20, L=-142739.447886
    # code #2
    #fitter = plfit.plfit(sizes)
    #xmin, alpha = fitter.plfit(xmin=10, verbose=True, #finite=True,
    #                           discrete=False,
    #                           )
    #L = fitter._likelihood
    #s_params = "alpha=%s, xmin=%s, L=%s"%(alpha,xmin,L)
    s_params = "alpha=%0.2f, xmin=%s"%(alpha,xmin)

    print "Fitted to powerlaw with %s"%s_params
    bins = bins=numpy.logspace(0,4, num=(4-0)*11+1, base=10)


    pylab.plot(sizes, [(Ncmty-i-1)/float(Ncmty) for i in range(Ncmty)],
               label="cCDF")
    N = xmin**(-alpha+1)/(alpha-1)
    pylab.plot(bins, bins**(-alpha+1)/(alpha-1)/N,
               label="CDF($x^{-%0.2f}$)"%alpha)
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.title("ND webgraph - complimentary CDF - %s"%methname)
    #pylab.suptitle("%s - %s"%(methname,s_params))
    pylab.ylabel("1-CDF(size)")
    pylab.xlabel("size")
    pylab.legend()#loc='lower left')
    pylab.savefig('webgraph-%s-sizes-cCDF.png'%methname)
    pylab.savefig('webgraph-%s-sizes-cCDF.pdf'%methname)

    #from fitz import interactnow


    import collections
    hist = collections.defaultdict(int)
    from math import exp, log
    log2 = lambda x: log(x)/log(1.1)
    exp2 = lambda x: 1.1**x
    for x in sizes:
        i = int(log2(x))
        hist[i] += 1
    low = min(hist)
    high = max(hist)
    bins = numpy.arange(low, high+1)

    y = [
        #hist[i]/(exp2(i)-exp2(i-1))
        hist[i]/float(sum(hist.values()))
        for i in bins]
    bins = exp2(bins)

    aderiv = lambda x: x**(-alpha+1)/(-alpha+1)

    expected = bins**(-alpha+1)
    #expected = [
    #    aderiv(exp2(i+1)) - aderiv(exp2(i))
    #    for i in numpy.arange(low, high+1)
    #    ]
    #expected = numpy.multiply(10, expected)

    #from fitz import interactnow

    pylab.cla()
    pylab.hist(sizes,
               #bins=100,
               bins=bins,
               label="PDF", normed=True, histtype='step',
               log=True)
    #pylab.plot(bins, y, label="PDF 2")
    pylab.plot(bins, expected, label="$x^{-%0.2f}$"%alpha)

    pylab.xscale('log')
    pylab.yscale('log')
    pylab.title("ND webgraph - community sizes - %s"%methname)
    #pylab.suptitle("%s - %s"%(methname,s_params))
    pylab.ylabel("P(size)")
    pylab.xlabel("size")
    pylab.legend()#loc='lower left')
    pylab.savefig('webgraph-%s-sizes-PDF.png'%methname)
    pylab.savefig('webgraph-%s-sizes-PDF.pdf'%methname)


run_method("Infomap")
run_method("Louvain")
#run_method("Copra")
#run_method("Oslom")

