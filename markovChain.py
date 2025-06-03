from multiprocessing import Pool
import numpy as np
from statmechcrack import CrackQ2D

def Nn(L,W,Nnom,n=0,verbose=False):
    if n==0:
        return  Nnom * np.ones(W, dtype=int) 
    elif n==1:
        pass
    

def gen_N(L,W,Nnom,verbose=False):
    N0 =  Nnom * np.ones(W, dtype=int) 
    N = [N0]
    if verbose:
        print(N[-1])
    for i in range(W):
        N1 = Nnom * np.ones(W, dtype=int) 
        N1[i] = Nnom+1
        N.append(N1)
        if verbose:
            print(N[-1])

    for i in range(W):
        N1 = Nnom * np.ones(W, dtype=int) 
        N1[i] = Nnom+2
        N.append(N1)
        if verbose:
            print(N[-1])

    for i in range(W-1):
        N1 = Nnom * np.ones(W, dtype=int) 
        N1[i] = Nnom+1
        N1[i+1] = Nnom+1
        N.append(N1)
        if verbose:
            print(N[-1])

    for i in range(W-2):
        N1 = Nnom * np.ones(W, dtype=int) 
        N1[i] = Nnom+1
        N1[i+1] = Nnom+1
        N1[i+2] = Nnom+1
        N.append(N1)
        if verbose:
            print(N[-1])

    for i in range(W-1):
        N1 = Nnom * np.ones(W, dtype=int) 
        N1[i] = Nnom+1
        N1[i+1] = Nnom+2
        N.append(N1)
        if verbose:
            print(N[-1])

    for i in range(W-2):
        N1 = Nnom * np.ones(W, dtype=int) 
        N1[i] = Nnom+2
        N1[i+1] = Nnom+1
        N1[i+2] = Nnom+1
        N.append(N1)
        if verbose:
            print(N[-1])

    for i in range(W-2):
        N1 = Nnom * np.ones(W, dtype=int) 
        N1[i] = Nnom+1
        N1[i+1] = Nnom+2
        N1[i+2] = Nnom+1
        N.append(N1)
        if verbose:
            print(N[-1])
    
    return N

def calc_rates(L,W,N,verbose=False):
    model = CrackQ2D(L=L,W=W,N=N)
    rates = np.zeros(model.W)
    for k in range(model.W):
        rates[k] = model.k_isometric(2*np.ones(model.W),k)
    if verbose:
        print(N)
        print(rates)
    return [N,rates]

def saveRates(N,rates,fname):
    ofile = open(fname,'w')
    nN = len(N)
    for i in range(nN):
        for n in N[i]:
            ofile.write('{}\t'.format(n))
        ofile.write('\n')
        for r in rates[i]:
            ofile.write('{}\t'.format(r))
        ofile.write('\n')
    ofile.close()

if __name__=="__main__":
    L = 25
    W = 11
    Nnom = 7

    NAll = gen_N(L,W,Nnom,verbose=True)
    nN = len(NAll)
    print(nN)

    Nout = [[] for i in range(nN)]
    rates = [[] for i in range(nN)]

    def mapFun(N):
        rates = calc_rates(L,W,N,verbose=True)
        return rates

    nproc = 30
    #pool = Pool(processes=nproc)
    pool = Pool()
    for ind, res in enumerate(pool.imap(mapFun, NAll, chunksize=3)):
        Nout[ind] = res[0]
        rates[ind] = res[1]

    fname = 'L{}W{}.rates'.format(L,W)
    saveRates(NAll,rates,fname)
