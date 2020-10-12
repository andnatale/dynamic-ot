
# Proximal operator of kinetic energy: projection onto parabola
# ==============================================================================

import numpy as np
import warnings



def coneProjectionJDv(a,b1,b2):

    PosIndex = (a +0.5*(b1**2+b2**2)>0)
    alpha = a[PosIndex].copy()
    beta1 = b1[PosIndex].copy()
    beta2 = b2[PosIndex].copy()

    k = 0
    reste = 1
    ll = 1000.*np.ones(np.size(alpha))
    Pnum = num(ll,alpha,beta1,beta2)
    Pdiv = den(ll,alpha)
    while (k<400 and reste >1.e-9):
        ll1 = ll.copy()
        ll -=Pnum/Pdiv
        reste = np.max(abs(ll-ll1))
        k+=1
        Pnum = num(ll,alpha,beta1,beta2)
        Pdiv = den(ll,alpha)
        if k == 399:
            warnings.warn("Reached maximum iteration in cone projection")


    alpha -=ll
    beta1 /=(1+ll)
    beta2 /=(1+ll)

    an = a.copy()
    b1n = b1.copy()
    b2n = b2.copy()
    an[PosIndex] = alpha.copy()
    b1n[PosIndex] = beta1.copy()
    b2n[PosIndex] = beta2.copy()
    return an,b1n,b2n


def num(l,a,b1,b2):
    return (-l**3+(a-2)*(l**2)+(2*a-1)*l+a+b1**2/2+b2**2/2)

def den(l,a):
    return (-3*l**2+2*(a-2)*l+2*a-1)




