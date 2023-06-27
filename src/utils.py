""" Projection on constraint a + |b|^2/2<=0 """
import numpy as np  

def projection(b1, b2,a):
    Index_1 = (a + 0.5*(b1**2 + b2**2)>0)
    Index_2 = ((b1 == 0) * (b2 == 0))
    
    bnulIndex = Index_1 * Index_2
    PosIndex = Index_1 * np.invert(Index_2)
    
    alpha = a[PosIndex].copy()
    beta1 = b1[PosIndex].copy()
    beta2 = b2[PosIndex].copy()
    
    norme_beta_carre = beta1**2 + beta2**2
    norme_beta = np.sqrt(norme_beta_carre)
    
    delta1 = np.sqrt(norme_beta_carre + 8*(alpha + 1)**3/27 + 0*1j)
    
    D1 = norme_beta - delta1
    D2 = norme_beta + delta1
    
    I1 = (np.real(D1) < 0)
    I1_inv = np.invert(I1)
    
    I2 = (np.real(D2) < 0)
    I2_inv = np.invert(I2)
    
    mu1 = np.zeros(len(D1)) + 0*1j
    mu1[I1]-= (-D1[I1])**(1/3)
    mu1[I1_inv]+= (D1[I1_inv])**(1/3)
    mu1[I2]-= (-D2[I2])**(1/3)
    mu1[I2_inv]+= (D2[I2_inv])**(1/3)
    
    delta2 = np.sqrt(-3*mu1**2 - 8*(alpha + 1))
    mu2 = (-mu1 - delta2)/2
    mu3 = (-mu1 + delta2)/2
    
    Mu = np.array([mu1, mu2, mu3])
    Mu[np.imag(Mu) != 0] = 0.0
    mu = np.real(Mu.max(axis = 0))
    
    alpha = -mu**2/2
    beta1 = mu*beta1/norme_beta
    beta2 = mu*beta2/norme_beta
    an = a.copy()
    b1n = b1.copy()
    b2n = b2.copy()
    
    an[PosIndex] = alpha.copy()
    b1n[PosIndex] = beta1.copy()
    b2n[PosIndex] = beta2.copy()
    
    an[bnulIndex] = 0.
    
    return  b1n, b2n,an
