import numpy as np
def matrix_quantities(t, G, s2):
    t=np.asarray(t,float); n=len(t); Z=np.column_stack([np.ones(n),t]); V=Z@G@Z.T+s2*np.eye(n)
    w=np.linalg.solve(V,Z@G[:,1]); tau2=G[1,1]; D2=w@Z@G[:,1]/tau2
    SS=((t-t.mean())**2).sum(); lam=tau2*SS/s2; R=lam/(1+lam)
    S=G[0,0]-G[0,1]**2/tau2; kappa=(w.sum()**2)*S/(tau2*D2)
    return D2,R,D2-R,kappa
def planning_quantities(t, G, s2):
    """lambda, m, rho_bar; requires tau2 > 0, s2 > 0 and Var(L) > 0 (L = level at the mean time)."""
    t=np.asarray(t,float); n=len(t); tb=t.mean(); tau2=G[1,1]
    VL=G[0,0]+2*tb*G[0,1]+tb**2*tau2; covLs=G[0,1]+tb*tau2
    if not (tau2>0 and s2>0 and VL>0):
        raise ValueError("planning quantities need tau2 > 0, sigma2 > 0 and Var(L) > 0")
    rho=covLs/np.sqrt(VL*tau2); m=n*VL/s2; lam=tau2*((t-tb)**2).sum()/s2
    return lam,m,rho
def from_planning(lam,m,rho):
    c=rho**2*m/(1+m*(1-rho**2)); R=lam/(1+lam); D2=(lam+c)/(1+lam+c); Df=D2-R
    sh=m*(1-rho**2)/(m*(1-rho**2)+1)
    # kappa is undefined when there is no determinacy (lambda + c = 0): return NaN, never 0
    kappa=Df*(1-D2)/(D2*(1-R))*sh if lam+c>0 else float("nan")
    return D2,R,Df,kappa
