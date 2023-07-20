import numpy as np
import matplotlib.pyplot as pl


wn = 1050*100 #1/m, si-o-si stretching frequency, measured from FTIR
b = 5e-10   # approximate lattice spacing
l = 1.55e-10 # Si-O bond length
ws = 7520 # m/s, elastic wave speed
E = 87e9 # modulus C11
udd = 8e-19 #J, Si-O bond energy in water

# set xdd to be a scaling of equilibrium bond length, if xdd=0 there is no Bell term acting
xdd = 0.0*l
k = 1.38e-23 #J/K
T = 293 # K


def velSM(K):
    R = K**2/E
    f = np.sqrt(R*E*b**3)

    omega = ws*wn

    v = b*omega/np.pi
    v *= np.exp( (f*xdd-udd)/(k*T) )
    v *= np.sinh( 0.5*(R*b**2)/(k*T) )
    return v

npt = 100
kr = np.linspace(0.1,1.0,npt)
vr = np.zeros(npt)
for i in range(npt):
    vr[i] = velSM(kr[i]*1e6)

pl.semilogy(kr,vr,'k-')
pl.show()
