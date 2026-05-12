import numpy as np
from scipy.interpolate import PchipInterpolator

mach_data = np.array([0.65, 0.9, 1.1, 1.3, 2.0, 5.37, 7.38, 10.61])

a_data = np.array([0.3804, 0.3418, 0.3459, 0.4006, 0.6049, 1.0314, 1.2753, 1.1948])
b_data = np.array([-0.0011, 0.0100, 0.0012, 0.0037, 0.0010, 0.0145, 0.0354, 0.0962])
c_data = np.array([0.0070, 0.0174, 0.0382, 0.0337, 0.0268, 0.0121, 0.0101, 0.0081])

a_interp = PchipInterpolator(mach_data, a_data)
b_interp = PchipInterpolator(mach_data, b_data)
c_interp = PchipInterpolator(mach_data, c_data)

def mach_regime(M):
    if M < 0.8:
        return "subsonic"
    elif M < 1.2:
        return "transonic"
    elif M < 5.0:
        return "supersonic"
    else:
        return "hypersonic"

def cd_from_mach_cl(M, CL):
    if M < mach_data.min() or M > mach_data.max():
        raise ValueError("Mach number outside available data range.")

    a = a_interp(M)
    b = b_interp(M)
    c = c_interp(M)

    CD = a * CL**2 + b * CL + c

    return CD, mach_regime(M)

CD, regime = cd_from_mach_cl(3.0, 0.4)

print(CD)
print(regime)