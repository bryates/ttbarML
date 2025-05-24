import vector
import numpy as np
import pandas as pd

# Top quark mass in GeV
TOP_MASS = 172.5

# Number of events
n = 1000

# Random transverse momentum (pT) between 0 and 300 GeV
pt = np.random.uniform(0, 300, n)

# Random pseudorapidity (eta) between -2.5 and 2.5
eta = np.random.uniform(-2.5, 2.5, n)

# Random azimuthal angle (phi) between -pi and pi
phi = np.random.uniform(-np.pi, np.pi, n)

# Calculate px, py, pz
px = pt * np.cos(phi)
py = pt * np.sin(phi)
pz = pt * np.sinh(eta)

# Calculate energy assuming top quark mass
E = np.sqrt(px**2 + py**2 + pz**2 + TOP_MASS**2)

# Build Lorentz vectors
tops = vector.arr({
    "px": px,
    "py": py,
    "pz": pz,
    "E": E,
})

# Example: print the first one
print(tops[0])

df = pd.DataFrame({
    "top_px": tops.px,
    "top_py": tops.py,
    "top_pz": tops.pz,
    "top_energy": tops.E,
    "top_pt": tops.pt,
    "top_eta": tops.eta,
    "top_phi": tops.phi,
    "top_mass": tops.mass,
})

# Save to Parquet
df.to_parquet("top_quarks.parquet")
