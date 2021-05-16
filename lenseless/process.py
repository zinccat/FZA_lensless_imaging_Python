
S = 2*dp*Nx  # aperture diameter
r1 = 0.23  # FZA constant

M = di/z1
ri = (1+M)*r1

mask = FZA(S, 2*Nx, ri)  # generate the FZA mask
