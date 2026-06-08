============================================================
RNA NEAREST-NEIGHBOR STACKING: HUBO vs QUBO
============================================================

[Design matrix]  Shape: (36, 22)  (36 data points, 22 features)
[Matrix rank]    20  (full rank = 22 means unique solution)

[Fit quality]
  RMSE      : 0.2707 kcal/mol
  R²        : 0.876613
  Max error : 0.5739 kcal/mol
  Mean |err|: 0.2159 kcal/mol

[QUBO coefficients c* — 22 values]
     c0 = -1.3212
     c1 = -1.0284
     c2 = +0.5216
     c3 = -0.0544
     c4 = -0.7993
     c5 = +0.0699
     c6 = -0.1656
    c12 = +0.8143
    c13 = +0.0583
    c14 = +0.2150
    c15 = +0.0575
    c16 = +0.0750
    c23 = -0.5783
    c24 = +0.1225
    c25 = -0.1175
    c26 = -0.0983
    c34 = +0.0950
    c35 = +0.3500
    c36 = +0.2889
    c45 = +0.5918
    c46 = -0.4350
    c56 = +0.3267

[Per-pair HUBO vs QUBO comparison]
  Stack      HUBO exact  QUBO approx      Error  Fit
  -------- ------------ ------------ ----------  ------
  AU/AU          -0.930       -1.012     -0.082  ok
  AU/UA          -1.100       -1.003     +0.097  ok
  AU/CG          -2.110       -2.290     -0.180  poor
  AU/GC          -2.240       -1.955     +0.285  poor
  AU/GU          -0.550       -0.802     -0.252  poor
  AU/UG          -1.360       -1.228     +0.132  poor
  UA/AU          -1.100       -1.171     -0.071  ok
  UA/UA          -0.930       -0.874     +0.056  ok
  UA/CG          -2.110       -2.099     +0.011  ok
  UA/GC          -1.360       -1.476     -0.116  poor
  UA/GU          -1.440       -0.866     +0.574  poor
  UA/UG          -0.550       -1.004     -0.454  poor
  CG/AU          -2.110       -1.676     +0.434  poor
  CG/UA          -1.360       -1.766     -0.406  poor
  CG/CG          -3.260       -3.072     +0.188  poor
  CG/GC          -2.360       -2.835     -0.475  poor
  CG/GU          -1.360       -1.343     +0.017  ok
  CG/UG          -2.110       -1.868     +0.242  poor
  GC/AU          -2.360       -2.414     -0.054  ok
  GC/UA          -2.240       -2.215     +0.025  ok
  GC/CG          -3.420       -3.459     -0.039  ok
  GC/GC          -3.260       -2.934     +0.326  poor
  GC/GU          -1.440       -1.986     -0.546  poor
  GC/UG          -2.510       -2.222     +0.288  poor
  GU/AU          -1.270       -1.203     +0.067  ok
  GU/UA          -1.010       -1.120     -0.110  poor
  GU/CG          -2.510       -2.424     +0.086  ok
  GU/GC          -2.110       -2.014     +0.096  ok
  GU/GU          -0.500       -0.778     -0.278  poor
  GU/UG          -1.270       -1.130     +0.140  poor
  UG/AU          -1.010       -1.304     -0.294  poor
  UG/UA          -1.270       -0.932     +0.338  poor
  UG/CG          -2.110       -2.175     -0.065  ok
  UG/GC          -1.360       -1.476     -0.116  poor
  UG/GU          -1.270       -0.784     +0.486  poor
  UG/UG          -0.500       -0.847     -0.347  poor

[Example stem energy: GC-AU-CG-GC]
  Stem       : GC — AU — CG — GC
  Pairs      : 4, stacking positions: 3
  Stack 1→2 (GC/AU): HUBO=-2.360, QUBO=-2.414, err=-0.054
  Stack 2→3 (AU/CG): HUBO=-2.110, QUBO=-2.290, err=-0.180
  Stack 3→4 (CG/GC): HUBO=-2.360, QUBO=-2.835, err=-0.475
  Total HUBO : -6.830 kcal/mol
  Total QUBO : -7.539 kcal/mol
  Total error: -0.709 kcal/mol

[Noise experiment — RMSE vs Gaussian noise sigma]
    Sigma   Mean RMSE
     0.00      0.2707  ||||||||||
     0.10      0.2835  |||||||||||
     0.20      0.2859  |||||||||||
     0.30      0.3401  |||||||||||||
     0.40      0.3750  ||||||||||||||
     0.50      0.4378  |||||||||||||||||
     0.60      0.5059  ||||||||||||||||||||
     0.70      0.5600  ||||||||||||||||||||||
     0.80      0.5707  ||||||||||||||||||||||
     0.90      0.6195  ||||||||||||||||||||||||
     1.00      0.6813  |||||||||||||||||||||||||||
     1.10      0.7671  ||||||||||||||||||||||||||||||
     1.20      0.7971  |||||||||||||||||||||||||||||||
     1.30      0.9097  ||||||||||||||||||||||||||||||||||||
     1.40      1.0125  ||||||||||||||||||||||||||||||||||||||||
     1.50      1.0201  ||||||||||||||||||||||||||||||||||||||||

[Scaling analysis]
  S (stems)         : 3
  M (total pairs)   : 12
  Symbolic summands : 36 × (M - S) = 36 × 9 = 324
  m_max             : 5
  O(S × m_max)      : 15

