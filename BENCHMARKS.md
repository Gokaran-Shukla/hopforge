# hopforge benchmarks against published results

hopforge 0.5.0: **37/37 pass**. Regenerate with `python -m hopforge.benchmarks -j -1`.

| # | Benchmark | Quantity | Published | hopforge | Status | Reference |
|---|---|---|---|---|---|---|
| 1 | Graphene Dirac velocity | v_F = 3 t a_cc / 2 | 0.866025 | 0.866024 | pass | Wallace, Phys. Rev. 71, 622 (1947); Castro Neto et al., RMP 81, 109 (2009) |
| 2 | AB bilayer graphene gap under bias | E_g = |U| g1 / sqrt(U^2 + g1^2) | 0.097014 | 0.097014 | pass | McCann, PRB 74, 161403(R) (2006) |
| 3 | Silicon sp3s* (Vogl) indirect gap | E_g(Si) [eV] | 1.17 | 1.171 | pass | Vogl, Hjalmarson & Dow, J. Phys. Chem. Solids 44, 365 (1983) |
| 4 | Lieb lattice flat band | E_flat = 0 | 0.0 | 0.0 | pass | Lieb, PRL 62, 1201 (1989) |
| 5 | Kagome flat band | E_flat = +2t | 2.0 | 2.0 | pass | Mielke, J. Phys. A 24, L73 (1991) |
| 6 | Zigzag graphene ribbon edge states | flat E=0 band for 2pi/3 < |k a| < pi | E=0 for |k|/2pi > 1/3 | max|E| in=5.5e-13, min|E| out=0.42 | pass | Fujita et al., JPSJ 65, 1920 (1996); Nakada et al., PRB 54, 17954 (1996) |
| 7 | Armchair ribbons: metallic iff N = 3p + 2 | gap(N) for N = 18, 20, 22, 24, 26 | metallic at N = 20, 26 | [0.188, 0.0, 0.16, 0.143, 0.0] | pass | Nakada et al., PRB 54, 17954 (1996); Son, Cohen & Louie, PRL 97, 216803 (2006) |
| 8 | Haldane phase boundary | C = 1 for |M| < 3 sqrt(3) t2 sin(phi), else 0 | C: 1 -> 0 at M_c | (1, 0, 'gap(M_c)=9.4e-16') | pass | Haldane, PRL 61, 2015 (1988) |
| 9 | Qi-Wu-Zhang Chern numbers | C(m) = -sgn(m) for 0<|m|<2, 0 for |m|>2 (hopforge sign) | [0, 1, -1, 0] | [0, 1, -1, 0] | pass | Qi, Wu & Zhang, PRB 74, 085308 (2006) |
| 10 | Hofstadter TKNN Chern numbers | Diophantine r = q s_r + p t_r, C_r = t_r - t_{r-1} | TKNN integers (up to the global B sign) | 1/3:[1, -2, 1]; 1/4:[1, 1]; 1/5:[1, 1, -4, 1, 1]; 2/5:[-2, 3, -2, 3, -2]; 3/7:[-2, 5, -2, -2, -2, 5, -2] | pass | Thouless, Kohmoto, Nightingale & den Nijs, PRL 49, 405 (1982) |
| 11 | Kane-Mele QSH phase boundary | Z2 = 1 for lambda_v < 3 sqrt(3) lambda_SO | (1, 0) | (1, 0) | pass | Kane & Mele, PRL 95, 146802 (2005) |
| 12 | Kane-Mele Rashba stability | QSH gap closes at lambda_R = 2 sqrt(3) lambda_SO (lambda_v = 0) | gap > 0, Z2 = 1 below; gap closed at the boundary | ('gap(0.8)=0.1172', 'Z2=1', 'gap(1.0)=-0.0138') | pass | Kane & Mele, PRL 95, 226801 (2005) |
| 13 | BHZ quantum spin Hall phase | Z2 = 1 for 0 < M/B < 4 (and 4 < M/B < 8 on the lattice) | [0, 1, 1, 1, 0] | [0, 1, 1, 1, 0] | pass | Bernevig, Hughes & Zhang, Science 314, 1757 (2006) |
| 14 | Fu-Kane-Mele 3D topological insulator | (nu0;nu1nu2nu3) = (1;111) dt>0, (0;111) dt<0 | ((1, 1, 1, 1), (0, 1, 1, 1)) | ((1, 1, 1, 1), (0, 1, 1, 1)) | pass | Fu, Kane & Mele, PRL 98, 106803 (2007) |
| 15 | Bi2Se3-type continuum TI | (1;000) with one surface Dirac cone at Gamma | ((1, 0, 0, 0), 0.0) | ((1, 0, 0, 0), 0.0) | pass | Zhang et al., Nat. Phys. 5, 438 (2009) |
| 16 | SSH polarisation and winding | P = e/2 (mod e), W = 1 in the topological phase | (3.141593, 1) | (3.141593, 1.0) | pass | Su, Schrieffer & Heeger, PRL 42, 1698 (1979); King-Smith & Vanderbilt, PRB 47, 1651 (1993) |
| 17 | Kitaev chain phase boundary | topological (W = 1) for |mu| < 2t | [1, 0, 1, 0] | [1, 0, 1, 0] | pass | Kitaev, Phys.-Usp. 44, 131 (2001) |
| 18 | Thouless pump | one electron per adiabatic cycle | 1.0 | 1.0 | pass | Thouless, PRB 27, 6083 (1983) |
| 19 | Weyl semimetal anomalous Hall | sigma_xy = (e^2/h) dk_W / 2pi per length | 1937.0 | 1934.6 | pass | Burkov & Balents, PRL 107, 127205 (2011) |
| 20 | Landau levels of the square lattice | E_n = -4t + (n + 1/2) w_c, w_c = 4 pi phi t (phi -> 0) | [-3.8953, -3.6858] | [-3.8966, -3.6926] | pass | Landau (1930); Hofstadter, PRB 14, 2239 (1976) |
| 21 | Kagome topological insulator (Guo-Franz) | Z2 = 1 at 1/3 filling with intrinsic SOC | 1 | (1, 'gap=0.346') | pass | Guo & Franz, PRB 80, 113102 (2009) |
| 22 | Quantised anomalous Hall | sigma_xy = C e^2/h (Haldane, C = 1) | 1.0 | 1.0 | pass | TKNN, PRL 49, 405 (1982) |
| 23 | Quantised spin Hall conductance (Kane-Mele) | |sigma^sz_xy| = e/2pi | 1.0 | 1.0 | pass | Kane & Mele, PRL 95, 226801 (2005) |
| 24 | Graphene universal optical conductivity | Re sigma = e^2/(4 hbar) incl. spin (1/8 spinless) | 0.125 | 0.1268 | pass | Ando, Zheng & Suzuki, JPSJ 71, 1318 (2002); Nair et al., Science 320, 1308 (2008) |
| 25 | Valley orbital moment of gapped graphene | m(K) = e v^2 hbar / (2 Delta) (e = hbar = 1: v^2/2Delta) | 7.5 | 7.5 | pass | Xiao, Yao & Niu, PRL 99, 236809 (2007) |
| 26 | Streda formula in a Chern insulator | dM/dmu = C e/h (per cell C A / 2pi) | 0.137832 | 0.137832 | pass | Streda, J. Phys. C 15, L717 (1982); Ceresoli et al., PRB 74, 024408 (2006) |
| 27 | Wiedemann-Franz law | L = pi^2/3 (k_B/e)^2 | 3.2899 | 3.2688 | pass | Sommerfeld (1928); Ashcroft & Mermin, ch. 13 |
| 28 | KPM Kubo-Bastin Hall conductivity | sigma_xy(mu in gap) = -C e^2/h, matches k-space Kubo | -1.0 | -0.987 | pass | Garcia, Covaci & Rappoport, PRL 114, 116602 (2015) |
| 29 | Quantised conductance of a ballistic wire | G = N e^2/h | [1, 2, 3, 5] | [1.0, 2.0, 3.0, 5.0] | pass | van Wees et al., PRL 60, 848 (1988) |
| 30 | Integer quantum Hall resistance | R_xy = h/(nu e^2), R_xx = 0 | [1, 0.5, 0.3333] | [1.0, 0.5, 0.3333] | pass | von Klitzing et al., PRL 45, 494 (1980) |
| 31 | Quantum spin Hall edge conductance | G = 2 e^2/h, robust to disorder | 2.0 | 2.0 | pass | Koenig et al., Science 318, 766 (2007) |
| 32 | Diffusive shot noise | Fano factor F = 1/3 | 0.3333 | 0.352 +- 0.016 | pass | Beenakker & Buettiker, PRB 46, 1889 (1992) |
| 33 | Andreev reflection at a transparent NS interface | G_NS = 2 N e^2/h below the gap | 10 | 9.904 | pass | Blonder, Tinkham & Klapwijk, PRB 25, 4515 (1982) |
| 34 | Majorana zero-bias peak | G(V=0) = 2 e^2/h, independent of the tunnel barrier | 2.0 | [2.0, 2.0] | pass | Law, Lee & Ng, PRL 103, 237001 (2009) |
| 35 | Honeycomb Hubbard mean-field AFM transition | U_c = 2.23 t | m = 0 at 2.12t, m > 0 at 2.36t | (0.0055, 0.1129) | pass | Sorella & Tosatti, EPL 19, 699 (1992) |
| 36 | Square-lattice Hubbard mean field | gap = U |m| (Slater AFM) | 0.7522 | 0.7522 | pass | Slater, PR 82, 538 (1951); Hirsch, PRB 31, 4403 (1985) |
| 37 | Light-induced Chern insulator in graphene | circular light: C = +-1 (sign set by helicity); linear: C = 0 | (+-1, -+1, 0) | (1, -1, 0, 'gap eff/exact=1.0012') | pass | Oka & Aoki, PRB 79, 081406 (2009); Kitagawa et al., PRB 84, 235108 (2011) |
