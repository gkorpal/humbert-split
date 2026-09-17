# Testing the folklore, detecting split surfaces, and minimum walks on isogeny graphs of abelian surfaces

This repository contains the code and data accompanying the paper.

**ePrint:** TBA

**Authors:** Eda Kırımlı and Gaurish Korpal

## Scripts

All scripts are in [`/scripts/`](/scripts/) and are implemented in Julia using the [Oscar](https://www.oscar-system.org/) package and its dependency [Nemo/Hecke](https://thofma.github.io/Hecke.jl/stable/). Each script is self-documented: its header comments explain the mathematics, the implementation and the usage.

Throughout, $p \equiv 11 \pmod{12}$, $B_p = (-1,-p \mid \mathbb{Q})$, and $\mathcal{O} = \mathbb{Z}\langle 1, \beta_1, \beta_2, \beta_3\rangle$ with $\beta_1 = i$, $\beta_2 = \tfrac{i+j}{2}$, $\beta_3 = \tfrac{1+ij}{2}$. A principal polarization on $E \times E$ is

$$\theta = \begin{bmatrix} u & \alpha \\ \bar\alpha & v \end{bmatrix}, \qquad u, v \in \mathbb{Z}_{>0}, \quad \alpha \in \mathcal{O}, \quad uv - \mathrm{nrd}(\alpha) = 1,$$

and every principally polarized superspecial abelian surface arises this way. Its refined Humbert invariant $q_{(E \times E,\theta)}$ is a positive definite quinary form of determinant $16p^2$. It represents $1$ exactly when $\theta$ is reducible, and it primitively represents $N^2$ exactly when $(E \times E, \theta)$ admits an $(N,N)$-splitting.

**Pipeline 1: Principal polarizations**

- [`polz_all.jl`](/scripts/polz_all.jl) is [`polz.jl`](https://github.com/gkorpal/humbert-degree/blob/main/scripts/polz.jl) of [humbert-degree](https://github.com/gkorpal/humbert-degree) ([eprint 2025/1605](https://eprint.iacr.org/2025/1605)): it enumerates all principal polarizations on $E \times E$ up to equivalence for small $p$, stopping when their number reaches the class number $\mathbf{H}(p)$.

- [`polz_random.jl`](/scripts/polz_random.jl) samples $N$ distinct random principal polarizations for $p$ of any size: draw $u, v \ge 1$ and solve $\mathrm{nrd}(\alpha) = uv - 1$ in $\mathcal{O}$. The polarization matrices are characterized by [Ibukiyama–Katsura–Oort](https://www.numdam.org/item/CM_1986__57_2_127_0/), the norm equation is solved as in [KLPT](https://eprint.iacr.org/2014/505) and [SQIsign](https://eprint.iacr.org/2020/1240), and the sampler is a port of `RandomPolarisation` from [KLPT2](https://github.com/KLPT2/KLPT2) ([eprint 2025/372](https://eprint.iacr.org/2025/372)), with parity-restricted sampling and size-dependent search bounds added.

**Pipeline 2: Refined Humbert invariants**

- [`forms.jl`](/scripts/forms.jl) computes $q_{(E \times E,\theta)}$ for each polarization and sorts the results into isometry classes (isometry invariants first, exact Plesken–Souvignier isometry only within a bucket). Every class is kept.

- [`rhi.jl`](/scripts/rhi.jl) is `forms.jl` for $p$ of any size, but keeps only the forms of irreducible polarizations, i.e. those of minimum $> 1$.

- [`aut.jl`](/scripts/aut.jl) is the small-$p$ version of `rhi.jl` for the exhaustive lists; it also records the number of representations of $4$ by each form and the [Ibukiyama–Katsura–Oort](https://www.numdam.org/item/CM_1986__57_2_127_0/) counts of superspecial genus-2 curves by automorphism group.

- [`genus.jl`](/scripts/genus.jl) computes the genus of $q_5 = t_0^2 + 4t_1^2 + 4t_1t_3 + 4t_2^2 + 4t_2t_4 + (p+1)t_3^2 + (p+1)t_4^2$, the invariant of the identity polarization, which contains every refined Humbert invariant.

**Pipeline 3: Representations**

- [`rm.jl`](/scripts/rm.jl) lists the square-free $n \le D$ primitively represented by each form.

- [`prim.jl`](/scripts/prim.jl) is a local–global experiment for small $p$: for a prime $\ell$ and each $k \le \lceil \log p \rceil$ it decides whether $\ell^{2k}$ is primitively represented, i.e. whether an $(\ell^k, \ell^k)$-splitting exists, first by local conditions at $2$ and the bad primes and then, for the pairs that pass, by exhaustive enumeration, so that the gap between the two is visible.

- [`kzero.jl`](/scripts/kzero.jl) computes $k_0 = \min\{k \ge 1 : \ell^{2k} \text{ is primitively represented}\}$ for each form by global search alone, enumerating the vectors of norm $\ell^{2k}$ with the [Fincke–Pohst algorithm](https://doi.org/10.1090/S0025-5718-1985-0777278-8) level by level, and compares the distribution of $k_0$ with the prediction $k_0 \approx \log p / (3 \log \ell)$ from Siegel's mass formula.

- [`simon.jl`](/scripts/simon.jl) finds a primitive $x$ with $q(x) = N^2$ from an isotropic vector of $2A \perp \langle -2 \rangle$ by [Simon's algorithm](https://doi.org/10.1090/S0025-5718-05-01729-1); the $N$ found is not necessarily minimal.

## Data

All data is in [`/data/`](/data/) as human-readable `.txt` files (CSV for `kzero_50bit_10k`). The suffix of each directory names the primes it covers: `_small_all` are the 14 primes $11 \le p \le 251$ with all polarizations; `_small_100k` are 56 primes $227 \le p \le 1619$ with $10^5$ random polarizations each; `_big_10k` and `_big_100k` are 20 primes of $50, 100, \ldots, 1000$ bits with $10^4$ and $10^5$ random polarizations each. The large primes are the $2^a 3^b - 1$ primes of Table 6 of [Corte-Real Santos–Costello–Frengley](https://eprint.iacr.org/2025/147) (e.g. $2^{721} \cdot 3^{176} - 1$ at 1000 bits); their file names carry the bit length and a hash of $p$ in place of $p$.

- [`/data/polz_small_all/`](/data/polz_small_all/), [`/data/polz_small_100k/`](/data/polz_small_100k/), [`/data/polz_big_10k/`](/data/polz_big_10k/), [`/data/polz_big_100k/`](/data/polz_big_100k/) contain the polarizations produced by `polz_all.jl` and `polz_random.jl`, one line $(u, v, w, x, y, z)$ each with $\alpha = w + x\beta_1 + y\beta_2 + z\beta_3$. For $p = 227, 239, 251$ the exhaustive lists are those of humbert-degree, which miss a few reducible polarizations but contain every irreducible one.

- [`/data/forms_small_all/`](/data/forms_small_all/) and [`/data/forms_small_100k/`](/data/forms_small_100k/) contain the isometry classes of refined Humbert invariants produced by `forms.jl`; [`/data/aut_small_all/`](/data/aut_small_all/), [`/data/rhi_big_10k/`](/data/rhi_big_10k/) and [`/data/rhi_big_100k/`](/data/rhi_big_100k/) contain those of Jacobians only, produced by `aut.jl` and `rhi.jl`.

- [`/data/genus_small/`](/data/genus_small/) contains the genus of $q_5$ for the 20 primes $11 \le p \le 419$.

- [`/data/rm_small_all/`](/data/rm_small_all/) and [`/data/rm_small_100k/`](/data/rm_small_100k/) contain the `rm.jl` output for $D = 100$.

- [`/data/prim_small_all_ell2/`](/data/prim_small_all_ell2/) and [`/data/prim_small_all_ell3/`](/data/prim_small_all_ell3/) contain the `prim.jl` output for $\ell = 2$ and $\ell = 3$.

- [`/data/kzero_50bit_10k/`](/data/kzero_50bit_10k/) contains the `kzero.jl` output for $\ell = 2$ on the $10^4$ Jacobian classes of the 50-bit prime, as 50 batch files and one report.

- [`/data/simon_big_100k/`](/data/simon_big_100k/) contains the `simon.jl` output for the $10^5$ Jacobian classes of each large prime.

Files over GitHub's 100 MB limit (all of `polz_big_100k` and the 500-bit and larger files of `rhi_big_100k`) are stored as `.partNN` pieces; `merge.sh` in each of those directories reassembles and checksums them.

## System Requirements

`polz_all.jl`, `polz_random.jl`, `rhi.jl` and `kzero.jl` were run on the HPC facilities of the *Advanced Computing Research Centre, University of Bristol*. All other scripts were run on a local machine with the following specifications.

- **CPU:** 12th Gen Intel i7-12800HX (24 cores)
- **Memory:** 15839 MiB
- **OS:** Ubuntu 22.04.5 LTS on Windows 11 x86\_64 (WSL)
- **Julia:** Version 1.12.7
- **Oscar:** Version 1.8.2
