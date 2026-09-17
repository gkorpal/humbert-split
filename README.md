# Testing the folklore, detecting split surfaces, and minimum walks on isogeny graphs of abelian surfaces

This repository contains the code and data accompanying the paper. 

**ePrint:** TBA

**Authors:** Eda Kırımlı and Gaurish Korpal

## Setting

Fix a prime $p \equiv 11 \pmod{12}$, the quaternion algebra $B_p = (-1,-p \mid \mathbb{Q})$, and its maximal order $\mathcal{O} = \mathbb{Z}\langle 1, \beta_1, \beta_2, \beta_3\rangle$ with $\beta_1 = i$, $\beta_2 = \tfrac{i+j}{2}$, $\beta_3 = \tfrac{1+ij}{2}$. A principal polarization on $E \times E$ (for $E$ supersingular) is
$$\theta = \begin{bmatrix} u & \alpha \\ \bar\alpha & v \end{bmatrix}, \qquad u, v \in \mathbb{Z}_{>0},\ \alpha \in \mathcal{O},\ uv - \mathrm{nrd}(\alpha) = 1,$$
and every principally polarized superspecial abelian surface is $(E \times E, \theta)$ for some such $\theta$. Its refined Humbert invariant $q_{(E\times E,\theta)}$ is a positive definite quinary quadratic form of determinant $16p^2$. The polarization $\theta$ is irreducible, i.e. $(E \times E,\theta)$ is the Jacobian of a genus-2 curve, exactly when $q$ does not represent $1$; and $q$ primitively represents $N^2$ exactly when the surface admits an $(N,N)$-splitting.

## Scripts

All scripts are in [`/scripts/`](/scripts/) and are implemented in Julia using the [Oscar](https://www.oscar-system.org/) package and its dependency [Nemo/Hecke](https://thofma.github.io/Hecke.jl/stable/). Each script is self-documented with usage instructions in its header comments. They fall into three stages: constructing principal polarizations, computing their refined Humbert invariants, and answering representation questions about those invariants.

### Stage 1: Principal polarizations

- [`polz_all.jl`](/scripts/polz_all.jl) is `polz.jl` from humbert-degree (Kırımlı–Korpal, eprint 2025/1605), renamed for this repository; see [its README](https://github.com/gkorpal/humbert-degree) for the full description of the algorithm. It exhaustively enumerates all principal polarizations on $E \times E$ up to equivalence for small $p$: each candidate $\theta$ is converted to a rank-8 integral lattice with four auxiliary trace forms, LLL-reduced, and compared with the classes found so far by exact simultaneous Plesken–Souvignier isometry, with the enumeration bound growing until the class count reaches the class number $\mathbf{H}(p)$. Usage: `julia polz_all.jl <prime> [outdir]`. Output: `polarizations_p<p>.txt`.
- [`polz_random.jl`](/scripts/polz_random.jl) samples $N$ distinct random principal polarizations for $p$ of any size (tested to 1000 bits), by drawing $u,v \ge 1$ and solving $\mathrm{nrd}(\alpha) = uv-1$ in $\mathcal{O}$ with a RepresentInteger-style norm-equation solver. The unimodular Hermitian matrices it samples from are characterized by Ibukiyama–Katsura–Oort, *Supersingular curves of genus two and class numbers*, Compositio Math. 57 (1986); the norm-equation solver follows Kohel–Lauter–Petit–Tignol (KLPT, LMS J. Comput. Math. 2014) and De Feo–Kohel–Leroux–Petit–Wesolowski (SQIsign, ASIACRYPT 2020); and the sampler `random_polarization` is a direct port of `RandomPolarisation` from Castryck–Decru–Kutas–Laval–Petit–Ti, *KLPT2: Algebraic Pathfinding in Dimension Two and Applications*, CRYPTO 2025, [eprint 2025/372](https://eprint.iacr.org/2025/372) ([github.com/KLPT2/KLPT2](https://github.com/KLPT2/KLPT2)). New relative to KLPT2: mod-4 parity-restricted sampling, a prime-only Cornacchia fast path, randomized rather than deterministic sampling, and bounds chosen from the bit length of $p$. Usage: `julia --threads auto polz_random.jl <p> <N> [--sbound S] [--outfile PATH] [--max-tries K] [--seed S] [--allow-factor]`, where `<p>` may be an expression such as `"2^721*3^176-1"` (`--seed` reproduces exactly only with `-t 1`). Output: `polz_<p>_<N>.txt` or `polz_<bits>bit_<digest>_<N>.txt`.

### Stage 2: Refined Humbert invariants

- [`forms.jl`](/scripts/forms.jl) computes $q_{(E\times E,\theta)}$ for each polarization and sorts the results into isometry classes in three stages: an exact hash on the Gram matrix catches identical forms, an invariant key (local genus at 2 and $p$, theta-series initial coefficients, minimum, kissing number) buckets the rest, and exact Hecke isometry testing is run only between forms sharing a key. It keeps every class, including those that represent 1. Usage: `julia forms.jl [<p>] [--N K] [--tmax T] [--dir D] [--no-verify]` (with no `<p>` it sweeps $743{:}12{:}1620$; with `--N K` it reads `polz_<p>_<K>.txt`, otherwise `polarizations_p<p>.txt`). Output: `RHI_<p>.txt`.
- [`rhi.jl`](/scripts/rhi.jl) performs the same construction for primes of any size, but keeps only forms of minimum $>1$, i.e. Jacobians, using anchored theta-series bucketing followed by exact isometry testing. Usage: `julia rhi.jl <p> <N>`, which reads the `polz_*_<N>.txt` file for `p`. Output: `RHI2_<p>_<N>.txt` or `RHI2_<bits>bit_<digest>_<N>.txt`.
- [`aut.jl`](/scripts/aut.jl) is the small-$p$ counterpart of `rhi.jl` run on the exhaustive `polz_all.jl` lists: for each class it records $\theta$ and $q$ and the number of representations of 4 by $q$, and at the end tabulates that distribution alongside the Ibukiyama–Katsura–Oort (Theorem 3.3) counts of superspecial genus-2 curves by $\mathrm{Aut}(C)$; its header records $\mathbf{H}(p)$ and $\mathbf{H}(p) - h(h+1)/2$, the number of Jacobians. It is driven by uncommenting the loop at the foot of the file. Output: `RHI2_<p>.txt`.
- [`genus.jl`](/scripts/genus.jl) computes the genus of the canonical form $q_5 = t_0^2 + 4t_1^2 + 4t_1t_3 + 4t_2^2 + 4t_2t_4 + (p+1)t_3^2 + (p+1)t_4^2$ (the refined Humbert invariant of the identity polarization); every RHI lies in this genus, and the script reports the number of classes in it and how many represent 1. It is driven by `include`-ing the file and calling `Genus5(p)`. Output: `Gen5_<p>.txt`.

### Stage 3: Representation questions on the RHIs

Each script here reads an `RHI_*`/`RHI2_*` file produced in Stage 2.

- [`rm.jl`](/scripts/rm.jl) lists, for each $q$, the square-free $n \le D$ it primitively represents (real multiplication data). Usage: `julia rm.jl <p> --D <D> [--dir <dir>] [--max-nodes <n>]`. Output: `reps_<p>.txt`.
- [`prim.jl`](/scripts/prim.jl) decides, for a fixed prime $\ell$ and each $k \le k_{\max} = \lceil \log p \rceil$ (natural log), whether $\ell^{2k}$ is primitively represented by each $q$, via a local sieve at 2 and bad primes followed by exhaustive enumeration, and records a witness vector when it exists — equivalently, whether an $(\ell^k,\ell^k)$-splitting exists. Usage: `julia --threads auto prim.jl <p> --ell <ell> [--dir <dir>] [--kmax <kmax>] [--timeout <secs>] [--mem-gb <GB>]`. Output: `RHI_prim_p<p>_ell<ell>_k<kmax>.txt`.
- [`kzero.jl`](/scripts/kzero.jl) computes $k_0(q) = \min\{k \ge 1 : \ell^{2k}$ is primitively represented$\}$ by direct shell enumeration; runs are batchable and restartable via `--batch j --of N`, and a separate `--report` pass aggregates the batch CSVs into a distribution to compare against the Siegel-mass heuristic $k_0 \approx \log p / (3 \log \ell)$. Usage: `julia --startup-file=no kzero.jl <p> --ell <ell> [--batch j --of N ...]` and `julia --startup-file=no kzero.jl --report <csv>...`. Output: CSV batches plus an aggregated report.
- [`simon.jl`](/scripts/simon.jl) finds some primitive $x$ with $q(x) = N^2$ by locating an isotropic vector of $2A \perp \langle -2 \rangle$ via Simon's algorithm (as implemented in Hecke); the $N$ it finds need not be minimal. Usage: `julia simon.jl <p> [--dir <dir>] [--first <i>] [--last <j>] [--trials <n>] [--steps <n>] [--coeff-bound <C>] [--seed <s>] [--no-fallback] [--no-stop-at-one]`. Output: `simon_p<prefix>_t<trials>_s<seed>.txt`.

## Data

All data files are plain text (or CSV, for `kzero_50bit_10k/`). Directories come in three regimes, named by suffix: `*_small_all` (exhaustive enumeration, the 14 primes $p \equiv 11 \pmod{12}$ with $11 \le p \le 251$), `*_small_100k` (56 primes with $227 \le p \le 1619$, 100 000 random polarizations each), and `*_big_10k`/`*_big_100k` (the 20 primes of 50, 100, …, 1000 bits, with $10^4$ or $10^5$ random polarizations each). Directories are listed below in pipeline order.

- [`/data/polz_small_all/`](/data/polz_small_all/) — output of `polz_all.jl`: a `p = …` header followed by one `[k] u v w x y z` line per class, where $\alpha = w + x\beta_1 + y\beta_2 + z\beta_3$. The $p = 227, 239, 251$ files are in humbert-degree's older output format and are a few classes short of $\mathbf{H}(p)$ (4708/4712, 5458/5460, 6275/6281 respectively), as humbert-degree's README records for these time-limited primes. By the results of eprint 2025/1605 the missing polarizations are all reducible ones (products $E_1 \times E_2$), so every irreducible polarization, i.e. every Jacobian, is present; indeed the Jacobian counts in [`/data/aut_small_all/`](/data/aut_small_all/) for these primes equal $\mathbf{H}(p) - h(h+1)/2$ exactly.
- [`/data/polz_small_100k/`](/data/polz_small_100k/), [`/data/polz_big_10k/`](/data/polz_big_10k/), [`/data/polz_big_100k/`](/data/polz_big_100k/) — output of `polz_random.jl`. The large primes are the $p = 2^a 3^b - 1$ (two of them $5 \cdot 2^a 3^b - 1$, at 300 and 600 bits) from Table 6 of Corte-Real Santos, Costello and Frengley, *Efficient algorithms for the detection of $(N,N)$-splittings and endomorphisms*, [eprint 2025/147](https://eprint.iacr.org/2025/147) — one prime for each of 50, 100, 150, …, 1000 bits, all $\equiv 11 \pmod{12}$, e.g. $p = 2^{11}\cdot 3^{24} - 1$ at 50 bits and $p = 2^{721}\cdot 3^{176} - 1$ at 1000 bits. Filenames use `<bits>bit_<digest>`, where `<digest>` is the low 32 bits of Julia's `hash(string(p))` in hex (e.g. `50bit_1a36af10`, `1000bit_faa1c19f`). The raw files in `polz_big_100k/` run 96–317 MB, over GitHub's 100 MB per-file limit, so each is stored as line-aligned `.partNN` pieces; run `./merge.sh` (or `./merge.sh 1000bit` for a single prime) to `cat` them back together and check `SHA256SUMS` (about 3.8 GB reassembled). `.gitattributes` keeps the pieces LF-only for Windows clones.
- [`/data/forms_small_all/`](/data/forms_small_all/), [`/data/forms_small_100k/`](/data/forms_small_100k/) — output of `forms.jl`: `RHI_<p>.txt` files listing one block per isometry class, headed `Type k`, with `θ = [...]` (small $p$ only) and `q(A,θ) = c11*x1^2 + c12*x1*x2 + …` (cross coefficients are $2A_{ij}$), ending with class counts. Includes classes that represent 1.
- [`/data/aut_small_all/`](/data/aut_small_all/) — output of `aut.jl`: `RHI2_<p>.txt` files in the same block format, restricted to Jacobians, with the representation-of-4 and $\mathrm{Aut}(C)$ counts described above.
- [`/data/rhi_big_10k/`](/data/rhi_big_10k/), [`/data/rhi_big_100k/`](/data/rhi_big_100k/) — output of `rhi.jl`: `RHI2_<bits>bit_…` files, Jacobians only. As with `polz_big_100k/`, the 11 files from 500 bits up in `rhi_big_100k/` are split into `.partNN` pieces; use the same `merge.sh`/`SHA256SUMS` procedure.
- [`/data/genus_small/`](/data/genus_small/) — output of `genus.jl`: `Gen5_<p>.txt` files, one `q = …` line per genus class plus totals, for the 20 primes $11 \le p \le 419$.
- [`/data/rm_small_all/`](/data/rm_small_all/), [`/data/rm_small_100k/`](/data/rm_small_100k/) — output of `rm.jl` with $D = 100$: `reps_<p>.txt` files listing, per Type, the square-free $n \le D$ primitively represented.
- [`/data/prim_small_all_ell2/`](/data/prim_small_all_ell2/), [`/data/prim_small_all_ell3/`](/data/prim_small_all_ell3/) — output of `prim.jl` for $\ell = 2$ and $\ell = 3$ with $k \le \lceil \log p \rceil$: `RHI_prim_*.txt` files listing, per form and $k$, a witness vector or `FAIL`, ending with a `Summary: min_k=…` line.
- [`/data/kzero_50bit_10k/`](/data/kzero_50bit_10k/) — output of `kzero.jl` for $\ell = 2$ on the 50-bit prime, covering 10 000 Jacobian classes from `rhi_big_10k/`: 50 interleaved batch CSVs (`_batch<j>of50.csv`, columns `form,status,in_genus,lambda1,aut,kmin,k0,witness,secs`) plus the aggregated report `kzero_p50bit_1a36af10_ell2_main.txt`.
- [`/data/simon_big_100k/`](/data/simon_big_100k/) — output of `simon.jl` for all 20 large primes on the 100 000-class `RHI2` files, with no random re-presentations (`t0`) and seed `20260904`: `simon_*.txt` files listing `Type k: N` per class and closing with the min/max $N$ found.

## Running the scripts

1. Install Julia $\ge 1.12$ and Oscar (`using Pkg; Pkg.add("Oscar")`).
2. Reassemble split data before use: `cd data/polz_big_100k && ./merge.sh [1000bit]` (likewise in `data/rhi_big_100k`); reassembly needs about 3.8 GB, and `.gitattributes` keeps the pieces LF-only so this works the same on a Windows clone.
3. One example per stage: `julia polz_all.jl 23`, `julia -t auto polz_random.jl "2^11*3^24-1" 100 --seed 1`, `julia rhi.jl "2^11*3^24-1" 100`, `julia prim.jl 23 --ell 2`, `julia simon.jl "2^11*3^24-1"`. Each script looks for its input file (e.g. `polz_*.txt`, `RHI2_*.txt`) in the working directory or the directory given by `--dir`, so run it from, or point it at, the relevant `data/` subdirectory. Reproducing a random sample exactly needs `-t 1` together with `--seed`.

## System Requirements

`polz_all.jl`, `polz_random.jl`, `rhi.jl`, and `kzero.jl` were run on the HPC facilities of the *Advanced Computing Research Centre, University of Bristol*. All other scripts were run on a local machine with the following specifications.

- **CPU:** 12th Gen Intel i7-12800HX (24 cores)
- **Memory:** 15839 MiB
- **OS:** Ubuntu 22.04.5 LTS on Windows 11 x86\_64 (WSL)
- **Julia:** Version 1.12.7
- **Oscar:** Version 1.8.2

## License

MIT (see [`LICENSE`](/LICENSE)).
