# prim.jl — primitive representation of ℓ^{2k} by a quinary form
#
# Fix a positive-definite integral quadratic form in five variables,
#
#     Q_A(x) = xᵀ A x,   x ∈ ℤ⁵,   A ∈ Mat₅(ℤ) symmetric, A > 0,
#
# and a prime ℓ. For each level k = 1, 2, … this file decides whether the
# integer n_k = ℓ^{2k} is *primitively* represented by Q_A, i.e. whether
# there is x ∈ ℤ⁵ with Q_A(x) = ℓ^{2k} and gcd(x₁, …, x₅) = 1, and when it
# is, exhibits such an x.
#
# ─── Notation ────────────────────────────────────────────────────────────────
#
#   A, Q_A, L     Gram matrix, the form x ↦ xᵀAx, and the lattice L = (ℤ⁵, Q_A).
#   p             the prime indexing the input file RHI2_p.txt; the forms
#                 in it have determinant 16p². p may be very large (hundreds
#                 of bits), so p and the entries of A are kept as BigInt.
#   ℓ             prime base of the targets; written `ell` in the code.
#   k, level k    the exponent; "level k" means the target n_k = ℓ^{2k}.
#   witness       a vector x ∈ ℤ⁵ with Q_A(x) = ℓ^{2k} and gcd(x) = 1.
#   bad prime     an odd prime q dividing det A; the form is q-adically
#                 unimodular at every other odd prime.
#   depth d       the 2-adic test is carried out modulo 2^d.
#   Jordan scale  the exponent s of a 2^s-modular component in the 2-adic
#                 Jordan decomposition L ⊗ ℤ₂ = ⊥ᵢ Lᵢ, Lᵢ of scale 2^{sᵢ}.
#   primitive residue set
#                 the set { Q_A(x) mod 2^d : x ∈ (ℤ/2^d)⁵, some xᵢ odd }.
#
# ─── Local–global strategy ───────────────────────────────────────────────────
#
#   Primitive representability over ℤ implies primitive representability over
#   ℤ_ν for every prime ν; the converse fails in general (the gap between
#   genus, spinor genus and class), and that gap is exactly what these
#   computations probe. The local conditions are therefore used only as a
#   one-sided sieve:
#
#     (i)  if Q_A fails to primitively represent n_k over some ℤ_ν, then n_k
#          is certainly not primitively represented over ℤ;
#     (ii) otherwise nothing is settled, and an exhaustive enumeration of the
#          vectors of norm n_k in L decides the question.
#
#   Every :NO_LOCAL_* answer below is therefore a proof; :LOCAL_PASS means
#   only "not yet excluded".
#
#   The primes ν that can obstruct (i) are ν = 2, the bad primes q | det A,
#   and ν = ℓ itself (where n_k is highly divisible). At an odd prime not
#   dividing 2·ℓ·det A the form is unimodular of rank 5, hence isotropic, and
#   primitively represents every unit.
#
#   ν = 2.  Q_A mod 2^d depends only on A mod 2^d, so the set of residues
#   mod 2^d attained at primitive vectors is a finite object, computed once
#   from the 2-adic Jordan decomposition and stored as a table of 2^d bits.
#   Deciding level k is then a single lookup of ℓ^{2k} mod 2^d. For ℓ = 2 the
#   residue 4^k mod 2^d is 0 once 2k ≥ d, and the table says nothing more;
#   a second table for the form with its minimal Jordan scale divided out
#   covers that regime (see "Scaled 2-adic check").
#
#   ν = q odd, q | det A.  Two tests, both answered by Oscar's genus
#   symbols. The level-independent one asks whether the genus of L at q
#   represents the rank-one lattice ⟨1⟩. The level-dependent one asks
#   whether it represents ⟨ℓ^{2k} mod q²⟩, which sees the square class of
#   the target and not only its unit-ness.
#
#   Global.  Once every local test passes, Hecke enumerates the vectors of
#   norm exactly ℓ^{2k} in an LLL-reduced basis of L; the first primitive one
#   found is returned as a witness, and if none exists the answer is a
#   definite no.
#
# ─── Algorithm, in order of increasing cost ──────────────────────────────────
#
#   build_problem(A; ell)   All level-independent work, once per form:
#                           Jordan decomposition and 2-adic residue tables,
#                           genus symbols at the bad primes, LLL reduction.
#   decide(P, k)            1. plain 2-adic table lookup;
#                           2. scaled 2-adic table lookup (large k only);
#                           3. level-independent bad-prime genus test;
#                           4. level-dependent bad-prime genus test, one
#                              genus call per bad prime, remembered per
#                              residue class ℓ^{2k} mod q²;
#                           5. exhaustive enumeration in the reduced lattice.
#   scan(P, ks)             [decide(P, k) for k in ks].
#   decide_local_only,      Steps 1–4 and step 5 separately. process_rhi2
#   decide_global_only      runs all local tests for every (form, k) first,
#                           in parallel, and then the enumeration only for
#                           the pairs that survived, one at a time.
#   process_rhi2(p; ell)    Reads RHI2_p.txt, runs the above for every form
#                           and every k ≤ kmax, writes RHI_prim_p…_ell…_k….txt.
#
# ─── Computing vocabulary used below ─────────────────────────────────────────
#
#   Oscar / Hecke / FLINT   Oscar is the computer-algebra system used;
#                           Hecke is its lattice and genus library; FLINT is
#                           the C library underneath doing the integer and
#                           matrix arithmetic.
#   BitVector               a table of bits; here always indexed by a residue
#                           r mod 2^d, entry r+1 being 1 iff r is attained.
#   thread                  one of several strands of computation the same
#                           Julia process runs at once (Phase 1 of
#                           process_rhi2 uses one per available core).
#   worker process          a second Julia process, started on demand, on
#                           which one enumeration at a time is run so that
#                           it can be killed if it exceeds the time limit.
#                           Only used when a --timeout is given.
#   node                    one lattice vector visited by the enumeration.
#
# ─── Launch ──────────────────────────────────────────────────────────────────
#
#   julia --threads auto prim.jl <p> --ell <ell> [--dir <dir>] [--kmax <kmax>] [--timeout <secs>] [--mem-gb <GB>]
#
#   <p> may be a plain integer or an expression over + - * ^ ( ) evaluated
#   in BigInt, e.g. "2^721*3^176-1". --threads auto is a flag of julia
#   itself (it must precede prim.jl) and lets Phase 1 use every core. The
#   full list of flags is at the end of the file, and from a Julia session
#
#       include("prim.jl"); process_rhi2(11; ell=2, kmax=20)
#
#   does the same as the command line.

using Oscar
using LinearAlgebra
using Distributed

const ZZ = Oscar.ZZ
const QQ = Oscar.QQ

# ══════════════════════════════════════════════════════════════════════════════
# Options
# ══════════════════════════════════════════════════════════════════════════════

"""
    Options(; ...)

Tuning parameters for one `Problem`.

| Field                       | Default        | Meaning                                                            |
|:----------------------------|:---------------|:--------------------------------------------------------------------|
| `two_adic_depth`            | `6`            | The 2-adic test works modulo `2^two_adic_depth`.                   |
| `two_adic_scaled_depth`     | `8`            | Depth of the second 2-adic table, for the form with its minimal Jordan scale divided out. |
| `odd_local_k_dependent`     | `true`         | Also test, at each bad prime q, whether the genus represents `⟨ℓ^{2k} mod q²⟩`, not only `⟨1⟩`. |
| `exact_cutoff`              | `typemax(Int)` | Skip the enumeration for `k > exact_cutoff`; the answer is then UNKNOWN. |
| `assume_after_exact_cutoff` | `false`        | Report YES instead of UNKNOWN above the cutoff. Unsound; for exploration only. |
| `max_nodes`                 | `300_000`      | Give up the enumeration after this many lattice vectors.           |
| `lll_delta`, `lll_eta`      | `0.99`, `0.501`| Lovász δ and size-reduction η for the LLL reduction.               |
| `global_timeout_secs`       | `nothing`      | Time limit for each enumeration, in seconds (see below).           |
| `worker_memory_limit_bytes` | `nothing`      | Memory cap for the worker process that runs the enumeration.       |

### Depth of the 2-adic test

The residue tables are built by enumerating vectors of each 2-adic Jordan
component modulo `2^d`; a component of rank r costs `2^{dr}` evaluations.
Nothing bounds the rank of a component below 5 (a 2-adically unimodular
form is a single rank-5 component), so at `d = 6` the worst case is
`64^5 ≈ 10^9`, feasible once per form, while `d = 8` would already be
`2^40`. Keep `two_adic_depth ≤ 6`.

The scaled table is built only when L has no unimodular 2-adic component
(`min_jordan_scale > 0`). For a form of determinant `16p²` that never
happens — five components of scale ≥ 2 would give `v₂(det A) ≥ 5 > 4` — so
for the forms read from the RHI2 files `two_adic_scaled_depth` is inert.
Where it is used, the rank warning above applies at that depth too.

For odd ℓ the target `ℓ^{2k} = (ℓ^k)²` is an odd square, hence `≡ 1 (mod 8)`
for every k, so the plain 2-adic test gives the same answer at every level;
it is still a genuine sieve. For ℓ = 2 the residue `4^k mod 2^d` runs
through `1, 4, 16, …` and is `0` for `2k ≥ d`.

### Level-dependent test at the bad primes

With `odd_local_k_dependent = true`, `decide` asks at each bad prime q
whether the genus of L represents the rank-one lattice `⟨ℓ^{2k} mod q²⟩`.
The residue `ℓ^{2k} mod q²` is periodic in k with period dividing
`q(q−1)`, and the answer is remembered per residue, so the genus call is
made at most `q(q−1)` times per bad prime however many levels are scanned.

### Time and memory limits for the enumeration

The cost of the enumeration grows with the target norm `ℓ^{2k}`; there is
no overflow to worry about (all arithmetic is in FLINT's multiprecision
integers), only time and memory. With `global_timeout_secs` set, each
enumeration is run on a separate worker process which is killed and
replaced when it exceeds the limit; the result is then `:UNKNOWN_TIMEOUT`.
With `worker_memory_limit_bytes` set, the worker's address space is capped
so that an oversized enumeration fails at once with `:UNKNOWN_WORKER_ERROR`
instead of driving the machine into swap and stretching the wall time far
past the limit. See "Global search on a worker process" below.
"""
Base.@kwdef struct Options
    two_adic_depth::Int             = 6
    exact_cutoff::Int               = typemax(Int)
    assume_after_exact_cutoff::Bool = false
    max_nodes::Int                  = 300_000
    lll_delta::Float64              = 0.99
    lll_eta::Float64                = 0.501
    odd_local_k_dependent::Bool     = true
    two_adic_scaled_depth::Int      = 8
    global_timeout_secs::Union{Nothing,Float64} = nothing
    worker_memory_limit_bytes::Union{Nothing,Int} = nothing
end

# Session-wide defaults for the `timeout` and `mem_gb` keywords of
# `process_rhi2`. They are `Ref`s so that they can be changed after the file
# has been included:
#
#     include("prim.jl")
#     PRIM_DEFAULT_TIMEOUT_SECS[] = 60.0   # seconds per (form, k) enumeration
#     PRIM_DEFAULT_MEM_GB[]       = 12.0   # GiB for the worker process
#     process_rhi2(11; ell=2, kmax=20)     # uses both
#     process_rhi2(23; ell=3, kmax=15, timeout=600.0)
#
# Left at `nothing`, there is no time limit and no memory cap.
const PRIM_DEFAULT_TIMEOUT_SECS = Ref{Union{Nothing,Float64}}(nothing)
const PRIM_DEFAULT_MEM_GB       = Ref{Union{Nothing,Float64}}(nothing)

# ══════════════════════════════════════════════════════════════════════════════
# Problem
# ══════════════════════════════════════════════════════════════════════════════

"""
    Problem

Everything about one form that does not depend on the level k, computed
once by `build_problem` and read by `decide`.

Fields
──────
  ell                   The prime ℓ, as BigInt.
  A                     Gram matrix, `Matrix{BigInt}`.
  A_zz                  The same matrix as a FLINT `ZZMatrix`, used to
                        evaluate Q_A inside the enumeration.
  L                     The lattice (ℤ⁵, Q_A) as an Oscar `ZZLat`.
  bad_primes            Odd primes dividing det A.
  options               See `Options`.
  two_adic_method       Description of how `local2_bitvec` was built
                        (Jordan decomposition, or direct enumeration if
                        Oscar could not decompose the form).
  jordan_scales         The 2-adic Jordan scales of L.
  local2_bitvec         Length `2^two_adic_depth`; entry r+1 is `true` iff
                        the residue r is attained by Q_A at a primitive
                        vector modulo `2^two_adic_depth`.
  scaled_local2_bitvec  The same table for the form Q_A / 4^s with
                        s = `min_jordan_scale`, at depth
                        `two_adic_scaled_depth`; `nothing` when s = 0,
                        since then it would coincide with `local2_bitvec`.
  min_jordan_scale      The minimal 2-adic Jordan scale s of L.
  odd_local_ok          `true` iff the genus of L represents ⟨1⟩ at every
                        bad prime.
  odd_local_genus       For each bad prime q: the genus symbol of L at q
                        and a table of the answers already obtained for
                        `⟨ℓ^{2k} mod q²⟩`. `nothing` when the
                        level-dependent test is switched off or there are
                        no bad primes.
  U                     Unimodular matrix from the LLL reduction, so that
                        Uᵀ A U is the reduced Gram matrix. `nothing` when
                        `exact_cutoff < 1` (no enumeration will ever run).
  U_zz                  U as a FLINT `ZZMatrix`, for the change of
                        coordinates x = U z.
  Lred                  The lattice with Gram matrix Uᵀ A U, on which Hecke
                        enumerates.
"""
struct Problem
    ell::BigInt
    A::Matrix{BigInt}
    A_zz::ZZMatrix
    L::ZZLat
    bad_primes::Vector{BigInt}
    options::Options
    two_adic_method::String
    jordan_scales::Vector{Int}
    local2_bitvec::BitVector
    odd_local_ok::Bool
    U::Union{Nothing,Matrix{BigInt}}
    U_zz::Union{Nothing,ZZMatrix}
    Lred::Union{Nothing,ZZLat}
    scaled_local2_bitvec::Union{Nothing,BitVector}
    min_jordan_scale::Int
    odd_local_genus::Union{Nothing,Vector{Any}}
end

# ══════════════════════════════════════════════════════════════════════════════
# Core arithmetic
# ══════════════════════════════════════════════════════════════════════════════

# ℓ^{2k}, exact for every k.
pow_ell2(ell::Integer, k::Integer) = BigInt(ell)^(2 * Int(k))

"""
    natural_log_bigint(n::BigInt) -> Float64

`log(n)` for a positive BigInt of any size: `log(n) = log(m) + s·log 2`
where `m = n >> s` has at most 53 bits, so the only conversion to Float64
is exact.
"""
function natural_log_bigint(n::BigInt)::Float64
    n > 0 || error("natural_log_bigint requires a positive argument, got $n")
    bits     = ndigits(n, base = 2)
    shift    = max(bits - 53, 0)
    mantissa = Float64(n >> shift)
    return log(mantissa) + shift * log(2.0)
end

"""
    qvalue(A_zz, xm) -> ZZRingElem
    qvalue(A_zz, x)  -> BigInt
    qvalue(A,    x)  -> BigInt

Evaluate Q_A(x) = xᵀ A x. The first form takes FLINT matrices and is the
one used inside the enumeration: one matrix–vector product and one dot
product, both in FLINT. `qvalue!` is the same with a caller-supplied
scratch vector for A·x, so that the inner loop creates no temporaries.
"""
function qvalue(A_zz::ZZMatrix, xm::ZZMatrix)
    Ax = A_zz * xm
    return dot(xm, Ax)
end

@inline function qvalue!(scratch::ZZMatrix, A_zz::ZZMatrix, xm::ZZMatrix)
    Oscar.mul!(scratch, A_zz, xm)
    return dot(xm, scratch)
end

function qvalue(A_zz::ZZMatrix, x::AbstractVector{<:Integer})
    n  = length(x)
    xm = matrix(ZZ, n, 1, ZZRingElem[ZZ(v) for v in x])
    return BigInt(qvalue(A_zz, xm))
end

function qvalue(A::AbstractMatrix{<:Integer}, x::AbstractVector{<:Integer})
    size(A, 1) == size(A, 2) == length(x) || error("dimension mismatch")
    return qvalue(matrix(ZZ, A), x)
end

"""
    primitive_gcd(x) -> BigInt

`gcd(x₁, …, xₙ)`, stopping as soon as the running gcd is 1.
"""
function primitive_gcd(x::AbstractVector{<:Integer})
    isempty(x) && return BigInt(0)
    g = ZZ(0)
    @inbounds for a in x
        g = gcd(g, ZZ(abs(a)))
        isone(g) && return BigInt(1)
    end
    return BigInt(g)
end

# Is gcd(x) = 1?  For vectors of FLINT integers as Hecke's enumeration
# produces them; stops at the first 1. An empty vector is not primitive.
function _isprimitive_zz(x::AbstractVector{ZZRingElem})
    g = ZZ(0)
    @inbounds for a in x
        g = gcd(g, a)
        isone(g) && return true
    end
    return false
end

"""
    verify_witness(A, x, k, ell) -> NamedTuple

Independent check that `x` is a primitive representation of `ell^{2k}` by
Q_A: recomputes Q_A(x) from the plain matrix `A` and gcd(x), using nothing
stored in a `Problem`.
"""
function verify_witness(A::AbstractMatrix{<:Integer}, x, k::Integer, ell::Integer)
    target = pow_ell2(ell, k)
    v      = qvalue(A, x)
    pg     = primitive_gcd(x)
    return (value = v, target = target, primitive_gcd = pg,
            ok    = v == target && pg == 1)
end

# ══════════════════════════════════════════════════════════════════════════════
# Lattice construction
# ══════════════════════════════════════════════════════════════════════════════

# Check that A is a symmetric positive-definite 5×5 integer matrix and build
# the lattice (ℤ⁵, Q_A). Positive-definiteness is what makes the set of
# vectors of a given norm finite.
function _check_quinary_form(Ain::AbstractMatrix{<:Integer})
    A = Matrix{BigInt}(Ain)
    size(A) == (5, 5) || error("expected a 5×5 Gram matrix")
    A == A'           || error("Gram matrix must be symmetric")
    L = integer_lattice(; gram = matrix(ZZ, A))
    is_positive_definite(L) || error("Gram matrix must be positive definite")
    return A, L
end

# The odd primes dividing det A. At every other odd prime L is unimodular of
# rank 5 and represents every unit primitively, so only these can obstruct.
function _odd_primes_from_det(L::ZZLat)
    # `det(gram_matrix(L))` is a rational number in Oscar even for an
    # integral lattice; take the numerator before factoring.
    raw = det(gram_matrix(L))
    d   = ZZ(abs(numerator(QQ(raw))))
    ps  = BigInt[]
    for (p, _) in factor(d)
        q = BigInt(p)
        q > 2 && push!(ps, q)
    end
    return ps
end

# ══════════════════════════════════════════════════════════════════════════════
# 2-adic local check
# ══════════════════════════════════════════════════════════════════════════════
#
# Question: is ℓ^{2k} primitively represented by Q_A over ℤ₂?
#
# Q_A mod 2^d depends only on A mod 2^d, so the set R_d of residues mod 2^d
# attained by Q_A at primitive vectors x ∈ (ℤ/2^d)⁵ is finite and is computed
# once; level k then passes iff ℓ^{2k} mod 2^d ∈ R_d. This is a necessary
# condition for 2-adic primitive representability. For the forms of
# determinant 16p² with p ≡ 3 (mod 4), whose 2-adic value set at primitive
# vectors is exactly {n ≡ 0, 1 (mod 4)}, it is also sufficient for every
# d ≥ 2; in general it is a sieve, sound in the direction "fails ⇒ no".
#
# R_d is assembled from the 2-adic Jordan decomposition L ⊗ ℤ₂ = ⊥ᵢ Lᵢ. A
# vector of L ⊗ ℤ₂ is primitive iff its component in at least one Lᵢ is
# primitive, i.e. has an odd coordinate. Processing the components one at a
# time, we keep two disjoint residue sets:
#
#     prim     residues attained by vectors that are primitive in at least
#              one of the components seen so far;
#     nonprim  residues attained by vectors whose coordinates are even in
#              every component seen so far.
#
# Adjoining a component with primitive residue set Bp and all-even residue
# set Bn (both mod 2^d),
#
#     prim'    = prim ∪ (nonprim + Bp) ∪ (prim + Bp) ∪ (prim + Bn),
#     nonprim' = nonprim + Bn,
#
# where X + Y is the sumset mod 2^d. The term prim + Bn accounts for vectors
# that are primitive only because of an earlier component while all-even in
# the new one; nonprim must be convolved with every Bn, since a vector that
# is all-even in every component is imprimitive whatever else it does.
#
# The sets are stored as BitVectors of length M = 2^d, so at d = 6 each
# union is one machine-word operation.

# ℓ^{2k} mod 2^depth as an Int in [0, 2^depth). For ℓ = 2 this is 4^k = 1 << 2k
# when 2k < depth and 0 otherwise; for odd ℓ it is a modular power.
@inline function _target_residue_mod_2N(ell::Integer, k::Int, depth::Int)
    M = 1 << depth
    if ell == 2
        shift = 2 * k
        shift >= depth && return 0
        return 1 << shift
    else
        return Int(powermod(BigInt(ell), 2 * k, M))
    end
end

# Reduce a Gram-matrix entry of a Jordan component — possibly a rational
# number with odd denominator — to an Int mod M = 2^depth. Numerator and
# denominator are reduced mod M *before* narrowing to Int, since for large p
# the entries are far beyond the Int64 range.
function _rat_mod_pow2(x, M::Int)
    if x isa QQFieldElem
        num = mod(BigInt(ZZ(numerator(x))),   M)
        den = mod(BigInt(ZZ(denominator(x))), M)
        isodd(BigInt(ZZ(denominator(x)))) || error("even denominator in 2-adic Gram entry: $x")
        return Int(mod(num * invmod(den, M), M))
    else
        return Int(mod(BigInt(ZZ(x)), M))
    end
end

# The set { Q_G(x) mod M : x ∈ {0,…,M-1}^r with some xᵢ odd }, for the Gram
# matrix G of one Jordan component of rank r, as a BitVector of length
# M = 2^depth. "Some xᵢ odd" is tested as "x₁ | x₂ | … | xᵣ is odd". Q_G is
# evaluated from the upper triangle: Q_G(x) = Σᵢ Gᵢᵢxᵢ² + 2Σ_{i<j} Gᵢⱼxᵢxⱼ.
# One explicit loop nest per rank 1–5 keeps the enumeration free of
# temporaries.
function _primitive_residues_bitvec(G, depth::Int)
    M    = 1 << depth
    r    = nrows(G)
    bv   = falses(M)
    r == 0 && return bv

    Gmod = Matrix{Int}(undef, r, r)
    for i in 1:r, j in 1:r
        Gmod[i, j] = _rat_mod_pow2(G[i, j], M)
    end

    @inline function qmod(xs::NTuple{N,Int}) where N
        q = 0
        @inbounds for i in 1:N
            q += xs[i] * Gmod[i, i] * xs[i]
            for j in i+1:N
                q += 2 * xs[i] * Gmod[i, j] * xs[j]
            end
        end
        return mod(q, M)
    end

    if r == 1
        for x1 in 0:M-1
            isodd(x1) || continue
            bv[qmod((x1,)) + 1] = true
        end
    elseif r == 2
        for x1 in 0:M-1, x2 in 0:M-1
            iseven(x1 | x2) && continue
            bv[qmod((x1, x2)) + 1] = true
        end
    elseif r == 3
        for x1 in 0:M-1, x2 in 0:M-1, x3 in 0:M-1
            iseven(x1 | x2 | x3) && continue
            bv[qmod((x1, x2, x3)) + 1] = true
        end
    elseif r == 4
        for x1 in 0:M-1, x2 in 0:M-1, x3 in 0:M-1, x4 in 0:M-1
            iseven(x1 | x2 | x3 | x4) && continue
            bv[qmod((x1, x2, x3, x4)) + 1] = true
        end
    else  # r == 5 (and any larger rank, though the form is quinary)
        for x1 in 0:M-1, x2 in 0:M-1, x3 in 0:M-1, x4 in 0:M-1, x5 in 0:M-1
            iseven(x1 | x2 | x3 | x4 | x5) && continue
            bv[qmod((x1, x2, x3, x4, x5)) + 1] = true
        end
    end
    return bv
end

# The complementary set { Q_G(x) mod M : x ∈ {0,…,M-1}^r with every xᵢ even }
# for one Jordan component, same layout as `_primitive_residues_bitvec`.
# For r = 0 the empty vector has Q = 0 and counts as all-even.
function _nonprimitive_residues_bitvec(G, depth::Int)
    M    = 1 << depth
    r    = nrows(G)
    bv   = falses(M)
    if r == 0
        bv[1] = true
        return bv
    end

    Gmod = Matrix{Int}(undef, r, r)
    for i in 1:r, j in 1:r
        Gmod[i, j] = _rat_mod_pow2(G[i, j], M)
    end

    @inline function qmod(xs::NTuple{N,Int}) where N
        q = 0
        @inbounds for i in 1:N
            q += xs[i] * Gmod[i, i] * xs[i]
            for j in i+1:N
                q += 2 * xs[i] * Gmod[i, j] * xs[j]
            end
        end
        return mod(q, M)
    end

    if r == 1
        for x1 in 0:M-1
            isodd(x1) && continue
            bv[qmod((x1,)) + 1] = true
        end
    elseif r == 2
        for x1 in 0:M-1, x2 in 0:M-1
            iseven(x1 | x2) || continue
            bv[qmod((x1, x2)) + 1] = true
        end
    elseif r == 3
        for x1 in 0:M-1, x2 in 0:M-1, x3 in 0:M-1
            iseven(x1 | x2 | x3) || continue
            bv[qmod((x1, x2, x3)) + 1] = true
        end
    elseif r == 4
        for x1 in 0:M-1, x2 in 0:M-1, x3 in 0:M-1, x4 in 0:M-1
            iseven(x1 | x2 | x3 | x4) || continue
            bv[qmod((x1, x2, x3, x4)) + 1] = true
        end
    else  # r == 5 (and any larger rank, though the form is quinary)
        for x1 in 0:M-1, x2 in 0:M-1, x3 in 0:M-1, x4 in 0:M-1, x5 in 0:M-1
            iseven(x1 | x2 | x3 | x4 | x5) || continue
            bv[qmod((x1, x2, x3, x4, x5)) + 1] = true
        end
    end
    return bv
end

# The translate X + b of a residue set X ⊂ ℤ/M, as a BitVector: rotate the
# table cyclically by b positions.
@inline function _rotate_bitvec(bv::BitVector, shift::Int, M::Int)
    shift = mod(shift, M)
    shift == 0 && return copy(bv)
    result = falses(M)
    @inbounds for i in 0:(M-1)
        bv[i+1] && (result[mod(i + shift, M) + 1] = true)
    end
    return result
end

# The primitive residue set R_d of L from its 2-adic Jordan decomposition,
# by the (prim, nonprim) recursion in the section header. Oscar returns each
# component's Gram matrix Gᵢ with its scale 2^{sᵢ} already included, so the
# residues of Gᵢ are used as they are; the scales are returned only for the
# record.
function _local2_by_jordan(L::ZZLat, depth::Int)
    M = 1 << depth
    _, blocks, scales = jordan_decomposition(L, ZZ(2))

    nonprim = falses(M); nonprim[1] = true   # the zero vector
    prim    = falses(M)

    for (G, _) in zip(blocks, scales)
        bp = _primitive_residues_bitvec(G, depth)
        bn = _nonprimitive_residues_bitvec(G, depth)

        next_prim = copy(prim)

        # prim' = prim ∪ (nonprim + Bp) ∪ (prim + Bp) ∪ (prim + Bn)
        @inbounds for b in 0:(M-1)
            bp[b+1] && (next_prim .|= _rotate_bitvec(nonprim, b, M))
            bp[b+1] && (next_prim .|= _rotate_bitvec(prim,    b, M))
            bn[b+1] && (next_prim .|= _rotate_bitvec(prim,    b, M))
        end

        # nonprim' = nonprim + Bn
        next_nonprim = falses(M)
        @inbounds for b in 0:(M-1)
            bn[b+1] && (next_nonprim .|= _rotate_bitvec(nonprim, b, M))
        end

        prim    = next_prim
        nonprim = next_nonprim
    end

    return (bitvec = prim,
            method = "jordan_decomposition(L, ZZ(2)) modulo 2^$depth (scale-corrected)",
            scales = Int[Int(scales[i]) for i in 1:length(scales)])
end

# Used when Oscar's `jordan_decomposition` throws: enumerate all primitive
# vectors of (ℤ/2^e)⁵ directly, with e = min(depth, 6) so that there are at
# most 64^5 ≈ 10^9 of them. If e < depth the result is lifted to length
# 2^depth by marking every residue mod 2^depth that reduces to an attained
# residue mod 2^e. The lift can only over-approximate R_depth, so it never
# excludes a level wrongly.
function _local2_direct_fallback(A::Matrix{BigInt}, depth::Int)
    enum_d = min(depth, 6)
    M_enum = 1 << enum_d
    M_full = 1 << depth
    bv_enum = falses(M_enum)
    A2 = mod.(A, M_enum)

    for x1 in 0:M_enum-1, x2 in 0:M_enum-1, x3 in 0:M_enum-1,
        x4 in 0:M_enum-1, x5 in 0:M_enum-1
        iseven(x1 | x2 | x3 | x4 | x5) && continue
        x = (x1, x2, x3, x4, x5)
        q = 0
        @inbounds for i in 1:5
            q += x[i] * A2[i, i] * x[i]
            for j in i+1:5
                q += 2 * x[i] * A2[i, j] * x[j]
            end
        end
        bv_enum[mod(q, M_enum) + 1] = true
    end

    if enum_d == depth
        return (bitvec = bv_enum,
                method = "direct primitive search modulo 2^$enum_d", scales = Int[])
    end
    bv_full = falses(M_full)
    step = M_enum
    for r in 0:M_enum-1
        bv_enum[r+1] || continue
        idx = r
        while idx < M_full
            bv_full[idx+1] = true
            idx += step
        end
    end
    return (bitvec = bv_full,
            method = "direct primitive search modulo 2^$enum_d (embedded into 2^$depth)",
            scales = Int[])
end

# The 2-adic primitive residue table, from the Jordan decomposition if Oscar
# can compute it and by direct enumeration otherwise.
function _build_local2_bitvec(A::Matrix{BigInt}, L::ZZLat, depth::Int)
    r = try
        _local2_by_jordan(L, depth)
    catch
        _local2_direct_fallback(A, depth)
    end
    return r.bitvec, r.method, r.scales
end

# ══════════════════════════════════════════════════════════════════════════════
# Odd-prime local check, level-independent
# ══════════════════════════════════════════════════════════════════════════════
#
# For an odd prime q ≠ ℓ the target ℓ^{2k} = (ℓ^k)² is a q-adic unit square,
# so Q_A represents it over ℤ_q iff it represents 1, iff the genus of L at q
# represents the rank-one lattice ⟨1⟩ — a condition on L alone, answered by
# Oscar's `represents` on local genus symbols. (When q = ℓ the target is
# divisible by q and this test says nothing; the level-dependent test below
# handles that case.)

function _odd_local_unit_test(L::ZZLat, bad_primes::Vector{BigInt})
    isempty(bad_primes) && return true
    unit_lattice = integer_lattice(; gram = matrix(ZZ, 1, 1, ZZRingElem[ZZ(1)]))
    for q in bad_primes
        q > 2 || continue
        qz = ZZ(q)
        represents(genus(L, qz), genus(unit_lattice, qz)) || return false
    end
    return true
end

# ══════════════════════════════════════════════════════════════════════════════
# Scaled 2-adic check for large k
# ══════════════════════════════════════════════════════════════════════════════
#
# For ℓ = 2 and 2k ≥ d the residue 4^k mod 2^d is 0, and the plain table can
# only tell whether 0 is attained — which it nearly always is. Let s be the
# minimal 2-adic Jordan scale of L, so that Q_A = 4^s·Q' with Q' having a
# unimodular component. A primitive x stays primitive under this scaling,
# so Q_A primitively represents ℓ^{2k} over ℤ₂ iff Q' primitively
# represents ℓ^{2k}/4^s; and the latter target is a non-zero residue at a
# second depth. The table for Q' is built by the same recursion as before,
# with every component's residues multiplied by 4^{sᵢ − s}.
#
# When s = 0 the two tables coincide and the scaled one is not built. This
# is the case for every form of determinant 16p² (see `Options`).

"""
    _min_jordan_scale(L) -> Int

The minimal 2-adic Jordan scale of `L`; 0 if L has a unimodular component,
and 0 (no scaling) if Oscar cannot decompose L.
"""
function _min_jordan_scale(L::ZZLat)::Int
    try
        _, _, scales = jordan_decomposition(L, ZZ(2))
        isempty(scales) && return 0
        return Int(minimum(scales))
    catch
        return 0
    end
end

"""
    _build_scaled_local2(L, A, min_scale, depth) -> Union{Nothing, BitVector}

The primitive residue table of Q_A / 4^min_scale modulo `2^depth`, or
`nothing` when `min_scale == 0`.
"""
function _build_scaled_local2(L::ZZLat, A::Matrix{BigInt}, min_scale::Int, depth::Int)
    min_scale == 0 && return nothing
    M = 1 << depth
    try
        _, blocks, scales = jordan_decomposition(L, ZZ(2))

        nonprim = falses(M); nonprim[1] = true
        prim    = falses(M)

        for (G, s) in zip(blocks, scales)
            shift = Int(s) - min_scale   # ≥ 0
            factor = Int(mod(BigInt(4)^shift, M))

            # Multiply every residue of this component by 4^shift mod M.
            rescale(bv_raw) = begin
                bv = falses(M)
                @inbounds for b in 0:(M-1)
                    bv_raw[b+1] && (bv[mod(b * factor, M) + 1] = true)
                end
                bv
            end

            bp = rescale(_primitive_residues_bitvec(G, depth))
            bn = rescale(_nonprimitive_residues_bitvec(G, depth))

            next_prim = copy(prim)
            @inbounds for b in 0:(M-1)
                bp[b+1] && (next_prim .|= _rotate_bitvec(nonprim, b, M))
                bp[b+1] && (next_prim .|= _rotate_bitvec(prim,    b, M))
                bn[b+1] && (next_prim .|= _rotate_bitvec(prim,    b, M))
            end

            next_nonprim = falses(M)
            @inbounds for b in 0:(M-1)
                bn[b+1] && (next_nonprim .|= _rotate_bitvec(nonprim, b, M))
            end

            prim    = next_prim
            nonprim = next_nonprim
        end

        return prim

    catch e
        @warn "Scaled 2-adic bitvec construction failed ($e); large-k 2-adic check will be skipped."
        return nothing
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Odd-prime local check, level-dependent
# ══════════════════════════════════════════════════════════════════════════════
#
# Representing the unit 1 and representing the unit ℓ^{2k} are the same
# question at the level of the genus, but the square class of ℓ^{2k} in
# ℤ_q^×/(ℤ_q^×)² enters the spinor norm, and the spinor genus is finer than
# the genus. To see it we ask, for each bad prime q, whether the genus of L
# at q represents the rank-one lattice ⟨ℓ^{2k} mod q²⟩; reducing modulo q²
# rather than q is what retains the square class. When q = ℓ the residue
# is 0 for k ≥ 1 and is replaced by 1, so this test is silent there.
#
# The genus symbol of L at q is computed once per bad prime in
# `build_problem`; the answers are remembered per residue ℓ^{2k} mod q²,
# of which there are at most q(q−1).

"""
    _build_odd_local_genus(L, bad_primes) -> Union{Nothing, Vector}

For each bad prime q, the named tuple `(q, genus_L, cache, lock)` with
`genus_L = genus(L, ZZ(q))` and `cache` the answers so far, keyed by
`ℓ^{2k} mod q²`. `nothing` if there are no bad primes. The lock guards the
cache when Phase 1 of `process_rhi2` runs on several threads.
"""
function _build_odd_local_genus(L::ZZLat, bad_primes::Vector{BigInt})
    isempty(bad_primes) && return nothing
    result = []
    for q in bad_primes
        q > 2 || continue
        qz = ZZ(q)
        push!(result, (q = q, genus_L = genus(L, qz),
                       cache = Dict{BigInt,Bool}(),
                       lock  = ReentrantLock()))
    end
    isempty(result) && return nothing
    return result
end

"""
    _odd_local_check_k(odd_local_genus, ell, k) -> Bool

`true` iff, at every bad prime q, the genus of L at q represents
`⟨ell^{2k} mod q²⟩` (with the residue 0, i.e. q = ell, replaced by 1).
"""
function _odd_local_check_k(odd_local_genus, ell::Integer, k::Integer)
    odd_local_genus === nothing && return true
    for entry in odd_local_genus
        q  = entry.q
        qz = ZZ(q)
        norm_val = powermod(BigInt(ell), 2 * k, q^2)
        norm_val = norm_val == 0 ? BigInt(1) : norm_val
        result = lock(entry.lock) do
            get!(entry.cache, norm_val) do
                target_lat = integer_lattice(; gram = matrix(ZZ, 1, 1, ZZRingElem[ZZ(norm_val)]))
                represents(entry.genus_L, genus(target_lat, qz))
            end
        end
        result || return false
    end
    return true
end

# ══════════════════════════════════════════════════════════════════════════════
# LLL reduction
# ══════════════════════════════════════════════════════════════════════════════
#
# LLL produces a unimodular U such that the basis with Gram matrix
# G' = UᵀAU is nearly orthogonal. Since U ∈ GL₅(ℤ), z ↦ Uz is a bijection of
# ℤ⁵ preserving primitivity, and Q_A(Uz) = G'(z); so the enumeration is
# carried out for G', whose shortest vectors are much shorter, and each
# vector found is carried back by x = Uz.

function _lll_reduce_form(A::Matrix{BigInt}, opt::Options)
    ctx        = LLLContext(opt.lll_delta, opt.lll_eta, :gram)
    Gred_zz, T = lll_gram_with_transform(matrix(ZZ, A), ctx)

    # U as a BigInt matrix (its entries need not fit in Int64 for large A),
    # T as the FLINT matrix used for x = Uz inside the enumeration.
    n = nrows(T)
    U = Matrix{BigInt}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        U[i, j] = BigInt(T[i, j])
    end

    return U, T, integer_lattice(; gram = Gred_zz)
end

# ══════════════════════════════════════════════════════════════════════════════
# Exact global search
# ══════════════════════════════════════════════════════════════════════════════
#
# `short_vectors_iterator(Lred, n, n)` from Hecke enumerates every z ∈ ℤ⁵ with
# G'(z) = n exactly. For each z:
#   1. test gcd(z) = 1 — since U is unimodular, gcd(Uz) = gcd(z), so
#      primitivity need not be re-tested after the change of coordinates;
#   2. form x = Uz;
#   3. confirm Q_A(x) = n in FLINT arithmetic and return x as the witness.
# If the iterator is exhausted the answer is a definite :NO_EXACT.
#
# The optional `stop` flag is polled every 256 vectors and ends the loop
# early with :UNKNOWN_TIMEOUT. It cannot interrupt the construction of the
# iterator itself, which for large targets is where Hecke spends most of
# its time inside C code; the time limit of `decide_global_only` therefore
# does not use it and runs the whole call on a separate process instead.

function _exact_search(A_zz::ZZMatrix, U_zz::ZZMatrix, Lred::ZZLat,
                       n::Int, ell::Integer, k::Int, max_nodes::Int,
                       stop::Union{Nothing, Base.Threads.Atomic{Bool}} = nothing)
    target    = pow_ell2(ell, k)
    target_zz = ZZ(target)

    iter  = short_vectors_iterator(Lred, target, target)
    nodes = 0

    # Scratch objects, reused for every candidate z.
    zm  = zero_matrix(ZZ, n, 1)
    xm  = zero_matrix(ZZ, n, 1)
    Ax  = zero_matrix(ZZ, n, 1)
    x   = Vector{BigInt}(undef, n)

    CHECK_INTERVAL = 256

    for (z_zz, _) in iter
        nodes += 1

        if stop !== nothing && (nodes % CHECK_INTERVAL == 0) && stop[]
            return (decision = :UNKNOWN_TIMEOUT,
                    witness  = nothing, value = nothing, gcd = nothing,
                    nodes    = nodes)
        end

        nodes > max_nodes && return (
            decision = :UNKNOWN_TOO_MANY_NODES,
            witness  = nothing, value = nothing, gcd = nothing, nodes = nodes)

        _isprimitive_zz(z_zz) || continue

        @inbounds for i in 1:n; zm[i, 1] = z_zz[i]; end
        Oscar.mul!(xm, U_zz, zm)                    # x = U z
        @inbounds for i in 1:n
            x[i] = BigInt(xm[i, 1])
        end

        if qvalue!(Ax, A_zz, xm) == target_zz
            return (decision = :YES_EXACT, witness = copy(x),
                    value = target, gcd = BigInt(1), nodes = nodes)
        end
    end

    return (decision = :NO_EXACT,
            witness  = nothing, value = nothing, gcd = nothing, nodes = nodes)
end

# ══════════════════════════════════════════════════════════════════════════════
# Global search on a worker process
# ══════════════════════════════════════════════════════════════════════════════
#
# With a time limit set, each enumeration is run on a second Julia process
# (a `Distributed` worker) rather than in this one. The reason is that the
# expensive step for a large target is the construction of Hecke's iterator,
# which runs to completion inside FLINT's C code without ever returning to
# Julia: no flag can be polled there and no signal is acted on, so nothing
# inside the same process can cut it short. A separate process can always be
# killed.
#
# One worker is started on first use, with Oscar and this file loaded on it,
# and is reused for every subsequent enumeration; loading Oscar costs some
# 10–20 s, and a worker is replaced only after it has been killed. Only
# plain BigInt matrices and integers are sent to it — never Oscar objects.
# The main process waits on a local channel fed by a task that fetches the
# worker's answer, since waiting on the remote result directly would block
# for as long as the worker does. On timeout the worker is killed with
# SIGKILL by its process id; a fresh one is started for the next call.
#
# Killing a worker this way makes `Distributed` print, once, something like
#
#     Worker 3 terminated.
#     Unhandled Task ERROR: EOFError: read end of file
#
# from its own bookkeeping. This is the expected sign of a deliberate kill,
# not an error in the results; the Phase 2 summary is what reflects them.

const _WORKER_LOCK = ReentrantLock()
const _WORKER_ID    = Ref{Int}(0)
const _WORKER_PID   = Ref{Int}(0)

"""
    _set_worker_memory_limit!(bytes::Integer) -> Bool

Cap the calling process's own address space at `bytes` (POSIX
`setrlimit(RLIMIT_AS)`, soft and hard limit both, so it cannot be raised
again). Run on the worker right after it is started, so that an oversized
enumeration fails at once with an `OutOfMemoryError` instead of pushing the
machine into swap. Linux and macOS only; elsewhere a warning and `false`.
"""
function _set_worker_memory_limit!(bytes::Integer)::Bool
    if !(Sys.islinux() || Sys.isapple())
        @warn "worker memory limit requested but setrlimit is only " *
              "supported on Linux/macOS; worker memory remains unbounded."
        return false
    end
    RLIMIT_AS = Sys.isapple() ? Cint(5) : Cint(9)  # the constant differs by OS
    # struct rlimit is two UInt64 fields on both platforms.
    rlim = Ref((UInt64(bytes), UInt64(bytes)))
    ret  = ccall(:setrlimit, Cint, (Cint, Ptr{Cvoid}), RLIMIT_AS, rlim)
    if ret != 0
        @warn "setrlimit(RLIMIT_AS, $bytes bytes) failed (return code $ret); " *
              "worker memory remains unbounded."
        return false
    end
    return true
end

"""
    _ensure_worker!(mem_limit_bytes=nothing) -> Int

Return the id of the worker process, starting one (and loading Oscar and
this file on it) if none is alive. The memory limit is applied only to a
newly started worker. Must be called holding `_WORKER_LOCK`.
"""
function _ensure_worker!(mem_limit_bytes::Union{Nothing,Integer} = nothing)::Int
    if _WORKER_ID[] == 0 || !(_WORKER_ID[] in workers())
        ids = addprocs(1)
        _WORKER_ID[]  = ids[1]
        _WORKER_PID[] = Distributed.remotecall_fetch(getpid, _WORKER_ID[])
        src = @__FILE__
        Distributed.remotecall_eval(Main, _WORKER_ID[], :(using Oscar))
        Distributed.remotecall_eval(Main, _WORKER_ID[], :(include($src)))
        if mem_limit_bytes !== nothing
            ok = Distributed.remotecall_fetch(
                _set_worker_memory_limit!,
                _WORKER_ID[], mem_limit_bytes)
            ok || @warn "failed to apply memory limit to new worker " *
                        "$(_WORKER_ID[]); it may exhaust system memory " *
                        "on large-k searches."
        end
    end
    return _WORKER_ID[]
end

"""
    _kill_worker!()

Kill the worker process outright (SIGKILL by process id) and forget it, so
that the next `_ensure_worker!` starts a fresh one. Assumes the worker runs
on this machine.
"""
function _kill_worker!()
    id  = _WORKER_ID[]
    pid = _WORKER_PID[]
    if id != 0
        if pid != 0
            try
                # stdio is discarded: if the worker has already died (e.g.
                # of its memory cap) `kill` would print "No such process".
                run(pipeline(Cmd(["kill", "-9", string(pid)]);
                             stdout = devnull, stderr = devnull))
            catch
            end

            # SIGKILL takes effect only when the kernel next schedules the
            # process; one stuck in uninterruptible I/O (swapping) can take
            # a while to die. Wait up to 5 s for it before starting a
            # replacement, so that two large processes do not compete for
            # memory.
            if Sys.islinux()
                grace_deadline = time() + 5.0
                still_alive = isdir("/proc/$pid")
                while still_alive && time() < grace_deadline
                    sleep(0.2)
                    try
                        run(pipeline(Cmd(["kill", "-9", string(pid)]);
                                     stdout = devnull, stderr = devnull))
                    catch
                    end
                    still_alive = isdir("/proc/$pid")
                end
                if still_alive
                    @warn "global-search worker pid $pid did not exit within " *
                          "5s of SIGKILL (likely stuck in an uninterruptible " *
                          "D-state from memory/swap pressure); it may still " *
                          "be running in the background and competing for " *
                          "resources with the freshly-spawned replacement worker."
                end
            end
        end
        if id in workers()
            try
                rmprocs(id; waitfor = 0)
            catch
            end
        end
    end
    _WORKER_ID[]  = 0
    _WORKER_PID[] = 0
    return nothing
end

"""
    shutdown_global_search_worker()

Kill the worker process if one is running; does nothing otherwise. Called
at the end of `process_rhi2` so that no process outlives the run.
"""
function shutdown_global_search_worker()
    lock(_WORKER_LOCK) do
        _kill_worker!()
    end
    return nothing
end

"""
    _worker_exact_search(Gred, U, A, ell, k, max_nodes) -> NamedTuple

The function run on the worker: rebuilds the FLINT objects from the plain
matrices it receives (`Gred` = Uᵀ A U, already LLL-reduced on the main
process) and calls `_exact_search`.
"""
function _worker_exact_search(Gred::Matrix{BigInt}, U::Matrix{BigInt},
                               A::Matrix{BigInt}, ell::BigInt, k::Int,
                               max_nodes::Int)
    n    = size(A, 1)
    A_zz = matrix(ZZ, A)
    U_zz = matrix(ZZ, U)
    Lred = integer_lattice(; gram = matrix(ZZ, Gred))
    return _exact_search(A_zz, U_zz, Lred, n, ell, k, max_nodes)
end

"""
    _exact_search_distributed(P, k, tmo) ->
        (decision, witness, value, gcd, nodes, spawn_time, search_time)

Run the enumeration for `(P, k)` on the worker process, killing it if no
answer has arrived after `tmo` seconds (result `:UNKNOWN_TIMEOUT`) or if it
fails for any other reason (`:UNKNOWN_WORKER_ERROR`, with the error logged).

`spawn_time` is the time spent starting a worker for this call — zero when
the previous one was still alive, some 10–20 s of Oscar loading when it had
been killed — and `search_time` the time waited for the answer, which is
what `tmo` bounds. A call following a timeout can thus take up to a worker
start longer than `tmo` in total.
"""
function _exact_search_distributed(P::Problem, k::Int, tmo::Real)
    n = size(P.A, 1)

    # Gram matrix of the reduced lattice as a plain BigInt matrix. Oscar
    # returns it with rational entries; the denominators are 1.
    Gm   = gram_matrix(P.Lred)
    Gred = Matrix{BigInt}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        Gred[i, j] = BigInt(numerator(Gm[i, j]))
    end

    # Copy the plain fields out of `P` first: `@spawnat` would otherwise
    # capture the whole `Problem`, Oscar objects included, and try to send
    # it to the worker.
    U         = P.U::Matrix{BigInt}
    Amat      = P.A
    ell       = P.ell
    max_nodes = P.options.max_nodes

    t_spawn0 = time()
    id = lock(_WORKER_LOCK) do
        _ensure_worker!(P.options.worker_memory_limit_bytes)
    end
    spawn_time = time() - t_spawn0

    # `_worker_exact_search` is resolved by name on the worker, against the
    # copy of this file loaded there.
    fut = Distributed.@spawnat id _worker_exact_search(
        Gred, U, Amat, ell, k, max_nodes)

    c = Channel{Any}(1)
    @async begin
        try
            put!(c, (:ok, fetch(fut)))
        catch e
            put!(c, (:err, e))
        end
    end

    # Poll the channel with a waiting time doubling from 1 ms to 500 ms.
    sleep_ms = 1.0
    t0 = time()
    while true
        if isready(c)
            status, payload = take!(c)
            search_time = time() - t0
            if status === :ok
                r = payload
                return (r.decision, r.witness, r.value, r.gcd, r.nodes,
                        spawn_time, search_time)
            else
                @warn "global search worker $id errored (k=$k)" exception = payload
                lock(_WORKER_LOCK) do
                    _kill_worker!()
                end
                return (:UNKNOWN_WORKER_ERROR, nothing, nothing, nothing, 0,
                        spawn_time, search_time)
            end
        end
        if time() - t0 > tmo
            search_time = time() - t0
            lock(_WORKER_LOCK) do
                _kill_worker!()
            end
            return (:UNKNOWN_TIMEOUT, nothing, nothing, nothing, 0,
                    spawn_time, search_time)
        end
        sleep(sleep_ms / 1000.0)
        sleep_ms = min(sleep_ms * 2.0, 500.0)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Deciding one level
# ══════════════════════════════════════════════════════════════════════════════

"""
    build_problem(A; ell=2, bad_primes=nothing, options=Options()) -> Problem

All level-independent work for the form with Gram matrix `A`: validation,
the 2-adic residue tables, the genus tests at the bad primes, and the LLL
reduction (skipped when `options.exact_cutoff < 1`, since then no
enumeration will ever be run).

`bad_primes` replaces the odd primes dividing det A, for use when they are
known in advance.
"""
function build_problem(Ain::AbstractMatrix{<:Integer};
                       ell::Integer = 2,
                       bad_primes::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                       options::Options = Options())
    ellZ = BigInt(ell)
    A, L = _check_quinary_form(Ain)
    bad  = bad_primes === nothing ? _odd_primes_from_det(L) : Vector{BigInt}(bad_primes)
    bad  = sort(unique(filter(>(2), bad)))

    bv, local2_method, scales = _build_local2_bitvec(A, L, options.two_adic_depth)
    odd_ok = _odd_local_unit_test(L, bad)

    min_s = _min_jordan_scale(L)
    scaled_bv = _build_scaled_local2(L, A, min_s, options.two_adic_scaled_depth)

    odd_genus_cache = options.odd_local_k_dependent ?
                      _build_odd_local_genus(L, bad) : nothing

    if all(bv)
        @debug "2-adic bitvec is all-true (every residue mod 2^$(options.two_adic_depth) " *
               "is primitively represented). jordan_scales = $scales."
    end
    if scaled_bv !== nothing && all(scaled_bv)
        @debug "Scaled 2-adic bitvec is also all-true at depth $(options.two_adic_scaled_depth)."
    end
    if !isempty(bad) && odd_ok
        @debug "Odd-prime check: bad_primes=$bad, odd_local_ok=true " *
               "(genus represents ⟨1⟩ at all bad primes — obstruction absent)."
    end
    if !isempty(bad) && !odd_ok
        @debug "Odd-prime check: bad_primes=$bad, odd_local_ok=false " *
               "(genus fails to represent ⟨1⟩ at some bad prime — obstruction present; " *
               "every k will fail at :NO_LOCAL_AT_ODD_BAD_PRIME)."
    end
    if odd_genus_cache !== nothing
        @debug "k-dependent odd-prime genus cache built for $(length(odd_genus_cache)) bad prime(s)."
    end

    if options.exact_cutoff < 1
        return Problem(ellZ, A, matrix(ZZ, A), L, bad, options,
                       local2_method, scales, bv, odd_ok,
                       nothing, nothing, nothing,
                       scaled_bv, min_s, odd_genus_cache)
    end

    U, U_zz, Lred = _lll_reduce_form(A, options)
    return Problem(ellZ, A, matrix(ZZ, A), L, bad, options,
                   local2_method, scales, bv, odd_ok, U, U_zz, Lred,
                   scaled_bv, min_s, odd_genus_cache)
end

"""
    decide(P, k) -> NamedTuple

Is `ell^{2k}` primitively represented by Q_A?  The local tests are applied
in order of cost and the enumeration only if all of them pass:

  1. plain 2-adic table:      `ell^{2k} mod 2^two_adic_depth ∈ P.local2_bitvec`;
  2. scaled 2-adic table:     for `2k ≥ two_adic_depth`, when the table exists;
  3. bad primes, level-independent: `P.odd_local_ok`;
  4. bad primes, level-dependent:   genus of L represents `⟨ell^{2k} mod q²⟩`;
  5. enumeration of the vectors of norm `ell^{2k}` in the reduced lattice.

The result carries `k`, `target`, the outcome of each local test, and

  decision   one of the symbols below;
  witness    the vector x found (step 5), or `nothing`;
  nodes      number of lattice vectors visited;
  method     a sentence describing which step decided.

Decisions
─────────
  `:YES_EXACT`                      A witness was found and verified.
  `:NO_EXACT`                       The enumeration was exhausted: none exists.
  `:NO_LOCAL_AT_2`                  Excluded by the plain 2-adic table.
  `:NO_LOCAL_AT_2_SCALED`           Excluded by the scaled 2-adic table.
  `:NO_LOCAL_AT_ODD_BAD_PRIME`      The genus of L does not represent ⟨1⟩ at
                                    some bad prime.
  `:NO_LOCAL_AT_ODD_BAD_PRIME_K`    The genus of L does not represent
                                    ⟨ell^{2k} mod q²⟩ at some bad prime q.
  `:UNKNOWN_ABOVE_EXACT_CUTOFF`     k > exact_cutoff; not searched.
  `:YES_ASSUMED_AFTER_EXACT_CUTOFF` k > exact_cutoff; assumed (unsound).
  `:UNKNOWN_TOO_MANY_NODES`         Enumeration stopped at max_nodes.
  `:UNKNOWN_TIMEOUT`                Enumeration exceeded the time limit
                                    (only via `decide_global_only`).
  `:UNKNOWN_WORKER_ERROR`           The worker process failed, e.g. hit its
                                    memory cap (only via `decide_global_only`).
"""
function decide(P::Problem, k::Integer)
    kk = Int(k)
    kk >= 0 || error("k must be non-negative")

    ell      = P.ell
    depth    = P.options.two_adic_depth
    target   = pow_ell2(ell, kk)

    # 1. plain 2-adic table
    residue2      = _target_residue_mod_2N(ell, kk, depth)
    local2        = P.local2_bitvec[residue2 + 1]
    two_adic_mod  = BigInt(1) << depth

    # 2. scaled 2-adic table: the target for Q_A / 4^s is ℓ^{2k} / 4^s, i.e.
    #    for ℓ = 2 the level k − s.
    scaled_depth   = P.options.two_adic_scaled_depth
    local2_scaled  = true
    if kk >= depth ÷ 2 && P.scaled_local2_bitvec !== nothing
        eff_k          = max(0, kk - P.min_jordan_scale)
        scaled_residue = _target_residue_mod_2N(ell, eff_k, scaled_depth)
        local2_scaled  = P.scaled_local2_bitvec[scaled_residue + 1]
    end

    # 3. bad primes, level-independent
    local_odd_ki = P.odd_local_ok

    # 4. bad primes, level-dependent
    local_odd_k  = true
    if P.options.odd_local_k_dependent && P.odd_local_genus !== nothing
        local_odd_k = _odd_local_check_k(P.odd_local_genus, ell, kk)
    end

    local_odd = local_odd_ki && local_odd_k

    base = (k                       = kk,
            target                  = target,
            local2                  = local2,
            local2_scaled           = local2_scaled,
            local_odd               = local_odd,
            local_odd_ki            = local_odd_ki,
            local_odd_k             = local_odd_k,
            two_adic_target_residue = residue2,
            two_adic_modulus        = two_adic_mod,
            two_adic_method         = P.two_adic_method,
            jordan_scales           = P.jordan_scales,
            bad_primes              = P.bad_primes)

    # spawn_time and search_time are 0 whenever no enumeration ran; they are
    # present so that every result has the same fields.
    local2 || return merge(base, (
        decision = :NO_LOCAL_AT_2,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at 2 (plain bitvec mod 2^$depth)"))

    local2_scaled || return merge(base, (
        decision = :NO_LOCAL_AT_2_SCALED,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at 2 (scaled residual form mod 2^$scaled_depth, " *
                   "min_jordan_scale=$(P.min_jordan_scale))"))

    local_odd_ki || return merge(base, (
        decision = :NO_LOCAL_AT_ODD_BAD_PRIME,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at odd bad prime (k-independent genus test)"))

    local_odd_k || return merge(base, (
        decision = :NO_LOCAL_AT_ODD_BAD_PRIME_K,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at odd bad prime (k-dependent genus test, k=$kk)"))

    if kk > P.options.exact_cutoff
        sym = P.options.assume_after_exact_cutoff ?
              :YES_ASSUMED_AFTER_EXACT_CUTOFF : :UNKNOWN_ABOVE_EXACT_CUTOFF
        return merge(base, (
            decision = sym,
            witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
            spawn_time = 0.0, search_time = 0.0,
            method   = "beyond exact cutoff (k > $(P.options.exact_cutoff))"))
    end

    # 5. enumeration, in this process and without time limit
    search_time = @elapsed begin
        decision, witness, value, gcd, nodes = _exact_search(
            P.A_zz, P.U_zz::ZZMatrix, P.Lred::ZZLat, size(P.A, 1), ell, kk, P.options.max_nodes)
    end
    return merge(base, (
        decision    = decision,
        witness     = witness,
        value       = value,
        gcd         = gcd,
        nodes       = nodes,
        spawn_time  = 0.0,
        search_time = search_time,
        method      = "Hecke.short_vectors_iterator on LLL-reduced lattice"))
end

"""
    scan(P, ks) -> Vector

`[decide(P, k) for k in ks]`.
"""
scan(P::Problem, ks) = [decide(P, k) for k in ks]

"""
    decide_local_only(P, k) -> NamedTuple

Steps 1–4 of `decide` only. Same result fields; the decision is
`:LOCAL_PASS` when every local test passes and one of the `:NO_LOCAL_*`
symbols otherwise. Cheap enough to run for every (form, k) at once.
"""
function decide_local_only(P::Problem, k::Integer)
    kk = Int(k)
    kk >= 0 || error("k must be non-negative")

    ell      = P.ell
    depth    = P.options.two_adic_depth
    target   = pow_ell2(ell, kk)

    residue2      = _target_residue_mod_2N(ell, kk, depth)
    local2        = P.local2_bitvec[residue2 + 1]
    two_adic_mod  = BigInt(1) << depth

    scaled_depth  = P.options.two_adic_scaled_depth
    local2_scaled = true
    if kk >= depth ÷ 2 && P.scaled_local2_bitvec !== nothing
        eff_k          = max(0, kk - P.min_jordan_scale)
        scaled_residue = _target_residue_mod_2N(ell, eff_k, scaled_depth)
        local2_scaled  = P.scaled_local2_bitvec[scaled_residue + 1]
    end

    local_odd_ki = P.odd_local_ok

    local_odd_k = true
    if P.options.odd_local_k_dependent && P.odd_local_genus !== nothing
        local_odd_k = _odd_local_check_k(P.odd_local_genus, ell, kk)
    end

    local_odd = local_odd_ki && local_odd_k

    base = (k                       = kk,
            target                  = target,
            local2                  = local2,
            local2_scaled           = local2_scaled,
            local_odd               = local_odd,
            local_odd_ki            = local_odd_ki,
            local_odd_k             = local_odd_k,
            two_adic_target_residue = residue2,
            two_adic_modulus        = two_adic_mod,
            two_adic_method         = P.two_adic_method,
            jordan_scales           = P.jordan_scales,
            bad_primes              = P.bad_primes)

    local2 || return merge(base, (
        decision = :NO_LOCAL_AT_2,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at 2 (plain bitvec mod 2^$depth)"))

    local2_scaled || return merge(base, (
        decision = :NO_LOCAL_AT_2_SCALED,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at 2 (scaled residual form mod 2^$scaled_depth, " *
                   "min_jordan_scale=$(P.min_jordan_scale))"))

    local_odd_ki || return merge(base, (
        decision = :NO_LOCAL_AT_ODD_BAD_PRIME,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at odd bad prime (k-independent genus test)"))

    local_odd_k || return merge(base, (
        decision = :NO_LOCAL_AT_ODD_BAD_PRIME_K,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "local obstruction at odd bad prime (k-dependent genus test, k=$kk)"))

    return merge(base, (
        decision = :LOCAL_PASS,
        witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
        spawn_time = 0.0, search_time = 0.0,
        method   = "all local checks passed (global search pending)"))
end

"""
    decide_global_only(P, k, local_result) -> NamedTuple

Step 5 of `decide` for a pair that `decide_local_only` returned as
`:LOCAL_PASS`; `local_result` supplies the common fields. With
`P.options.global_timeout_secs` set, the enumeration runs on the worker
process under that time limit (see "Global search on a worker process");
otherwise it runs here without limit.
"""
function decide_global_only(P::Problem, k::Integer, local_result)
    kk = Int(k)

    if kk > P.options.exact_cutoff
        sym = P.options.assume_after_exact_cutoff ?
              :YES_ASSUMED_AFTER_EXACT_CUTOFF : :UNKNOWN_ABOVE_EXACT_CUTOFF
        return merge(local_result, (
            decision = sym,
            witness  = nothing, value = nothing, gcd = nothing, nodes = 0,
            spawn_time = 0.0, search_time = 0.0,
            method   = "beyond exact cutoff (k > $(P.options.exact_cutoff))"))
    end

    tmo = P.options.global_timeout_secs
    if tmo === nothing
        spawn_time = 0.0
        search_time = @elapsed begin
            decision, witness, value, gcd, nodes = _exact_search(
                P.A_zz, P.U_zz::ZZMatrix, P.Lred::ZZLat,
                size(P.A, 1), P.ell, kk, P.options.max_nodes)
        end
    else
        decision, witness, value, gcd, nodes, spawn_time, search_time =
            _exact_search_distributed(P, kk, tmo)
    end

    return merge(local_result, (
        decision    = decision,
        witness     = witness,
        value       = value,
        gcd         = gcd,
        nodes       = nodes,
        spawn_time  = spawn_time,
        search_time = search_time,
        method      = "Hecke.short_vectors_iterator on LLL-reduced lattice"))
end

# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

"""
    result_summary(rows) -> NamedTuple

Sort the levels of a result vector (from `scan`) into represented, assumed,
not represented, and undecided.
"""
function result_summary(rows)
    yes_exact   = Int[]
    yes_assumed = Int[]
    no          = Int[]
    unknown     = Int[]
    for r in rows
        if     r.decision == :YES_EXACT
            push!(yes_exact,   r.k)
        elseif r.decision == :YES_ASSUMED_AFTER_EXACT_CUTOFF
            push!(yes_assumed, r.k)
        elseif r.decision in (:NO_EXACT,
                              :NO_LOCAL_AT_2,
                              :NO_LOCAL_AT_2_SCALED,
                              :NO_LOCAL_AT_ODD_BAD_PRIME,
                              :NO_LOCAL_AT_ODD_BAD_PRIME_K)
            push!(no,          r.k)
        else   # :UNKNOWN_*, :LOCAL_PASS
            push!(unknown,     r.k)
        end
    end
    return (yes_exact_ks      = yes_exact,
            yes_assumed_ks    = yes_assumed,
            no_ks             = no,
            unknown_ks        = unknown,
            count_yes_exact   = length(yes_exact),
            count_yes_assumed = length(yes_assumed),
            count_no          = length(no),
            count_unknown     = length(unknown))
end

# Short name for `result_summary`.
const summary = result_summary

"""
    print_rows(rows)

One line per result: the outcome of each local test, PASS/FAIL/SKIPPED
words for the local and global stages, the decision, and the witness.
"""
function print_rows(rows)
    for r in rows
        local_str, global_str = _local_global_status(r)
        l2s  = get(r, :local2_scaled, true)
        lokk = get(r, :local_odd_k,   true)
        println("k=", r.k,
                "  target=",         r.target,
                "  local2=",         r.local2,
                "  local2_scaled=",  l2s,
                "  odd=",            r.local_odd,
                "  odd_k_dep=",      lokk,
                "  local=",          local_str,
                "  global=",         global_str,
                "  decision=",       r.decision,
                "  nodes=",          r.nodes,
                "  witness=",        r.witness)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Reading the RHI2 files
# ══════════════════════════════════════════════════════════════════════════════

"""
    rhi2_filename_prefix(pZ) -> String

The part of an RHI2 filename that names the prime: the decimal string of
`p` when it has at most 24 digits, and otherwise `"{bits}bit_{hash}"` with
`bits` the bit length of p and `hash` eight hex digits computed from its
decimal string — e.g. `"1000bit_faa1c19f"` for `p = 2^721*3^176-1`. This
is the convention of the program that writes the RHI2 files.
"""
function rhi2_filename_prefix(pZ::BigInt)::String
    s = string(pZ)
    if length(s) <= 24
        return s
    end
    bits   = ndigits(pZ, base = 2)
    digest = string(hash(s) & 0xffffffff; base = 16, pad = 8)
    return "$(bits)bit_$(digest)"
end
rhi2_filename_prefix(p::Integer) = rhi2_filename_prefix(BigInt(p))

"""
    parse_gram_matrix(q_str) -> Matrix{BigInt}

Read a form written as
    "c₁₁*x1^2 + c₁₂*x1*x2 + … + c₅₅*x5^2"
into the symmetric Gram matrix A with Q_A(x) = xᵀAx: the coefficient of
`xi^2` is `A[i,i]`, and the coefficient of `xi*xj` (i ≠ j) is `2·A[i,j]`.
Coefficients are read as BigInt, since for large p they exceed Int64.
"""
function parse_gram_matrix(q_str::AbstractString)
    A = zeros(BigInt, 5, 5)
    s = replace(q_str, r"^q\(A,θ\)\s*=\s*" => "")

    for m in eachmatch(r"([+-]?\s*\d+)\s*\*\s*x(\d)\^2", s)
        i       = parse(Int, m.captures[2])
        A[i, i] = parse(BigInt, replace(m.captures[1], " " => ""))
    end
    for m in eachmatch(r"([+-]?\s*\d+)\s*\*\s*x(\d)\s*\*\s*x(\d)", s)
        coeff = parse(BigInt, replace(m.captures[1], " " => ""))
        i     = parse(Int, m.captures[2])
        j     = parse(Int, m.captures[3])
        i == j && continue
        half    = coeff ÷ 2
        A[i, j] = half
        A[j, i] = half
    end
    return A
end

"""
    read_rhi2_file(p; dir=".") -> Vector{Matrix{BigInt}}

Find `RHI2_{prefix}.txt` or `RHI2_{prefix}_N.txt` in `dir`, with `prefix`
as in `rhi2_filename_prefix` and `N` a run of digits, and return one Gram
matrix per "Type N" block, in file order.

A block's `q(A,θ) = …` line is taken as its form. The block ends at a
"ways to represent" line if there is one, and otherwise at the next
"Type" header or the end of the file. Reading stops at the summary line
"p = … (IKO …)", which is matched against the full decimal value of p.
"""
function read_rhi2_file(p::Integer; dir::AbstractString = ".")
    pZ     = BigInt(p)
    prefix = rhi2_filename_prefix(pZ)
    candidates = filter(readdir(dir; join = true)) do f
        occursin(Regex("RHI2_$(prefix)(_\\d+)?\\.txt\$"), f)
    end
    isempty(candidates)    && error("No RHI2_$(prefix).txt or RHI2_$(prefix)_*.txt found in \"$dir\"")
    length(candidates) > 1 && @warn "Multiple matches; using $(candidates[1])"

    all_lines = readlines(candidates[1])
    stop_idx  = findfirst(l -> occursin(Regex("^p\\s*=\\s*$(pZ)\\s+\\(IKO"), l),
                          all_lines)
    lines     = stop_idx === nothing ? all_lines : all_lines[1:stop_idx-1]

    matrices  = Matrix{BigInt}[]
    current_q = nothing
    for line in lines
        if occursin(r"^Type\s+\d+", line)
            current_q !== nothing && push!(matrices, parse_gram_matrix(current_q))
            current_q = nothing
        elseif occursin(r"^q\(A,θ\)\s*=", line)
            current_q = line
        elseif occursin(r"^ways to represent", line) && current_q !== nothing
            push!(matrices, parse_gram_matrix(current_q))
            current_q = nothing
        end
    end
    # The last block, when nothing follows it.
    current_q !== nothing && push!(matrices, parse_gram_matrix(current_q))

    isempty(matrices) && error("No form blocks parsed from $(candidates[1])")
    return matrices
end

# ══════════════════════════════════════════════════════════════════════════════
# Writing the results
# ══════════════════════════════════════════════════════════════════════════════

function _fmt_elapsed(secs::Float64)
    h = floor(Int, secs / 3600)
    m = floor(Int, (secs % 3600) / 60)
    s = secs % 60
    return "$(h)h $(m)m $(round(s, digits=1))s"
end

"""
    _local_global_status(r) -> (local_str, global_str)

Words for the local and global stages of one result row `r`.

Local:   "PASS", or "FAIL (2-adic)", "FAIL (2-adic scaled)",
         "FAIL (odd prime)", "FAIL (odd prime, k-dep)".
Global:  "SKIPPED (local fail)", "PASS (witness found)", "FAIL (no witness)",
         "SKIPPED (k > cutoff)", "ASSUMED (k > cutoff)", "ABORTED (node limit)",
         "ABORTED (timeout)", "ABORTED (worker error)".
"""
function _local_global_status(r)
    # Rows lacking a field are treated as having passed that test.
    l2s  = get(r, :local2_scaled, true)
    loki = get(r, :local_odd_ki,  r.local_odd)
    lokk = get(r, :local_odd_k,   true)

    local_pass = r.local2 && l2s && loki && lokk
    local_str  = if local_pass
        "PASS"
    elseif !r.local2
        "FAIL (2-adic)"
    elseif !l2s
        "FAIL (2-adic scaled)"
    elseif !loki
        "FAIL (odd prime)"
    else
        "FAIL (odd prime, k-dep)"
    end

    global_str = if !local_pass
        "SKIPPED (local fail)"
    elseif r.decision == :YES_EXACT
        "PASS (witness found)"
    elseif r.decision == :NO_EXACT
        "FAIL (no witness)"
    elseif r.decision == :UNKNOWN_ABOVE_EXACT_CUTOFF
        "SKIPPED (k > cutoff)"
    elseif r.decision == :YES_ASSUMED_AFTER_EXACT_CUTOFF
        "ASSUMED (k > cutoff)"
    elseif r.decision == :UNKNOWN_TOO_MANY_NODES
        "ABORTED (node limit)"
    elseif r.decision == :UNKNOWN_TIMEOUT
        "ABORTED (timeout)"
    elseif r.decision == :UNKNOWN_WORKER_ERROR
        "ABORTED (worker error)"
    else
        string(r.decision)
    end

    return local_str, global_str
end

function _fmt_global_row(r)
    _, global_str = _local_global_status(r)
    # Print the witness as "[1,0,…]" rather than "BigInt[1,0,…]".
    w = r.witness === nothing ? "none" :
        replace(string(r.witness), r"^\w+\[" => "[")
    return "  k=$(r.k)  $(global_str)  witness=$(w)  nodes=$(r.nodes)"
end

"""
    _fmt_timing_suffix(r) -> String

`"  elapsed=80.4s (spawn=20.1s, search=60.3s)"`: `search` is the time
bounded by `--timeout`, `spawn` the time spent starting a worker process
(non-zero only after the previous enumeration timed out).
"""
function _fmt_timing_suffix(r)
    spawn_time  = get(r, :spawn_time,  0.0)
    search_time = get(r, :search_time, 0.0)
    total = spawn_time + search_time
    return "  elapsed=$(round(total, digits=1))s" *
           " (spawn=$(round(spawn_time, digits=1))s," *
           " search=$(round(search_time, digits=1))s)"
end

function _write_form_block(io::IO, form_idx::Int, rows::Vector)
    println(io, "\nForm $form_idx")
    yes_rows = filter(r -> r.decision in
                     (:YES_EXACT, :YES_ASSUMED_AFTER_EXACT_CUTOFF), rows)
    yes_ks   = sort([r.k for r in yes_rows])
    if isempty(yes_ks)
        println(io, "min_k = none")
        println(io, "represented_ks = []")
        println(io, "count = 0")
    else
        println(io, "min_k = $(minimum(yes_ks))")
        println(io, "represented_ks = $yes_ks")
        println(io, "count = $(length(yes_ks))")
    end
    println(io, "witnesses:")
    for r in yes_rows
        println(io, _fmt_global_row(r))
    end
end

"""
    write_rhi_optimal_file(p, ell, all_rows; dir=".", kmax=nothing, elapsed_secs=nothing)

Write `RHI_prim_p{prefix}_ell{ell}_k{kmax}.txt` from result rows computed
elsewhere (`all_rows[i]` being the rows of form i): for each form, the
levels represented, the least of them, and the witnesses.
"""
function write_rhi_optimal_file(p::Integer, ell::Integer, all_rows::Vector;
                                dir::AbstractString                  = ".",
                                kmax::Union{Nothing,Int}             = nothing,
                                elapsed_secs::Union{Nothing,Float64} = nothing)
    isempty(all_rows) && error("all_rows is empty")
    pZ       = BigInt(p)
    pdisp    = rhi2_filename_prefix(pZ)
    kmax_val = kmax === nothing ?
        maximum(r.k for rows in all_rows for r in rows) : kmax
    outpath  = joinpath(dir, "RHI_prim_p$(pdisp)_ell$(ell)_k$(kmax_val).txt")
    log_p    = natural_log_bigint(pZ)

    open(outpath, "w") do io
        println(io, "=" ^ 60)
        println(io, "p = $pZ   ell = $ell   log(p) = $log_p   Kmax = $kmax_val")
        println(io, "forms = $(length(all_rows))")
        println(io, "=" ^ 60)
        for (i, rows) in enumerate(all_rows)
            _write_form_block(io, i, rows)
        end
        elapsed_secs !== nothing &&
            println(io, "\nTotal run time: $(_fmt_elapsed(elapsed_secs))")
    end
    @info "Written: $outpath"
    return outpath
end

# ══════════════════════════════════════════════════════════════════════════════
# The whole computation for one prime p
# ══════════════════════════════════════════════════════════════════════════════

"""
    process_rhi2(p; ell=2, dir=".", timeout=PRIM_DEFAULT_TIMEOUT_SECS[],
                 mem_gb=PRIM_DEFAULT_MEM_GB[], options=nothing,
                 bad_primes=nothing, kmax=nothing) -> outpath

Read the RHI2 file for `p` from `dir`, build a `Problem` for every form,
and decide every level `1 ≤ k ≤ kmax` (default `kmax = ⌈log p⌉`) for every
form, in two phases:

  Phase 1  the local tests for all (form, k) pairs, on all threads;
  Phase 2  the enumeration for each pair that passed Phase 1, one at a
           time, under the time limit if one is set.

Results go to `RHI_prim_p{prefix}_ell{ell}_k{kmax}.txt` in `dir`, written
as they are obtained: the Phase 1 failures, then for each form one line
per Phase 2 result other than `:NO_EXACT` (those are only counted, being
the great majority at large `kmax`), a summary line per form, and the
timings.

`timeout` (seconds per enumeration) and `mem_gb` (GiB for the worker
process) build the `Options` when `options` is not given; pass `options`
to control any other field, in which case these two are ignored.
"""
function process_rhi2(p::Integer;
                      ell::Integer = 2,
                      dir::AbstractString                                  = ".",
                      timeout::Union{Nothing,Real}                         = PRIM_DEFAULT_TIMEOUT_SECS[],
                      mem_gb::Union{Nothing,Real}                          = PRIM_DEFAULT_MEM_GB[],
                      options::Union{Nothing,Options}                      = nothing,
                      bad_primes::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                      kmax::Union{Nothing,Int}                             = nothing)
    if options === nothing
        mem_limit_bytes = mem_gb === nothing ? nothing : round(Int, Float64(mem_gb) * 1024^3)
        options = Options(global_timeout_secs       = timeout === nothing ? nothing : Float64(timeout),
                           worker_memory_limit_bytes = mem_limit_bytes)
    end
    pZ    = BigInt(p)
    ellZ  = BigInt(ell)
    pdisp = rhi2_filename_prefix(pZ)

    @info "Reading RHI2 file for p = $pZ …"
    matrices = read_rhi2_file(pZ; dir)
    nforms   = length(matrices)
    @info "  Found $nforms forms."

    log_p    = natural_log_bigint(pZ)
    kmax_nat = ceil(Int, log_p)
    kmax     = kmax === nothing ? kmax_nat : kmax
    @info "  Scanning k = 1…$kmax  (target = ell^{2k} = $(ellZ)^{2k}) …"

    timeout_str = options.global_timeout_secs === nothing ? "none" :
                  string(options.global_timeout_secs) * "s"

    outpath = joinpath(dir, "RHI_prim_p$(pdisp)_ell$(ellZ)_k$(kmax).txt")
    t_start = time()

    @info "  Building Problem structs for all $nforms form(s)…"
    problems = [build_problem(A; ell=ellZ, bad_primes, options) for A in matrices]

    try
        open(outpath, "w") do io

        println(io, "=" ^ 60)
        println(io, "p = $pZ   ell = $ellZ   log(p) = $log_p   Kmax = $kmax" *
                    "   timeout = $timeout_str")
        println(io, "forms = $nforms")
        println(io, "=" ^ 60)
        flush(io)

        # ── Phase 1: local tests for every (form, k), in parallel ────────────
        @info "  Phase 1: running local checks in parallel ($(Threads.nthreads()) thread(s))…"
        t_local = time()

        local_results = [Vector{Any}(undef, kmax) for _ in 1:nforms]
        work_items    = [(fi, k) for fi in 1:nforms for k in 1:kmax]
        Threads.@threads for (fi, k) in work_items
            local_results[fi][k] = decide_local_only(problems[fi], k)
        end

        elapsed_local = time() - t_local

        local_fail_log = String[]
        for fi in 1:nforms, k in 1:kmax
            r = local_results[fi][k]
            r.decision !== :LOCAL_PASS &&
                push!(local_fail_log, "  Form $fi  k=$k  $(first(_local_global_status(r)))")
        end

        println(io, "\nPhase 1 (local checks, parallel)  —  $(_fmt_elapsed(elapsed_local))")
        if isempty(local_fail_log)
            println(io, "  All $(nforms * kmax) (form, k) pairs passed.")
        else
            println(io, "  $(length(local_fail_log)) failure(s) out of $(nforms * kmax) pairs:")
            for msg in local_fail_log
                println(io, msg)
            end
        end
        flush(io)

        if isempty(local_fail_log)
            println("Phase 1 complete — all $(nforms * kmax) pairs passed. ($(_fmt_elapsed(elapsed_local)))")
        else
            println("Phase 1 complete — $(length(local_fail_log)) failure(s). ($(_fmt_elapsed(elapsed_local)))")
            for msg in local_fail_log; println(msg); end
        end

        # ── Phase 2: enumeration for the survivors, one at a time ────────────
        @info "  Phase 2: running global search (timeout=$timeout_str)…"
        if options.global_timeout_secs !== nothing
            @info "  (a timed-out search kills its worker process; Distributed " *
                  "may then print a \"Worker N terminated\" / EOFError message " *
                  "below, which is expected and does not affect the results)"
        end
        t_global = time()

        println(io, "\nPhase 2 (global search, serial)  —  timeout=$timeout_str")
        final_rows = [copy(local_results[fi]) for fi in 1:nforms]
        n_pass = 0; n_fail = 0; n_other = 0

        for (fi, prob) in enumerate(problems)
            println(io, "\n  Form $fi")
            n_fail_form = 0
            for k in 1:kmax
                lr = local_results[fi][k]
                if lr.decision !== :LOCAL_PASS
                    continue
                end
                r = decide_global_only(prob, k, lr)
                final_rows[fi][k] = r
                d = r.decision
                if d === :NO_EXACT
                    # Only counted; see the docstring.
                    n_fail_form += 1
                else
                    println(io, "  " * _fmt_global_row(r) * _fmt_timing_suffix(r))
                end
                flush(io)
                if d === :YES_EXACT || d === :YES_ASSUMED_AFTER_EXACT_CUTOFF
                    n_pass += 1
                elseif d === :NO_EXACT
                    n_fail += 1
                else
                    n_other += 1
                end
            end
            n_fail_form > 0 &&
                println(io, "  ($n_fail_form k-value(s): FAIL (no witness) — not shown individually)")

            yes_ks = sort([r.k for r in final_rows[fi]
                           if r.decision in (:YES_EXACT,
                                             :YES_ASSUMED_AFTER_EXACT_CUTOFF)])
            if isempty(yes_ks)
                println(io, "  Summary: min_k=none  represented_ks=[]  count=0")
            else
                println(io, "  Summary: min_k=$(minimum(yes_ks))" *
                            "  represented_ks=$yes_ks  count=$(length(yes_ks))")
            end
            flush(io)
        end

        elapsed_global = time() - t_global

        println(io, "\nPhase 2 complete  —  $(_fmt_elapsed(elapsed_global))")
        flush(io)

        parts = filter(!isempty, [
            n_pass  > 0 ? "$n_pass passed"      : "",
            n_fail  > 0 ? "$n_fail failed"       : "",
            n_other > 0 ? "$n_other inconclusive" : "",
        ])
        println("Phase 2 complete — $(join(parts, ", ")). ($(_fmt_elapsed(elapsed_global)))")

        total = time() - t_start
        println(io, "\nTotal wall time: $(_fmt_elapsed(total))")
        end
    finally
        # A worker exists only if a time limit was set; kill it even if a
        # phase threw.
        options.global_timeout_secs !== nothing && shutdown_global_search_worker()
    end

    @info "Done — results written to $outpath"
    return outpath
end

# ══════════════════════════════════════════════════════════════════════════════
# Command line
# ══════════════════════════════════════════════════════════════════════════════

"""
    parse_prime_expr(s) -> BigInt

Read `s` as a non-negative integer or as an expression in integer literals
and `+ - * ^ ( )` with the usual precedence (unary minus allowed), e.g.
`"2^721*3^176-1"`, evaluated in BigInt.

Julia's own parser produces the syntax tree, which is then evaluated by
hand, node by node, accepting only integer literals and the four
operators; no other Julia code can be run through this function. Exponents
are converted to `Int`.
"""
function parse_prime_expr(s::AbstractString)::BigInt
    str = strip(s)
    isempty(str) && error("empty prime expression")
    occursin(r"^[0-9+\-*^()\s]+$", str) ||
        error("prime expression \"$s\" contains characters other than digits and + - * ^ ( )")

    ast = try
        Meta.parse(str)
    catch e
        error("could not parse prime expression \"$s\": $e")
    end

    function eval_node(node)::BigInt
        if node isa Integer
            return BigInt(node)
        elseif node isa Expr && node.head === :call
            op   = node.args[1]
            args = node.args[2:end]
            if op === :+ && length(args) == 2
                return eval_node(args[1]) + eval_node(args[2])
            elseif op === :+ && length(args) == 1
                return eval_node(args[1])
            elseif op === :- && length(args) == 2
                return eval_node(args[1]) - eval_node(args[2])
            elseif op === :- && length(args) == 1
                return -eval_node(args[1])
            elseif op === :* && length(args) >= 2
                v = eval_node(args[1])
                for a in args[2:end]
                    v *= eval_node(a)
                end
                return v
            elseif op === :^ && length(args) == 2
                base = eval_node(args[1])
                exp  = eval_node(args[2])
                exp < 0 && error("negative exponent in prime expression \"$s\"")
                exp > typemax(Int) &&
                    error("exponent too large in prime expression \"$s\" (must fit in Int)")
                return base ^ Int(exp)
            else
                error("unsupported operator \"$op\" in prime expression \"$s\"")
            end
        else
            error("unsupported syntax in prime expression \"$s\" (node: $node)")
        end
    end

    return eval_node(ast)
end

#   julia --threads auto prim.jl <p> --ell <ell> [--dir <dir>] [--kmax <kmax>] [--timeout <secs>] [--mem-gb <GB>]
#
#   --threads auto    A flag of julia itself, so it must come before
#                     prim.jl. Lets Phase 1 use every core; without it
#                     Phase 1 runs on one thread. Affects only speed.
#   <p>               The prime (required, first argument): a plain integer
#                     or an expression over + - * ^ ( ) evaluated in
#                     BigInt, e.g. 2^721*3^176-1. Large primes are matched
#                     to files named RHI2_{bits}bit_{hash}_N.txt, see
#                     rhi2_filename_prefix.
#   --ell <ell>       The prime ℓ; the targets are ℓ^{2k} (required).
#   --dir <dir>       Directory holding RHI2_{prefix}.txt or
#                     RHI2_{prefix}_*.txt, where the result file is also
#                     written (default: the directory of this file).
#   --kmax <kmax>     Largest level tested for each form (default ⌈log p⌉).
#   --timeout <secs>  Time limit for each enumeration in Phase 2; an
#                     enumeration exceeding it is recorded as
#                     :UNKNOWN_TIMEOUT. Decimal values allowed. Default:
#                     no limit.
#   --mem-gb <GB>     Memory cap, in GiB, for the worker process running
#                     the enumerations; keeps a huge target from driving
#                     the machine into swap. Set it below the machine's
#                     RAM, leaving room for this process and the system.
#                     Decimal values allowed. Default: no cap.
#
# Flags may appear in any order after <p>.
if abspath(PROGRAM_FILE) == @__FILE__
    let
        function usage()
            println(stderr, """
Usage: julia --threads auto prim.jl <p> --ell <ell> [--dir <dir>] [--kmax <kmax>] [--timeout <secs>] [--mem-gb <GB>]

  <p>               prime to process (required).
                    Either a plain integer (e.g. 23), or an arithmetic
                    expression over + - * ^ ( ) evaluated in BigInt
                    (e.g. 2^721*3^176-1).
  --ell <ell>       prime base ell: target is ell^{2k}  (required).
  --dir <dir>       input/output directory  (default: script directory)
  --kmax <kmax>     maximum k value tested per form  (default: ⌈log p⌉)
  --timeout <secs>  per-(form,k) wall-time limit for global search in seconds,
                    enforced via a persistent Distributed worker that is
                    killed and replaced on timeout (default: no timeout;
                    decimal values accepted, e.g. 30.5)
  --mem-gb <GB>     hard memory cap for the global-search worker, in GiB
                    (via setrlimit; prevents a huge-k search from swapping
                    the whole machine and making --timeout overshoot;
                    decimal values accepted, e.g. 12.5; default: no cap)

Note: --threads auto is a Julia runtime flag, not a script flag, so it must
      precede `prim.jl` rather than follow it. It lets Phase 1 (local
      checks) run in parallel across all cores instead of one thread.
      Omitting it does not affect correctness, only Phase 1's speed.

Examples:
  julia --threads auto prim.jl 11 --ell 2
  julia --threads auto prim.jl 11 --ell 3
  julia --threads auto prim.jl 2^721*3^176-1 --ell 5 --dir ./data
  julia --threads auto prim.jl 23 --ell 7 --kmax 15
  julia --threads auto prim.jl 23 --ell 2 --kmax 30 --timeout 60
  julia --threads auto prim.jl 23 --ell 3 --kmax 1 --timeout 600
  julia --threads auto prim.jl 23 --ell 3 --kmax 30 --timeout 60 --mem-gb 12
""")
            exit(1)
        end

        length(ARGS) < 1 && usage()

        local p = parse_prime_expr(ARGS[1])
        p > 0 || error("parsed prime expression \"$(ARGS[1])\" evaluated to $p, which is not a positive integer")

        local dir     = dirname(@__FILE__)
        local kmax    = nothing   # Union{Nothing,Int}
        local timeout = nothing   # Union{Nothing,Float64}
        local mem_gb  = nothing   # Union{Nothing,Float64}
        local ell     = nothing   # Union{Nothing,BigInt}

        local i = 2
        while i <= length(ARGS)
            flag = ARGS[i]
            if flag == "--ell"
                i + 1 > length(ARGS) && (println(stderr, "Error: --ell requires an argument"); usage())
                ell_parsed = tryparse(Int, ARGS[i+1])
                (ell_parsed === nothing || ell_parsed < 2 || !is_probable_prime(ell_parsed)) &&
                    (println(stderr, "Error: --ell requires a prime integer ≥ 2, got \"$(ARGS[i+1])\""); usage())
                ell = BigInt(ell_parsed); i += 2
            elseif flag == "--dir"
                i + 1 > length(ARGS) && (println(stderr, "Error: --dir requires an argument"); usage())
                dir = ARGS[i+1]; i += 2
            elseif flag == "--kmax"
                i + 1 > length(ARGS) && (println(stderr, "Error: --kmax requires an argument"); usage())
                kmax = tryparse(Int, ARGS[i+1])
                (kmax === nothing || kmax < 1) &&
                    (println(stderr, "Error: --kmax requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--timeout"
                i + 1 > length(ARGS) && (println(stderr, "Error: --timeout requires an argument"); usage())
                timeout = tryparse(Float64, ARGS[i+1])
                (timeout === nothing || timeout <= 0) &&
                    (println(stderr, "Error: --timeout requires a positive number, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--mem-gb"
                i + 1 > length(ARGS) && (println(stderr, "Error: --mem-gb requires an argument"); usage())
                mem_gb = tryparse(Float64, ARGS[i+1])
                (mem_gb === nothing || mem_gb <= 0) &&
                    (println(stderr, "Error: --mem-gb requires a positive number, got \"$(ARGS[i+1])\""); usage())
                i += 2
            else
                println(stderr, "Error: unknown argument \"$flag\""); usage()
            end
        end

        ell === nothing && (println(stderr, "Error: --ell <prime> is required"); usage())
        isdir(dir) || (println(stderr, "Error: directory not found: \"$dir\""); exit(1))

        @info "Starting" p ell dir kmax timeout mem_gb
        outpath = process_rhi2(p; ell, dir, timeout, mem_gb, kmax)
        println("Done → $outpath")
    end
end


# Batch run over the primes p ≡ 11 (mod 12) below 260, at ℓ = 2 and ℓ = 3,
# with the RHI2 files in the current directory. This block is live: it runs
# whenever the file is loaded, including by `include("prim.jl")` and by the
# worker process. Comment it out for interactive use or before running with
# --timeout.
# for p in 11:12:260
#     if is_probable_prime(p)
#         process_rhi2(p; ell=2)
#         process_rhi2(p; ell=3)
#     end
# end
