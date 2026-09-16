# kzero.jl — the first level at which a quinary form primitively represents ℓ^{2k}
#
# For every positive definite quinary form L listed in an RHI2 file, and a
# fixed prime ℓ, this program computes
#
#     k0(L) = min { k ≥ 1 : r*(L, ℓ^{2k}) > 0 },
#
# the least k for which ℓ^{2k} is represented by a PRIMITIVE vector of L.
#
# An RHI2 file (`RHI2_<p>.txt`) is a plain-text table with one block per
# form, each block headed `Type N` and containing a line
# `q(A,θ) = c11*x1^2 + c12*x1*x2 + …` giving the form as a polynomial. Only
# those lines are read; everything else in the file is ignored.
#
# ─── Notation ────────────────────────────────────────────────────────────────
#
#   A          a symmetric positive definite 5×5 integer matrix.
#   Q_A(x)     the quadratic form x'Ax on x ∈ ℤ⁵. Every form in one RHI2 file
#              has the same determinant, 16p².
#   L          the lattice ℤ⁵ carrying Q_A. "Form", "lattice" and "class" are
#              used interchangeably: the RHI2 file lists one Gram matrix per
#              isometry class, so one form is one class.
#   primitive  x ∈ L is primitive when gcd(x₁,…,x₅) = 1, i.e. x is not a
#              proper multiple of another lattice vector.
#   shell      { x ∈ L : Q_A(x) = m } for a fixed m.
#   r(L,m)     #{ x ∈ L : Q_A(x) = m }, the size of the shell;
#   r*(L,m)    #{ x ∈ L : Q_A(x) = m, x primitive }, its primitive part.
#              Neither is ever counted here. The program only asks whether a
#              shell contains a primitive vector and stops at the first one.
#   level k    the shell of norm ℓ^{2k}.
#   λ₁(L)      min{ Q_A(x) : x ≠ 0 }, the lattice minimum — a SQUARED length
#              in the convention Q_A(x) = x'Ax.
#   |Aut L|    the order of the isometry group of L; always ≥ 2 since −1 is
#              an isometry.
#   A_p        the form ⟨1⟩ ⊥ B ⊥ B with B = [4 2; 2 p+1], of determinant
#              16p². The forms in an RHI2 file lie in genus(A_p); the program
#              confirms this once per form and requires p ≡ 11 (mod 12), the
#              setting the RHI2 files are generated in.
#
# k0 is found by direct search: for k = 1, 2, … the shell of norm ℓ^{2k} is
# enumerated until a primitive vector appears. No local (ν-adic) condition is
# tested at any level.
#
# ─── Why the primitive question is the interesting one ───────────────────────
#
# Every x with Q_A(x) = m is uniquely d·y with y primitive and Q_A(y) = m/d².
# Summing over d = ℓ^j,
#
#     r(L, ℓ^{2k}) = Σ_{i=0}^{k} r*(L, ℓ^{2i}),   so
#     r*(L, ℓ^{2k}) = r(L, ℓ^{2k}) − r(L, ℓ^{2k−2})   for k ≥ 1.
#
# Once any level is represented every higher level is too (take ℓx), but those
# inherited vectors are imprimitive and contribute nothing to r*. So
# representation is monotone in k, primitive representation is not, and k0 is
# a genuine invariant of the class rather than something read off a bound.
#
# ─── Where to start and where to stop ────────────────────────────────────────
#
# (1) The lattice minimum is a barrier (a theorem).
#     If ℓ^{2k} < λ₁(L) then L represents nothing of that size. Hermite's
#     inequality gives λ₁(L) ≤ 8^{1/5}(16p²)^{1/5} ≍ p^{2/5}, so the barrier
#     sits near k ≈ (log p)/(5 log ℓ). Levels below it are skipped without any
#     search; λ₁ is computed exactly.
#
# (2) Heuristic scales, from Siegel's mass formula.
#     Siegel [Ann. of Math. 36 (1935)] gives the |Aut|-weighted average of
#     r(L,m) over the classes of a genus as a product of local densities.
#     For rank 5 and det = 16p², suppressing the local densities,
#
#         r(gen, m) ≈ (π²/3) · m^{3/2} / p.
#
#     A class typically begins to represent ℓ^{2k} where this average reaches
#     1, i.e. at  k0 ≈ (log p)/(3 log ℓ);  and every class is expected to
#     represent once m ≫ det^{2/3}, i.e. at  k ≈ 2·(log p)/(3 log ℓ) = 2k0,
#     where the Eisenstein part of the theta series dominates the cusp part.
#     Neither statement is a theorem. The first is the "predicted" column in
#     the report; the second is the default level at which the scan gives up.
#
# ─── Algorithms used as black boxes ──────────────────────────────────────────
#
# Almost none of the arithmetic below is implemented here; four published
# algorithms, reached through Oscar/Hecke/Nemo/FLINT, do the work.
#
#   Fincke–Pohst enumeration  `short_vectors_iterator`
#     U. Fincke, M. Pohst, Math. Comp. 44 (1985), 463–471.
#     Enumerates the vectors of a shell. More than 99% of the running time.
#   LLL reduction  `lll_gram_with_transform`
#     A. K. Lenstra, H. W. Lenstra Jr., L. Lovász, Math. Ann. 261 (1982),
#     515–534. Run once per form; without it the enumeration does not finish.
#   Genus symbols  `genus(L, ν)`
#     Conway–Sloane, Sphere Packings, Lattices and Groups, 3rd ed., Ch. 15.
#     Two comparisons (at 2 and at p) confirm L ∈ genus(A_p); < 1 ms.
#   Isometry group  `automorphism_group_order`
#     W. Plesken, B. Souvignier, J. Symbolic Comput. 24 (1997), 327–334.
#     Supplies |Aut L|, reported as a covariate of k0.
#
# The enumeration is never truncated: a cut-off search would report a shell
# as empty when it is not, giving a k0 that is too LARGE, and nothing
# downstream could detect it. A shell that never finishes is handled outside
# the arithmetic, by the wall clock and the ABANDONED status below.
#
# ─── Status of a form ────────────────────────────────────────────────────────
#
#   OK               k0 was found and the vector realising it was re-checked
#                    in the original basis. Every column is meaningful.
#   NO_K0_BELOW_CAP  the scan reached --kmax-cap without a primitive vector.
#                    This says k0 > cap and nothing more. Raise the cap and
#                    rerun; levels already proved empty are not rescanned.
#   NOT_IN_GENUS     the form is not in genus(A_p); it was not scanned.
#   ABANDONED        started --max-attempts times and never finished. No
#                    mathematical content; it stops one pathological form
#                    from consuming every future resubmission.
#
# Only OK rows carry a k0. Elsewhere k0 = −1 means "unknown", and aut = 0
# means "not computed" (|Aut L| ≥ 2 always).
#
# ─── Running several batches at once ─────────────────────────────────────────
#
# One process is strictly serial: FLINT and Hecke are not safe to drive from
# several Julia threads. Parallelism comes from independent processes that
# share nothing. With `--batch j --of N` a process handles the forms whose
# 1-based index i satisfies i ≡ j+1 (mod N). Interleaving rather than cutting
# the file into consecutive blocks is deliberate: the cost of a form varies by
# more than an order of magnitude and the expensive ones are not spread
# evenly, so consecutive blocks would finish at very different times.
#
# ─── Restarting after an interruption ────────────────────────────────────────
#
# Every finished form is appended to the results file and forced to disk at
# once, so an interruption costs at most the form in flight. On restart a
# batch reads its own file and skips what it already has; rerunning the same
# command is safe and is the intended way to finish a run.
#
# Just before form i is started, a comment line `# S <i>` is appended. A form
# with --max-attempts such lines and no data row is written off as ABANDONED.
# The counter sees interruptions, not their causes: two unrelated kills look
# exactly like one pathological form, which is why the default is 3 and why
# --max-attempts exists.
#
# ─── Glossary of machine words ───────────────────────────────────────────────
#
#   job array   one submission to a cluster scheduler that starts N copies of
#               a program differing only in an index (here, --batch).
#   core-hour   one processor busy for one hour.
#   flush       force buffered output onto disk immediately.
#   node        a vertex of the Fincke–Pohst search tree; not a machine.
#
# ─── Launch ──────────────────────────────────────────────────────────────────
#
#   julia --startup-file=no kzero.jl <p> --ell <ℓ> [options]
#   julia --startup-file=no kzero.jl --report <csv>...
#
# See `usage()` at the foot of the file for the options.

using Oscar
using Dates
using Printf

const ZZ = Oscar.ZZ
const QQ = Oscar.QQ

# Default for --max-attempts. Three allows two genuine retries, since an
# interrupted attempt usually means the job was killed, not that the form is
# pathological.
const DEFAULT_MAX_ATTEMPTS = 3

# ══════════════════════════════════════════════════════════════════════════════
# Evaluating the form
# ══════════════════════════════════════════════════════════════════════════════

pow_ell2(ell::Integer, k::Integer) = BigInt(ell)^(2 * Int(k))

"""
    natural_log_bigint(n) -> Float64

`log(n)` for a `BigInt` that may exceed the range of `Float64`.
"""
function natural_log_bigint(n::BigInt)::Float64
    n > 0 || error("natural_log_bigint requires a positive argument, got $n")
    return Float64(log(BigFloat(n)))
end

# Q_A(x) = x'Ax, computed inside FLINT.
function qvalue(A_zz::ZZMatrix, xm::ZZMatrix)
    return dot(xm, A_zz * xm)
end

# Allocation-free variant; `scratch` is an n×1 matrix reused for A·x.
@inline function qvalue!(scratch::ZZMatrix, A_zz::ZZMatrix, xm::ZZMatrix)
    Oscar.mul!(scratch, A_zz, xm)
    return dot(xm, scratch)
end

function qvalue(A_zz::ZZMatrix, x::AbstractVector{<:Integer})
    xm = matrix(ZZ, length(x), 1, ZZRingElem[ZZ(v) for v in x])
    return BigInt(qvalue(A_zz, xm))
end

function qvalue(A::AbstractMatrix{<:Integer}, x::AbstractVector{<:Integer})
    size(A, 1) == size(A, 2) == length(x) || error("dimension mismatch")
    return qvalue(matrix(ZZ, A), x)
end

# gcd(x) = 1, with an early exit once the running gcd reaches 1.
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

Independent check that `x` primitively represents `ell^{2k}` under `Q_A`.
It works from the plain integer matrix and shares nothing with the reduced
lattice or the enumeration, so a wrong basis change cannot make a bad vector
look good.
"""
function verify_witness(A::AbstractMatrix{<:Integer}, x, k::Integer, ell::Integer)
    target = pow_ell2(ell, k)
    v      = qvalue(A, x)
    pg     = isempty(x) ? BigInt(0) : BigInt(reduce(gcd, x))
    return (value = v, target = target, primitive_gcd = pg,
            ok    = v == target && pg == 1)
end

# ══════════════════════════════════════════════════════════════════════════════
# The form A_p and the genus check
# ══════════════════════════════════════════════════════════════════════════════

"""
    Qp_lattice(p) -> ZZLat

The lattice A_p = ⟨1⟩ ⊥ B ⊥ B, B = [4 2; 2 p+1], of determinant 16p².
"""
function Qp_lattice(p::BigInt)
    A = BigInt[1 0 0 0 0;
               0 4 0 2 0;
               0 0 4 0 2;
               0 2 0 p+1 0;
               0 0 2 0 p+1]
    return integer_lattice(; gram = matrix(ZZ, A))
end

"""
    in_genus_Qp(L, ref, p) -> Bool

Whether `L ∈ genus(A_p)`: equal determinant and equal genus symbols at 2 and
at p (for L positive definite of rank 5 with det L = 16p², these are the only
primes at which the local structure can differ). `ref` is `Qp_lattice(p)`.
"""
function in_genus_Qp(L::ZZLat, ref::ZZLat, p::BigInt)::Bool
    det(gram_matrix(L)) == det(gram_matrix(ref)) || return false
    genus(L, ZZ(2)) == genus(ref, ZZ(2))         || return false
    genus(L, ZZ(p)) == genus(ref, ZZ(p))         || return false
    return true
end

# ══════════════════════════════════════════════════════════════════════════════
# LLL reduction and the lattice minimum
# ══════════════════════════════════════════════════════════════════════════════

function _check_quinary_form(Ain::AbstractMatrix{<:Integer})
    A = Matrix{BigInt}(Ain)
    size(A) == (5, 5) || error("expected a 5×5 Gram matrix")
    A == A'           || error("Gram matrix must be symmetric")
    L = integer_lattice(; gram = matrix(ZZ, A))
    is_positive_definite(L) || error("Gram matrix must be positive definite")
    return A, L
end

"""
    _lll_reduce_form(A; delta, eta) -> (U, Lred)

LLL-reduce the Gram matrix `A`. Returns the unimodular matrix `U` with
`U'·A·U = Gred` and the lattice `Lred` with Gram matrix `Gred`, so that a
vector `z` in the reduced basis is `x = U·z` in the original one.

`delta` and `eta` are LLL's own parameters. `delta < 1` is the slack in the
Lovász condition; taking it near 1 gives the strongest reduction and hence the
smallest enumeration tree. `eta > 1/2` is the slack in size reduction. Neither
affects correctness: any basis of the same lattice has the same shells.

`lll_gram_with_transform` returns `(Gred, T)` with `Gred = T·A·T'`, i.e. the
ROWS of `T` are the reduced basis, so `U = T'`. Getting this transposed is
invisible on a form that is already reduced (`T = I`, the majority of the
input) and turns representations into apparent non-representations on the
rest; the identity `U'·A·U = Gred` is therefore checked outright.
"""
function _lll_reduce_form(A::Matrix{BigInt}; delta::Float64 = 0.99,
                          eta::Float64 = 0.501)
    ctx        = LLLContext(delta, eta, :gram)
    A_zz       = matrix(ZZ, A)
    Gred_zz, T = lll_gram_with_transform(A_zz, ctx)
    U_zz       = transpose(T)

    transpose(U_zz) * A_zz * U_zz == Gred_zz ||
        error("LLL transform convention violated: U'·A·U ≠ Gred; " *
              "see `_lll_reduce_form`")

    return U_zz, integer_lattice(; gram = Gred_zz)
end

"""
    _lattice_minimum(Lred) -> BigInt

λ₁ = min{ Q(x) : x ≠ 0 }, a SQUARED length. Hecke's exact `minimum` is itself
a Fincke–Pohst computation and is cheap only because the lattice is reduced.

Should it fail, the first vector b₁ of an LLL-reduced basis satisfies
Q(b₁) ≤ 2^{n−1}·λ₁ (the constant for δ = 3/4, hence conservative here), and
since `Gred[1,1] = Q(b₁)` this gives the LOWER bound `λ₁ ≥ Gred[1,1]/2^{n−1}`.
A lower bound is safe: it can make the scan visit a level that is provably
empty, never skip one that is not. Failing that too, 0 disables the barrier,
which costs time and nothing else.
"""
function _lattice_minimum(Lred::ZZLat)::BigInt
    try
        return BigInt(numerator(QQ(minimum(Lred))))
    catch
    end
    try
        G = gram_matrix(Lred)
        n = nrows(G)
        return BigInt(numerator(QQ(G[1, 1]))) >> (n - 1)
    catch
    end
    return BigInt(0)
end

"""
    default_kmax_cap(p, ell) -> Int

The level at which the scan gives up when --kmax-cap is not given:
`ceil(2·ln p / (3·ln ell))`, the level at which ℓ^{2k} passes det^{2/3} and
every class is heuristically expected to represent (item (2) of the header).
It is twice the heuristic k0.

The cap must scale with log p / log ℓ. A constant is wrong as soon as p
changes: 40 is generous for a 50-bit p at ℓ = 2 (typical k0 ≈ 16) and below
the typical k0 for a 200-bit p, where every form would report NO_K0_BELOW_CAP
at enormous cost.
"""
function default_kmax_cap(p::BigInt, ell::BigInt)::Int
    return max(1, ceil(Int, 2 * natural_log_bigint(p) /
                            (3 * log(Float64(ell)))))
end

"""
    first_k_at_or_above_minimum(ell, lambda1) -> Int

Least k ≥ 1 with ℓ^{2k} ≥ λ₁. No vector of L has norm below λ₁, so the
levels below this one are empty without any enumeration.
"""
function first_k_at_or_above_minimum(ell::Integer, lambda1::BigInt)::Int
    lambda1 > 0 || return 1
    step = BigInt(ell)^2
    t    = step
    k    = 1
    while t < lambda1
        t *= step
        k += 1
    end
    return k
end

# ══════════════════════════════════════════════════════════════════════════════
# Searching one shell for a primitive vector
# ══════════════════════════════════════════════════════════════════════════════

"""
    first_primitive_in_shell(A_zz, U_zz, Lred, n, target)
        -> Union{Vector{BigInt}, Nothing}

The first primitive `x` with `Q_A(x) = target`, in the ORIGINAL basis, or
`nothing` if the shell has none.

`short_vectors_iterator(Lred, lo, hi)` enumerates the vectors z of the
reduced lattice with lo ≤ Q(z) ≤ hi; passing lo = hi = target pins it to one
shell, so the Fincke–Pohst branch-and-bound prunes from both sides. Three
properties of the enumeration shape this function:

  * It is depth-first, so a vector of the target norm typically surfaces long
    before the tree is exhausted — roughly ten times sooner. Stopping at the
    first primitive vector is what makes the program affordable. An empty
    shell must still be exhausted, but the levels below k0 are geometrically
    cheaper (about ℓ³ per level), so the expensive level is the one that
    succeeds.
  * It returns one of each pair ±z. Harmless: ±z are primitive together and
    represent the same value. The witness recorded is one of a pair.
  * It is never truncated (see the header).

Primitivity is tested in the reduced coordinates, which is legitimate because
U is unimodular: gcd(z) = 1 ⟺ gcd(Uz) = 1. The back-transform is then done
only for the vector kept, and its value is re-checked in the original basis
before it is returned.
"""
function first_primitive_in_shell(A_zz::ZZMatrix, U_zz::ZZMatrix, Lred::ZZLat,
                                  n::Int, target::BigInt)
    target_zz = ZZ(target)
    iter = short_vectors_iterator(Lred, target, target)

    zm = zero_matrix(ZZ, n, 1)
    xm = zero_matrix(ZZ, n, 1)
    Ax = zero_matrix(ZZ, n, 1)
    x  = Vector{BigInt}(undef, n)

    for (z_zz, _) in iter
        _isprimitive_zz(z_zz) || continue

        @inbounds for i in 1:n; zm[i, 1] = z_zz[i]; end
        Oscar.mul!(xm, U_zz, zm)
        @inbounds for i in 1:n
            x[i] = BigInt(xm[i, 1])
        end
        qvalue!(Ax, A_zz, xm) == target_zz ||
            error("back-transformed vector has Q_A(x) ≠ target; " *
                  "the LLL transform convention is wrong")
        return copy(x)
    end

    return nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# One form, start to finish
# ══════════════════════════════════════════════════════════════════════════════

"""
    study_form(A, ell, p, ref; kmax_cap, want_aut, label) -> NamedTuple

The complete record for one form `A`; one row of the results file is exactly
its return value. In order of increasing cost:

  1. shape and positive-definiteness;
  2. the genus check; a form outside genus(A_p) is returned as
     `NOT_IN_GENUS` without enumeration;
  3. |Aut L| by Plesken–Souvignier, 1–4 ms (the seconds seen on the very
     first form are Julia compiling);
  4. LLL reduction and λ₁, which fix `kmin`, the first level that can hold a
     vector at all;
  5. shells at k = kmin, kmin+1, …, stopping at the first primitive vector or
     at `kmax_cap`.

Steps 1–4 cost about 10 ms; step 5 is everything else, and within it the
level k = k0 is almost the whole cost.

Fields of the result, which are the columns of the results file:

  `status`    `"OK"`, `"NO_K0_BELOW_CAP"` or `"NOT_IN_GENUS"` (`"ABANDONED"`
              is written by the caller, never here).
  `in_genus`  the outcome of the genus test.
  `lambda1`   λ₁, a squared length; 0 if unavailable.
  `aut`       |Aut L|, or 0 for "not computed".
  `kmin`      least k with ℓ^{2k} ≥ λ₁.
  `k0`        least k with a primitive representation of ℓ^{2k}, or −1 if
              none was found at or below `kmax_cap`.
  `witness`   the primitive vector realising k0, in the original basis, or
              `nothing`. Verified independently before being returned.
  `secs`      wall-clock seconds for this form.

Throws rather than returning a row if a witness fails re-verification: a wrong
witness is a bug and must not reach the results file looking like data.
"""
function study_form(A::Matrix{BigInt}, ell::BigInt, p::BigInt, ref::ZZLat;
                    kmax_cap::Int, want_aut::Bool = true,
                    label::AbstractString = "")
    t0 = time()
    Amat, L = _check_quinary_form(A)

    ingenus = in_genus_Qp(L, ref, p)
    if !ingenus
        return (status = "NOT_IN_GENUS", in_genus = false,
                lambda1 = BigInt(0), aut = BigInt(0), kmin = 0, k0 = -1,
                witness = nothing, secs = time() - t0)
    end

    # A failure here is recorded as 0 rather than allowed to abort the form:
    # |Aut L| is a covariate, whereas losing the form loses its k0.
    aut = BigInt(0)
    if want_aut
        try
            aut = BigInt(automorphism_group_order(L))
        catch
            aut = BigInt(0)
        end
    end

    A_zz       = matrix(ZZ, Amat)
    U_zz, Lred = _lll_reduce_form(Amat)
    n          = size(Amat, 1)
    lambda1    = _lattice_minimum(Lred)

    kmin    = first_k_at_or_above_minimum(ell, lambda1)
    k0      = -1
    witness = nothing

    # Announce the form before the expensive levels run: a single level can
    # take an hour, and the log must say what is being attempted before the
    # silence starts.
    k0_heur = natural_log_bigint(p) / (3 * log(Float64(ell)))
    _progress("$label  lambda1=$lambda1  kmin=$kmin  k0~$(round(k0_heur, digits=1))" *
              "  |Aut|=$aut")

    # Progress is reported per level, and cannot be finer: the dominant cost
    # inside one level is spent in FLINT's C code, which cannot be polled from
    # Julia.
    for k in kmin:kmax_cap
        t_lvl = time()
        w = first_primitive_in_shell(A_zz, U_zz, Lred, n, pow_ell2(ell, k))
        el = time() - t_lvl
        _progress(@sprintf("%s  k=%-3d %-22s %s", label, k,
                           w === nothing ? "empty" : "PRIMITIVE VECTOR FOUND",
                           _fmt_secs(el)))
        if w !== nothing
            k0      = k
            witness = w
            vw = verify_witness(Amat, w, k, ell)
            vw.ok || error("witness failed independent re-verification at k=$k")
            break
        end
    end

    status = k0 < 0 ? "NO_K0_BELOW_CAP" : "OK"
    return (status = status, in_genus = true, lambda1 = lambda1, aut = aut,
            kmin = kmin, k0 = k0, witness = witness, secs = time() - t0)
end

# ══════════════════════════════════════════════════════════════════════════════
# Reading the RHI2 input file
# ══════════════════════════════════════════════════════════════════════════════

"""
    rhi2_filename_prefix(p) -> String

The part of the RHI2 file name that identifies p: its decimal string for
primes of at most 25 bits, and `"{bits}bit_{hash}"` otherwise, where `hash`
is eight hex digits derived from the decimal string.
"""
function rhi2_filename_prefix(pZ::BigInt)::String
    s    = string(pZ)
    bits = ndigits(pZ, base = 2)
    bits <= 25 && return s
    h = string(hash(s) & 0xffffffff; base = 16, pad = 8)
    return "$(bits)bit_$(h)"
end
rhi2_filename_prefix(p::Integer) = rhi2_filename_prefix(BigInt(p))

"""
    parse_gram_matrix(q_str) -> Matrix{BigInt}

Parse `c11*x1^2 + c12*x1*x2 + …` into the symmetric Gram matrix `A` with
`Q_A(x) = x'Ax`; an off-diagonal coefficient of the polynomial is `2·A[i,j]`.
Coefficients are read as `BigInt`, since for large p they exceed `Int64`.
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
        iseven(coeff) || error("odd off-diagonal coefficient $coeff for x$i*x$j")
        half    = coeff ÷ 2
        A[i, j] = half
        A[j, i] = half
    end
    return A
end

"""
    read_rhi2_file(p; dir=".", file=nothing) -> Vector{Matrix{BigInt}}

One 5×5 Gram matrix per "Type N" block of the file. Pass `file` to name the
input directly; otherwise `RHI2_{prefix}_{N}.txt` or `RHI2_{prefix}.txt` is
looked up in `dir`. Some files carry a second table after the first, headed
by a line of the form `p = <p> (…)`; reading stops there.
"""
function read_rhi2_file(p::Integer; dir::AbstractString = ".",
                        file::Union{Nothing,AbstractString} = nothing)
    pZ     = BigInt(p)
    prefix = rhi2_filename_prefix(pZ)

    path = if file !== nothing
        isfile(file) || error("input file not found: \"$file\"")
        file
    else
        cands = filter(readdir(dir; join = true)) do f
            occursin(Regex("RHI2_$(prefix)(_\\d+)?\\.txt\$"), f)
        end
        isempty(cands) && error("No RHI2_$(prefix)[_N].txt found in \"$dir\"")
        length(cands) > 1 && @warn "Multiple matches; using $(cands[1])"
        cands[1]
    end

    all_lines = readlines(path)
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
    current_q !== nothing && push!(matrices, parse_gram_matrix(current_q))

    isempty(matrices) && error("No form blocks parsed from $path")
    return matrices
end

# ══════════════════════════════════════════════════════════════════════════════
# Progress output
# ══════════════════════════════════════════════════════════════════════════════

# One timestamped line to stdout, flushed at once. Plain `println` rather than
# `@info`, because this output is read while a job runs and must stay one
# line per event.
function _progress(msg::AbstractString)
    println(Dates.format(Dates.now(), "HH:MM:SS"), "  ", msg)
    flush(stdout)
end

# Compact duration: 0.23s, 5.4s, 2m26s, 1h08m.
function _fmt_secs(s::Float64)
    s <  60 && return string(round(s, digits = 2), "s")
    s < 3600 && return string(floor(Int, s / 60), "m", lpad(round(Int, s % 60), 2, '0'), "s")
    return string(floor(Int, s / 3600), "h", lpad(round(Int, (s % 3600) / 60), 2, '0'), "m")
end

function _fmt_elapsed(secs::Float64)
    h = floor(Int, secs / 3600)
    m = floor(Int, (secs % 3600) / 60)
    s = secs % 60
    return "$(h)h $(m)m $(round(s, digits=1))s"
end

# ══════════════════════════════════════════════════════════════════════════════
# The results file
# ══════════════════════════════════════════════════════════════════════════════
#
# A `#` preamble recording the run's parameters (the file name carries only p,
# ℓ and a tag, and --report reads p and ℓ back from the preamble), the header
# line below, then one comma-separated row per finished form:
#
#   form      1-based index of the form in the RHI2 file (its "Type N").
#   status    OK | NO_K0_BELOW_CAP | NOT_IN_GENUS | ABANDONED (see the header).
#   in_genus  1 or 0, the outcome of the genus test. Expected to be 1.
#   lambda1   λ₁, a SQUARED length. 0 = unavailable.
#   aut       |Aut L|, or 0 for "not computed".
#   kmin      least k with ℓ^{2k} ≥ λ₁.
#   k0        least k primitively representing ℓ^{2k}; −1 = unknown. The
#             levels kmin, …, k0−1 were each enumerated to exhaustion and
#             found to hold no primitive vector.
#   witness   space-separated coordinates of the primitive vector realising
#             k0, in the ORIGINAL basis; empty when there is none.
#   secs      wall-clock seconds spent on this form.
#
# No field contains a comma, so splitting on commas is a correct parser.
# Comment lines `# S <i>` record the start of an attempt on form i; every
# reader skips `#` lines, so the whole run stays in one file.

const CSV_HEADER = "form,status,in_genus,lambda1,aut,kmin,k0,witness,secs"

"""
    csv_preamble(p, ell, nforms, batch, of, kmax_cap, tag) -> String

The `#` block written once at the head of a fresh results file.
"""
function csv_preamble(p::BigInt, ell::BigInt, nforms::Int, batch::Int, of::Int,
                      kmax_cap::Int, tag::AbstractString)
    return string(
        "# kzero.jl — least k with a primitive representation of ell^{2k}\n",
        "# p = ", p, "\n",
        "# ell = ", ell, "\n",
        "# det = 16p^2 = ", 16 * p^2, "\n",
        "# forms_in_file = ", nforms, "\n",
        "# tag = ", isempty(tag) ? "-" : tag, "\n",
        "# batch = ", batch, " of = ", of, "\n",
        "# kmax_cap = ", kmax_cap, "\n",
        "# generated = ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"), "\n")
end

"""
    read_preamble(path) -> Dict{String,String}

The `key = value` pairs from a results file's `#` preamble.
"""
function read_preamble(path::AbstractString)
    kv = Dict{String,String}()
    isfile(path) || return kv
    for line in eachline(path)
        startswith(line, "#") || break
        for m in eachmatch(r"(\w+)\s*=\s*(\S+)", line)
            kv[m.captures[1]] = m.captures[2]
        end
    end
    return kv
end

"""
    check_resume_compatible(path, expect)

Refuse to append to a results file whose preamble disagrees with the current
invocation. Otherwise restarting with a different --ell or --kmax-cap would
produce a file whose rows mean different things under one preamble.
"""
function check_resume_compatible(path::AbstractString, expect::Dict{String,String})
    kv = read_preamble(path)
    isempty(kv) && return
    bad = [(k, get(kv, k, "<absent>"), v) for (k, v) in expect
           if get(kv, k, v) != v]
    isempty(bad) && return
    error("refusing to resume \"$path\": it was written with " *
          join(["$k=$was (now $now)" for (k, was, now) in bad], ", ") *
          ". Rows already in that file are not comparable with the ones this " *
          "run would add. Use a different --tag, or --no-resume to start over.")
end

_witness_field(w) = w === nothing ? "" : join(string.(w), " ")

function _csv_row(i::Int, rec)
    return string(i, ",", rec.status, ",", rec.in_genus ? 1 : 0, ",",
                  rec.lambda1, ",", rec.aut, ",", rec.kmin, ",", rec.k0, ",",
                  _witness_field(rec.witness), ",", round(rec.secs, digits=3))
end

"""
    completed_forms(csvpath) -> Set{Int}

Form indices already recorded in `csvpath`, whatever their status.

A row counts only if it has the full number of fields. A process killed while
writing leaves a fragment such as `7,OK,1,99`, which would otherwise mark form
7 as done forever; requiring the field count makes the fragment ignored and
the form redone. The fragment stays in the file and the report counts it as an
unparsable line.
"""
function completed_forms(csvpath::AbstractString)::Set{Int}
    ncommas = count(==(','), CSV_HEADER)
    done = Set{Int}()
    isfile(csvpath) || return done
    for line in eachline(csvpath)
        (isempty(line) || startswith(line, "#") || startswith(line, "form,")) &&
            continue
        count(==(','), line) == ncommas || continue
        c = findfirst(',', line)
        c === nothing && continue
        i = tryparse(Int, line[1:c-1])
        i === nothing || push!(done, i)
    end
    return done
end

"""
    attempt_counts(csvpath) -> Dict{Int,Int}

How many times each form has been started, from the `# S <i>` lines.
Completions are not subtracted: the caller consults this only for forms that
`completed_forms` has not excluded, so a surviving count means "started and
never finished".
"""
function attempt_counts(csvpath::AbstractString)::Dict{Int,Int}
    n = Dict{Int,Int}()
    isfile(csvpath) || return n
    for line in eachline(csvpath)
        startswith(line, "# S ") || continue
        i = tryparse(Int, strip(line[5:end]))
        i === nothing || (n[i] = get(n, i, 0) + 1)
    end
    return n
end

# ══════════════════════════════════════════════════════════════════════════════
# Running one batch of forms
# ══════════════════════════════════════════════════════════════════════════════

"""
    run_batch(p; ell, dir, file, batch, of, limit, kmax_cap,
              want_aut, resume, tag, max_attempts) -> outpath

Process this batch's share of the RHI2 file — the forms with index
i ≡ batch+1 (mod of) — appending one row per form, and return the path
written.

`limit` restricts the batch to its first `limit` forms BEFORE the restart
filter, so rerunning with the same limit finishes the same fixed set rather
than advancing through the file. It is for calibration only.

Preconditions are checked once, up front: a run that should not have started
must not leave a partially valid results file behind.
"""
function run_batch(p::Integer;
                   ell::Integer,
                   dir::AbstractString = ".",
                   file::Union{Nothing,AbstractString} = nothing,
                   batch::Int = 0, of::Int = 1,
                   limit::Union{Nothing,Int} = nothing,
                   kmax_cap::Union{Nothing,Int} = nothing,
                   want_aut::Bool = true, resume::Bool = true,
                   tag::AbstractString = "",
                   max_attempts::Int = DEFAULT_MAX_ATTEMPTS)
    pZ   = BigInt(p)
    ellZ = BigInt(ell)

    mod(pZ, 12) == 11 || error(
        "p = $pZ is $(mod(pZ,12)) mod 12; the RHI2 files are for p = 11 mod 12")
    ellZ != pZ || error("ell must differ from p")
    is_probable_prime(ZZ(ellZ)) || error("ell = $ellZ is not prime")
    0 <= batch < of || error("need 0 <= batch < of, got batch=$batch of=$of")

    pdisp = rhi2_filename_prefix(pZ)
    @info "Reading RHI2 file for p = $pZ …"
    matrices = read_rhi2_file(pZ; dir, file)
    nforms   = length(matrices)
    @info "  Found $nforms forms; batch $batch of $of."

    ref = Qp_lattice(pZ)
    det(gram_matrix(ref)) == 16 * ZZ(pZ)^2 ||
        error("reference form A_p does not have det 16p^2 — check Qp_lattice")

    cap = kmax_cap === nothing ? default_kmax_cap(pZ, ellZ) : kmax_cap
    k0_heur = natural_log_bigint(pZ) / (3 * log(Float64(ellZ)))
    @info "  Heuristic k0 ≈ $(round(k0_heur, digits=2)); scan gives up at k = $cap" *
          (kmax_cap === nothing ? " (default: 2·ln p / (3·ln ell))" : " (--kmax-cap)")
    cap > k0_heur || @warn "  kmax-cap $cap is at or below the heuristic k0 " *
                           "$(round(k0_heur, digits=2)); most forms will report " *
                           "NO_K0_BELOW_CAP and the run will cost a great deal " *
                           "for nothing."

    # The tag goes in the file name, so that two runs over different form sets
    # (a calibration subsample and a full run, say) never share a file.
    parts   = String[]
    isempty(tag) || push!(parts, tag)
    of == 1      || push!(parts, "batch$(batch)of$(of)")
    suffix  = isempty(parts) ? "" : "_" * join(parts, "_")
    outpath = joinpath(dir, "kzero_p$(pdisp)_ell$(ellZ)$(suffix).csv")

    expect = Dict("p" => string(pZ), "ell" => string(ellZ),
                  "kmax_cap" => string(cap), "of" => string(of),
                  "batch" => string(batch))
    if resume
        check_resume_compatible(outpath, expect)
    elseif isfile(outpath)
        @warn "  --no-resume: truncating existing $outpath"
        open(outpath, "w") do io; end
    end

    done     = resume ? completed_forms(outpath) : Set{Int}()
    attempts = resume ? attempt_counts(outpath)  : Dict{Int,Int}()
    isempty(done) || @info "  Resuming: $(length(done)) form(s) already recorded."

    mine = [i for i in 1:nforms if (i - 1) % of == batch]
    limit === nothing || (mine = mine[1:min(limit, length(mine))])

    # Counted over this batch's own list: under --limit the file can hold
    # rows for forms outside `mine`, so `length(mine) - length(done)` can be
    # negative.
    mine_todo = count(i -> !(i in done), mine)
    @info "  $mine_todo form(s) to do in this batch."
    of == 1 && mine_todo > 500 && @warn "  --of 1: all $mine_todo forms will run in this " *
        "one process, and nothing here is parallel within a process. Split the " *
        "work with --batch j --of N across independent jobs."

    newly    = 0
    statuses = Dict{String,Int}()
    k0s      = Int[]
    t_start  = time()
    ndone    = 0                   # rows written this run, ABANDONED included
    open(outpath, "a") do io
        if position(io) == 0
            print(io, csv_preamble(pZ, ellZ, nforms, batch, of, cap, tag))
            println(io, CSV_HEADER)
            flush(io)
        end
        for i in mine
            i in done && continue

            att = get(attempts, i, 0)
            if att >= max_attempts
                @warn "  form $i abandoned after $att attempt(s). If those were " *
                      "external interruptions rather than a pathological form, " *
                      "raise --max-attempts (currently $max_attempts) and delete " *
                      "its ABANDONED row."
                println(io, string(i, ",ABANDONED,,,,,-1,,")); flush(io)
                statuses["ABANDONED"] = get(statuses, "ABANDONED", 0) + 1
                ndone += 1
                continue
            end
            att == max_attempts - 1 &&
                @warn "  form $i is on its final attempt ($(att + 1) of $max_attempts)"

            label = "form $i" * (att > 0 ? " (attempt $(att + 1))" : "")
            println(io, "# S $i"); flush(io)
            rec = study_form(matrices[i], ellZ, pZ, ref;
                             kmax_cap = cap, want_aut, label)
            println(io, _csv_row(i, rec)); flush(io)

            newly += 1
            ndone += 1
            statuses[rec.status] = get(statuses, rec.status, 0) + 1
            rec.k0 > 0 && push!(k0s, rec.k0)
            _progress("$label DONE  k0=$(rec.k0)  $(_fmt_secs(rec.secs))  " *
                      "[$newly this run, $(mine_todo - ndone) left, " *
                      "elapsed $(_fmt_elapsed(time() - t_start))]")
        end
    end

    _print_run_summary(newly, ndone, length(mine), length(mine) - mine_todo,
                       mine_todo, statuses, k0s, time() - t_start, outpath)
    return outpath
end

"""
    _print_run_summary(newly, ndone, assigned, resumed, outstanding, statuses,
                       k0s, elapsed, outpath)

End-of-run summary: did the batch finish, did anything need inspection, and
what do the k0 values look like so far. `ndone` counts ABANDONED rows as well
as computed ones, so that a finished batch reports itself finished.
"""
function _print_run_summary(newly::Int, ndone::Int, assigned::Int, resumed::Int,
                            outstanding::Int, statuses::Dict{String,Int},
                            k0s::Vector{Int}, elapsed::Float64,
                            outpath::AbstractString)
    remaining = outstanding - ndone
    println()
    println("─"^72)
    @printf("Run complete: %d form(s) this run in %s\n", newly,
            _fmt_elapsed(elapsed))
    @printf("  assigned to this batch %6d\n", assigned)
    @printf("  already recorded       %6d\n", resumed)
    @printf("  remaining              %6d%s\n", remaining,
            remaining == 0 ? "   (batch complete)" :
                             "   — resubmit to continue")
    if !isempty(statuses)
        println("  statuses this run:")
        for s in sort(collect(keys(statuses)))
            marker = s == "OK" ? "" : "   <-- inspect"
            @printf("    %-18s %6d%s\n", s, statuses[s], marker)
        end
    end
    if !isempty(k0s)
        sorted = sort(k0s)
        @printf("  k0 this run: min %d  median %d  max %d  (mean %.2f)\n",
                sorted[1], sorted[cld(end, 2)], sorted[end],
                sum(sorted) / length(sorted))
        for k in sorted[1]:sorted[end]
            c = count(==(k), sorted)
            c == 0 && continue
            @printf("    k0 = %2d  %5d  %s\n", k, c, "█"^min(50, c))
        end
    end
    newly > 0 && @printf("  mean %.1f s/form → %.1f core-hours per 1000 forms\n",
                         elapsed / newly, elapsed / newly * 1000 / 3600)
    println("  results: ", relpath(outpath))
    println("─"^72)
end

# ══════════════════════════════════════════════════════════════════════════════
# Reading <p> from the command line
# ══════════════════════════════════════════════════════════════════════════════

"""
    parse_prime_expr(s) -> BigInt

`s` as a non-negative integer or an arithmetic expression over `+ - * ^ ( )`,
e.g. `"2^11*3^24-1"`. The expression is parsed by `Meta.parse` and evaluated
by walking the syntax tree in `BigInt`; Julia's `eval` is never called, so
nothing beyond the four operators can run.
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
                exp < 0 && error("negative exponent in \"$s\"")
                exp > typemax(Int) && error("exponent too large in \"$s\"")
                return base ^ Int(exp)
            else
                error("unsupported operator \"$op\" in \"$s\"")
            end
        else
            error("unsupported syntax in \"$s\" (node: $node)")
        end
    end

    return eval_node(ast)
end


# ══════════════════════════════════════════════════════════════════════════════
#                                  REPORTING
# ══════════════════════════════════════════════════════════════════════════════
#
# `kzero.jl --report <csv>...` reads results files written by this program and
# describes the k0 data. The batches of one run may be listed or globbed; their
# rows are pooled, since every section wants the whole population and no
# single batch is a representative sample of it. p and ℓ are read from the
# preambles, and files that disagree are refused rather than pooled. Section 0
# checks coverage against `forms_in_file`, so a missing batch is reported.
#
# Sections 1 and 2 compare data against the Siegel heuristics of item (2) in
# the header, which suppress the local densities and the level dependence of
# the cusp bound. A mismatch there is information about the heuristic, not
# evidence of a bug. Section 0 is bookkeeping only: with no shell counts
# recorded there is no theorem to check the rows against; what guarantees a k0
# is that its witness was re-verified before the row was written.

# ══════════════════════════════════════════════════════════════════════════════
# The Siegel constant and the law the k0 distribution is compared against
# ══════════════════════════════════════════════════════════════════════════════
#
# For a positive definite quinary lattice of determinant D, Siegel's formula
# reads r(gen L, m) = (π^{5/2}/Γ(5/2)) · m^{3/2}/√D · Π_ν β_ν(L,m). With
# Γ(5/2) = (3/4)√π and √D = 4p this is
#
#     r(gen L, m) ≈ SIEGEL_C · m^{3/2} / p,     SIEGEL_C = π²/3 ≈ 3.2899,
#
# counting ±x separately, under the heuristic Π_ν β_ν ≈ 1.
const SIEGEL_C = π^2 / 3

# Write m = ℓ^{2k} and μ_k = r*(gen L, m). By the identity
# r*(m) = r(m) − r(m/ℓ²) and Siegel,
#
#     μ_k ≈ (SIEGEL_C/p)(1 − ℓ^{−3}) ℓ^{3k}.
#
# Model "class L has a primitive vector of norm ℓ^{2k}" as Poisson with that
# mean, independently across k. Vectors come in pairs ±x and a pair is one
# opportunity, so the intensity is μ_k/2, and
#
#     P(k0 > k) ≈ exp( −½ Σ_{j≤k} μ_j ).
#
# The geometric sum Σ_{j≤k} ℓ^{3j} ≈ ℓ^{3k}/(1 − ℓ^{−3}) cancels the
# primitivity factor, leaving
#
#     P(k0 ≤ k) ≈ 1 − exp( −SIEGEL_C · ℓ^{3k} / (2p) ).
#
# So the coefficient to fit is SIEGEL_C/2 = π²/6 ≈ 1.645, and the predicted
# median solves SIEGEL_C·ℓ^{3k}/(2p) = log 2. Neither is a theorem: the
# independence across k is an assumption.
const POISSON_C = SIEGEL_C / 2

# The law and its inverse: `poisson_cdf` is P(k0 ≤ k); `predicted_k_at`
# solves it for k at a given probability q, giving the level by which a
# fraction q of classes is expected to have represented. q = 1/2 is the
# predicted median.
poisson_cdf(ell::Int, k::Int, p::BigInt) = 1 - exp(-POISSON_C * Float64(ell)^(3k) / Float64(p))
predicted_k_at(ell::Int, p::BigInt, q::Float64) =
    log(-Float64(p) * log(1 - q) / POISSON_C) / (3 * log(ell))

# ══════════════════════════════════════════════════════════════════════════════
# Reading a results file back
# ══════════════════════════════════════════════════════════════════════════════

"One row of a results file; the fields are the columns documented above."
struct Row
    form::Int
    status::String
    in_genus::Bool
    lambda1::BigInt
    aut::BigInt
    kmin::Int
    k0::Int
    secs::Float64
end

"Run parameters recovered from a results file's preamble."
struct RunMeta
    p::BigInt
    ell::Int
end

"""
    read_meta(path) -> Union{RunMeta,Nothing}

p and ℓ from the `#` preamble; `nothing` if the file has none.
"""
function read_meta(path::AbstractString)
    kv = read_preamble(path)
    p   = haskey(kv, "p")   ? tryparse(BigInt, kv["p"])  : nothing
    ell = haskey(kv, "ell") ? tryparse(Int,    kv["ell"]) : nothing
    (p === nothing || ell === nothing) && return nothing
    return RunMeta(p, ell)
end

"""
    read_rows(path) -> (rows, tally, ids)

Every data row in the file, a tally by status (with `"__unparsable__"` for
lines that could not be read), and the set of form indices seen — ABANDONED
rows included, since those are recorded rather than missing. No filtering is
done here; each section says what it excludes.
"""
function read_rows(path::AbstractString)
    rows  = Row[]
    tally = Dict{String,Int}()
    ids   = Set{Int}()
    for line in eachline(path)
        (isempty(line) || startswith(line, "#") || startswith(line, "form,")) && continue
        f = split(line, ',')
        if length(f) < 9 || tryparse(Int, f[1]) === nothing
            tally["__unparsable__"] = get(tally, "__unparsable__", 0) + 1
            continue
        end
        status = String(f[2])
        tally[status] = get(tally, status, 0) + 1
        push!(ids, parse(Int, f[1]))
        # ABANDONED rows have empty numeric fields; they are counted only.
        status == "ABANDONED" && continue
        push!(rows, Row(parse(Int, f[1]), status, f[3] == "1",
                        parse(BigInt, f[4]), parse(BigInt, f[5]),
                        parse(Int, f[6]), parse(Int, f[7]),
                        parse(Float64, f[9])))
    end
    return rows, tally, ids
end

# ══════════════════════════════════════════════════════════════════════════════
# 0. Where the data came from, and what was checked
# ══════════════════════════════════════════════════════════════════════════════

"""
    scanned_rows(rows) -> Vector{Row}

The rows that carry measurements. A NOT_IN_GENUS row was rejected before any
enumeration, so its numeric columns are placeholders; sections 1–3 work from
this list.
"""
scanned_rows(rows::Vector{Row}) = [r for r in rows if r.in_genus]

function report_integrity(rows::Vector{Row}, tally::Dict{String,Int},
                          ids::Set{Int}, meta::RunMeta, files::Vector{String})
    total = sum(values(tally))
    println("0. Data and integrity checks")
    @printf("   p   = %s  (%d bits)\n", meta.p, ndigits(meta.p, base = 2))
    @printf("   ell = %d,  det = 16p^2,  targets are ell^{2k}\n", meta.ell)
    @printf("   %d data row(s) in %d file(s); sections 1-3 use the %d scanned\n",
            total, length(files), count(r -> r.in_genus, rows))
    println("   rows by status:")
    for s in sort(collect(keys(tally)))
        note = s == "OK"              ? "" :
               s == "NOT_IN_GENUS"    ? "   <-- not in genus(A_p); not scanned" :
               s == "NO_K0_BELOW_CAP" ? "   <-- k0 > kmax_cap; raise the cap and resume" :
               s == "ABANDONED"       ? "   <-- never finished; no mathematical content" :
               s == "__unparsable__"  ? "   <-- damaged or truncated lines" : ""
        @printf("     %-18s %6d%s\n", s, tally[s], note)
    end

    dupes = Int[]
    seen  = Set{Int}()
    for r in rows
        r.form in seen ? push!(dupes, r.form) : push!(seen, r.form)
    end
    isempty(dupes) || @printf("   *** %d DUPLICATE form index/indices: %s\n",
                              length(dupes), join(first(dupes, 10), ", "))

    # Coverage: a batch that never ran leaves no trace in the rows present,
    # so compare against the form count recorded in the preamble.
    nfile = 0
    for f in files
        kv = read_preamble(f)
        haskey(kv, "forms_in_file") &&
            (nfile = max(nfile, something(tryparse(Int, kv["forms_in_file"]), 0)))
    end
    covered = length(ids)
    if nfile > 0
        missing_n = nfile - covered
        @printf("   coverage: %d / %d forms%s\n", covered, nfile,
                missing_n == 0 ? "  — complete" : "")
        if missing_n > 0
            gaps = [i for i in 1:nfile if !(i in ids)]
            @printf("   *** %d FORM(S) MISSING, e.g. %s\n", missing_n,
                    join(first(gaps, 12), ", "))
            println("   *** Either batches are still running, or the job array")
            println("   *** size disagrees with --of. Do not treat this as a result.")
        end
    end

    sc = scanned_rows(rows)
    misordered = count(r -> r.k0 > 0 && r.k0 < r.kmin, sc)
    @printf("   k0 >= kmin on every scanned row: %s\n",
            misordered == 0 ? "consistent" :
            "*** $misordered row(s) with k0 < kmin — BUG")
    incomplete = (nfile > 0 && covered < nfile) ? 1 : 0
    println()
    return misordered + length(dupes) + incomplete
end

# ══════════════════════════════════════════════════════════════════════════════
# 1. The k0 distribution
# ══════════════════════════════════════════════════════════════════════════════

function report_k0(rows::Vector{Row}, meta::RunMeta)
    ell, p = meta.ell, meta.p
    ks = sort([r.k0 for r in rows if r.k0 > 0])
    println("1. k0 distribution  (least k primitively representing ell^{2k})")
    if isempty(ks)
        println("   no k0 values in this data\n"); return nothing
    end
    n  = length(ks)
    lo, hi = ks[1], ks[end]
    @printf("   %d classes:  min %d   median %d   mean %.2f   max %d\n",
            n, lo, ks[cld(n, 2)], sum(ks)/n, hi)
    @printf("   predicted median (HEURISTIC, Siegel + Poisson): %.2f\n",
            predicted_k_at(ell, p, 0.5))
    println()
    println("   Empirical CDF against  P(k0 <= k) = 1 - exp(-(pi^2/6)·ell^{3k}/p):")
    println("        k   count      cum    empirical   predicted      diff")
    maxdev = 0.0
    for k in lo:hi
        c   = count(==(k), ks)
        cum = count(<=(k), ks)
        e   = cum/n
        q   = poisson_cdf(ell, k, p)
        maxdev = max(maxdev, abs(e - q))
        @printf("      %3d  %7d  %7d     %8.4f    %8.4f  %+8.4f\n", k, c, cum, e, q, e - q)
    end
    @printf("\n   max |empirical - predicted| = %.4f", maxdev)
    # The Kolmogorov 95% band is a scale to read the deviation against, not a
    # valid test: the law has a fitted constant and the classes are not
    # independent.
    @printf("   (1.36/sqrt(n) = %.4f)\n", 1.36/sqrt(n))

    # Least squares through the origin on −log(1−F), which linearises the law.
    # Levels with F = 0 or 1 carry no information and are dropped.
    num = 0.0; den = 0.0; used = 0
    for k in lo:hi
        f = count(<=(k), ks)/n
        (0 < f < 1) || continue
        x = Float64(ell)^(3k) / Float64(p)
        num += x * (-log(1 - f)); den += x*x; used += 1
    end
    if den > 0
        @printf("   fitted coefficient = %.3f from %d level(s)   (predicted pi^2/6 = %.3f, ratio %.2f)\n",
                num/den, used, POISSON_C, (num/den)/POISSON_C)
        println("   A ratio near 1 means the Siegel average is the right scale AND the")
        println("   Poisson independence across k is not badly wrong. This fit cannot")
        println("   separate the two.")
    end

    # The law read backwards: the level by which a fraction q of classes is
    # expected to have represented, against the level at which q of them had.
    # The empirical column cannot see a class rarer than 1/n; the predicted
    # column extrapolates past that at the cost of assuming the law.
    println()
    println("   Bound: least k capturing a given fraction of classes")
    println("        fraction   empirical k   predicted k")
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999)
        # Skip quantiles finer than the sample resolves: at least one class
        # must lie above q. The 1e-9 absorbs floating-point error in 1 − q.
        (1 - q) * n + 1e-9 < 1 && continue
        emp = findfirst(k -> count(<=(k), ks) / n >= q, lo:hi)
        @printf("      %8s  %12s   %11.2f\n",
                string(round(100q, digits = 2), "%"),
                emp === nothing ? "> $hi" : string(lo + emp - 1),
                predicted_k_at(ell, p, q))
    end
    @printf("   observed max k0 = %d over %d class(es)\n", hi, n)
    @printf("   --kmax-cap %d would have held every class in this data\n", hi + 1)
    println("   That is a measurement, not a guarantee: a sample of $n cannot")
    println("   see a class rarer than 1 in $n. The predicted column is what")
    println("   extrapolates past the sample, at the cost of assuming the law.")
    println()
    return ks
end

# ══════════════════════════════════════════════════════════════════════════════
# 2. Covariates: does anything cheap predict k0?
# ══════════════════════════════════════════════════════════════════════════════
#
# Reported with counts and standard deviations, never as a bare mean: an
# association visible in an aggregate can dissolve once the data are
# stratified. Anything here is a prompt to stratify, not a result.

_msd(v) = isempty(v) ? (NaN, NaN) : begin
    m = sum(v)/length(v)
    (m, length(v) > 1 ? sqrt(sum((x - m)^2 for x in v)/(length(v) - 1)) : 0.0)
end

function report_covariates(rows::Vector{Row}, meta::RunMeta)
    println("2. k0 against covariates")
    det5 = exp(log(16*Float64(meta.p)^2)/5)
    @printf("   det^(1/5) = %.4g  (the Minkowski scale for lambda1)\n", det5)

    byk = Dict{Int,Vector{Float64}}()
    for r in rows
        r.k0 > 0 && push!(get!(byk, r.k0, Float64[]), Float64(r.lambda1)/det5)
    end
    if !isempty(byk)
        println("       k0   classes    mean lambda1/det^(1/5)        sd")
        for k in sort(collect(keys(byk)))
            m, s = _msd(byk[k])
            @printf("      %3d   %7d   %22.4f   %7.4f\n", k, length(byk[k]), m, s)
        end
        println("   A trend here would say short lattices represent sooner, which is the")
        println("   λ₁ barrier showing up statistically. Compare the spread between rows")
        println("   against the sd within them before believing it.")
    end

    byaut = Dict{BigInt,Vector{Int}}()
    for r in rows
        r.k0 > 0 && r.aut > 0 && push!(get!(byaut, r.aut, Int[]), r.k0)
    end
    if !isempty(byaut)
        println("\n    |Aut|   classes    mean k0        sd")
        for a in sort(collect(keys(byaut)))
            m, s = _msd(Float64.(byaut[a]))
            @printf("     %6d   %7d   %8.3f   %7.4f\n", a, length(byaut[a]), m, s)
        end
        println("   |Aut| > 2 classes are rare and carry more mass each, so they are")
        println("   where a mass-formula discrepancy would concentrate.")
    end
    println()
    return byk
end

# ══════════════════════════════════════════════════════════════════════════════
# 3. Cost
# ══════════════════════════════════════════════════════════════════════════════

function report_cost(rows::Vector{Row})
    secs = [r.secs for r in rows]
    isempty(secs) && return nothing
    s = sort(secs)
    n = length(s)
    println("3. Cost")
    @printf("   %d forms, %.2f core-hours total, %.1f s/form mean, %.1f s median\n",
            n, sum(s)/3600, sum(s)/n, s[cld(n, 2)])
    @printf("   quartiles %.1f / %.1f / %.1f s,  max %.1f s (%.0fx the median)\n",
            s[max(1, cld(n,4))], s[cld(n,2)], s[max(1, cld(3n,4))], s[end],
            s[cld(n,2)] > 0 ? s[end]/s[cld(n,2)] : NaN)
    slow = sort(rows; by = r -> -r.secs)
    println("   slowest forms: ",
            join(["$(r.form) ($(round(r.secs, digits=1))s, k0=$(r.k0))"
                  for r in first(slow, min(5, n))], ", "))
    println()
    return sum(s)
end

# ══════════════════════════════════════════════════════════════════════════════
# The report as a whole
# ══════════════════════════════════════════════════════════════════════════════

"""
    report_output_path(files) -> String

Where --report saves its transcript: the first input's name with `.csv`
replaced by `.txt` and any trailing `_batch{j}of{N}` removed, since the report
covers every batch of the run:

    kzero_p251_ell2.csv                  ->  kzero_p251_ell2.txt
    kzero_p251_ell2_main_batch3of50.csv  ->  kzero_p251_ell2_main.txt
"""
function report_output_path(files::Vector{String})
    name = basename(first(files))
    name = replace(name, r"\.csv$"i => "")
    name = replace(name, r"_batch\d+of\d+$" => "")
    return joinpath(dirname(first(files)), name * ".txt")
end

"""
    report_main(args) -> Int

Entry point for `kzero.jl --report <csv>...`. Returns 0 if every integrity
check in section 0 passed and 2 otherwise; that number becomes the process's
exit status, so a run can be checked from a shell script.
"""
function report_main(args::Vector{String})
    if isempty(args) || args[1] in ("-h", "--help")
        println(stderr, """
Usage: julia --startup-file=no kzero.jl --report <csv> [<csv> ...]

Reads one run's results files and reports the k0 structure. List or glob the
batches of a run; their rows are pooled into a single report, and a form
appearing in two files is flagged as a duplicate rather than counted twice.
p and ell come from the `#` preamble of the files, which must all agree.

The transcript is printed and also saved as a .txt beside the first input,
named after the run rather than the batch:

    ..._ell2_main_batch3of50.csv  ->  ..._ell2_main.txt

Exit status is 0 if every integrity check in section 0 passed, 2 if any failed.

For a results file with no preamble, supply p and ell in the environment:
  KZERO_P=251 KZERO_ELL=2 julia ... kzero.jl --report file.csv
""")
        return 1
    end

    rows  = Row[]
    tally = Dict{String,Int}()
    ids   = Set{Int}()
    files = String[]
    meta  = nothing

    for f in args
        if !isfile(f)
            println(stderr, "skipping missing $f"); continue
        end
        push!(files, f)
        m = read_meta(f)
        if m !== nothing
            if meta === nothing
                meta = m
            elseif meta.p != m.p || meta.ell != m.ell
                println(stderr, "ERROR: $f is p=$(m.p) ell=$(m.ell) but an earlier " *
                                "file is p=$(meta.p) ell=$(meta.ell). Refusing to pool " *
                                "unrelated runs.")
                return 1
            end
        end
        r, t, idset = read_rows(f)
        append!(rows, r)
        union!(ids, idset)
        for (k, v) in t; tally[k] = get(tally, k, 0) + v; end
    end

    if meta === nothing
        # No preamble, and p cannot be recovered from the file name for large
        # primes, so ask rather than guess.
        ps = get(ENV, "KZERO_P", ""); es = get(ENV, "KZERO_ELL", "")
        p  = tryparse(BigInt, ps); ell = tryparse(Int, es)
        if p === nothing || ell === nothing
            println(stderr, """
ERROR: no `#` preamble in these files and KZERO_P / KZERO_ELL are not set.
Set them explicitly, e.g.
  KZERO_P=251 KZERO_ELL=2 julia ... kzero.jl --report $(args[1])
""")
            return 1
        end
        meta = RunMeta(p, ell)
    end

    if isempty(rows)
        println(stderr, "no usable rows in $(length(files)) file(s)")
        return 1
    end

    sc = scanned_rows(rows)

    # The sections print to stdout; the transcript is captured by redirecting
    # stdout to a temporary file, then both printed and saved.
    problems = Ref(0)
    tmppath, tmpio = mktemp()
    try
        redirect_stdout(tmpio) do
            problems[] = report_integrity(rows, tally, ids, meta, files)
            report_k0(sc, meta)
            report_covariates(sc, meta)
            report_cost(sc)
            problems[] == 0 ||
                println("*** $(problems[]) integrity problem(s) — see section 0 " *
                        "before using any of the above.")
        end
    finally
        close(tmpio)
    end
    text = read(tmppath, String)
    rm(tmppath; force = true)

    print(stdout, text)

    outpath = report_output_path(files)
    try
        write(outpath, text)
        println("\nreport saved to ", relpath(outpath))
    catch e
        # The report has already been printed; failing to save it is not a
        # reason to fail the run.
        @warn "could not write $outpath: $e"
    end

    return problems[] == 0 ? 0 : 2
end

# ══════════════════════════════════════════════════════════════════════════════
# Command-line entry point
# ══════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    let
        SCRIPT = basename(@__FILE__)

        function usage()
            println(stderr, """
Usage: julia --startup-file=no $(SCRIPT) <p> --ell <ell> [options]

  <p>                 prime to process (required). Plain integer or an
                      arithmetic expression over + - * ^ ( ), e.g.
                      2^11*3^24-1. Must satisfy p = 11 mod 12.
  --ell <ell>         prime ell; the targets are ell^{2k}  (required)
  --dir <dir>         input/output directory (default: script directory)
  --file <path>       read this RHI2 file directly, bypassing name lookup
  --batch <j> --of <N>
                      process only the forms whose 1-based index i satisfies
                      i = j+1 (mod N). Batches are independent processes and
                      share nothing; this is the only form of parallelism.
  --limit <n>         restrict this batch to its first n forms. For
                      calibration only: rerunning finishes that same fixed
                      set rather than advancing through the file.
  --kmax-cap <k>      never scan beyond this level. Default
                      ceil(2 ln p / (3 ln ell)), twice the heuristic k0.
                      Reaching it records NO_K0_BELOW_CAP, meaning k0 > cap,
                      not that the class fails to represent.
  --tag <name>        put <name> in the output file name. Use it whenever two
                      runs over different form sets share a directory.
  --max-attempts <n>  give up on a form after n interrupted starts
                      (default: 3). Raise this if interruptions are external
                      (a wall-clock limit, say) rather than the form's fault.
  --no-aut            skip |Aut L|. It costs ~1 ms per form and is the only
                      covariate of k0 besides lambda1, so there is little
                      reason to set this.
  --no-resume         truncate any existing results file and recompute from
                      scratch
  --report <csv>...   read finished results files and print the analysis, then
                      exit. p and ell come from the files' own preambles. List
                      or glob a run's batches to pool them into one report; the
                      transcript is also saved beside the first input as .txt.
  -h, --help          print this and exit

Output: kzero_p{prefix}_ell{ell}[_{tag}][_batch{j}of{N}].csv — a `#` preamble
recording p, ell and the run parameters, then one row per form, appended and
written to disk immediately. Rerunning the same command resumes; it is safe to
do so repeatedly.

Examples:
  # every form of RHI2_251.txt in this directory, targets 4^k
  julia --startup-file=no $(SCRIPT) 251 --ell 2
  # a large prime, split into fifty independent batches (this is batch 3)
  julia --startup-file=no $(SCRIPT) 2^11*3^24-1 --ell 2 --tag main --batch 3 --of 50
  # the report, pooling all fifty batches into one transcript
  julia --startup-file=no $(SCRIPT) --report kzero_p50bit_*_ell2_main_batch*of50.csv
""")
            exit(1)
        end

        (length(ARGS) < 1 || ARGS[1] in ("-h", "--help", "help")) && usage()

        # --report reads finished results files and needs no <p>.
        if ARGS[1] == "--report"
            exit(report_main(String[a for a in ARGS[2:end]]))
        end

        local p = parse_prime_expr(ARGS[1])
        p > 0 || error("parsed \"$(ARGS[1])\" as $p, which is not positive")

        local dir          = dirname(@__FILE__)
        local file         = nothing
        local ell          = nothing
        local batch        = 0
        local of           = 1
        local limit        = nothing
        local kmax_cap     = nothing   # nothing => default_kmax_cap(p, ell)
        local want_aut     = true
        local resume       = true
        local tag          = ""
        local max_att      = DEFAULT_MAX_ATTEMPTS

        local i = 2
        while i <= length(ARGS)
            flag = ARGS[i]
            need() = (i + 1 > length(ARGS) &&
                      (println(stderr, "Error: $flag requires an argument"); usage()))
            if flag == "--ell"
                need()
                v = tryparse(Int, ARGS[i+1])
                (v === nothing || v < 2 || !is_probable_prime(v)) &&
                    (println(stderr, "Error: --ell requires a prime ≥ 2"); usage())
                ell = BigInt(v); i += 2
            elseif flag == "--dir"
                need(); dir = ARGS[i+1]; i += 2
            elseif flag == "--file"
                need(); file = ARGS[i+1]; i += 2
            elseif flag == "--batch"
                need()
                batch = something(tryparse(Int, ARGS[i+1]), -1)
                batch < 0 && (println(stderr, "Error: --batch needs a non-negative integer"); usage())
                i += 2
            elseif flag == "--of"
                need()
                of = something(tryparse(Int, ARGS[i+1]), 0)
                of < 1 && (println(stderr, "Error: --of needs a positive integer"); usage())
                i += 2
            elseif flag == "--limit"
                need()
                limit = something(tryparse(Int, ARGS[i+1]), 0)
                limit < 1 && (println(stderr, "Error: --limit needs a positive integer"); usage())
                i += 2
            elseif flag == "--kmax-cap"
                need()
                kmax_cap = something(tryparse(Int, ARGS[i+1]), 0)
                kmax_cap < 1 && (println(stderr, "Error: --kmax-cap needs a positive integer"); usage())
                i += 2
            elseif flag == "--tag"
                need()
                tag = ARGS[i+1]
                occursin(r"^[A-Za-z0-9._-]+$", tag) ||
                    (println(stderr, "Error: --tag must be filename-safe " *
                             "([A-Za-z0-9._-]), got \"$tag\""); usage())
                i += 2
            elseif flag == "--max-attempts"
                need()
                max_att = something(tryparse(Int, ARGS[i+1]), 0)
                max_att < 1 && (println(stderr, "Error: --max-attempts needs a positive integer"); usage())
                i += 2
            elseif flag == "--no-aut"
                want_aut = false; i += 1
            elseif flag == "--no-resume"
                resume = false; i += 1
            else
                println(stderr, "Error: unknown argument \"$flag\""); usage()
            end
        end

        ell === nothing && (println(stderr, "Error: --ell <prime> is required"); usage())
        isdir(dir) || (println(stderr, "Error: directory not found: \"$dir\""); exit(1))

        @info "Starting" p ell tag batch of limit kmax_cap want_aut resume max_att
        out = run_batch(p; ell, dir, file, batch, of, limit,
                        kmax_cap, want_aut, resume, tag,
                        max_attempts = max_att)
        println("Done → ", relpath(out))
    end
end
