# simon.jl — squares primitively represented by a positive definite quinary form
#
# For each positive definite integral quinary form A in an input file, this
# script finds a primitive vector x ∈ ℤ⁵ and an integer N ≥ 1 with
#
#     Q_A(x) = x' A x = N²,
#
# by locating an isotropic vector of the indefinite senary form
#
#     F = 2A ⊥ ⟨-2⟩
#
# with Denis Simon's algorithm ("Solving quadratic equations using reduced
# unimodular quadratic forms", Math. Comp. 74 (2005), and the later note
# "Quadratic equations in dimensions 4, 5 and more"), as implemented in Hecke,
# the number-theory library inside Oscar.
#
# ─── Notation ────────────────────────────────────────────────────────────────
#
#   A              a symmetric 5×5 integer matrix, positive definite; the
#                  quinary form is Q_A(x) = x'Ax for a row vector x ∈ ℤ⁵.
#   primitive      a nonzero integer vector whose entries have gcd 1.
#   F              the 6×6 matrix 2A ⊥ ⟨-2⟩, of signature (5,1).
#   isotropic      a nonzero v with v F v' = 0.
#   witness        a primitive x with Q_A(x) = N² for some N ≥ 1; N is the
#                  number reported for the form.
#   presentation   a Gram matrix U F U' of the same lattice, U ∈ GL₆(ℤ).
#   p              a prime, given on the command line, such that
#                  det A = 16p² for every form of the file.
#
# ─── The reduction to isotropy ───────────────────────────────────────────────
#
#   Write B = 2A, so that Q_A(x) = ½·x'Bx, and set F = B ⊥ ⟨-2⟩. For
#   v = (x, z) ∈ ℤ⁵ × ℤ,
#
#       v F v' = x'Bx - 2z² = 2(Q_A(x) - z²),
#
#   so v is isotropic for F exactly when Q_A(x) = z². Finding a square
#   represented by A is thus the same problem as finding an isotropic vector
#   of F.
#
#   Two facts make the reduction clean, and both are checked below as
#   assertions rather than taken as assumptions:
#
#     (i)  z ≠ 0 automatically. A is positive definite, so x'Bx = 0 forces
#          x = 0, and then v = 0. Hence every isotropic v has z ≠ 0 and yields
#          a genuine square N² = z² ≥ 1.
#
#     (ii) x is primitive automatically, provided v is. Put g = gcd(x). Then
#          g² divides Q_A(x) = z², so g divides z, hence g divides gcd(v) = 1.
#          Primitivity of the senary vector is free: an isotropic vector may
#          always be divided by its content.
#
#   Since F is indefinite of rank 6 ≥ 5, Meyer's theorem says F is isotropic
#   over ℚ, hence over ℤ after clearing denominators. So every positive
#   definite quinary form primitively represents at least one nonzero square.
#   What the script measures is *which* square one can produce cheaply.
#
# ─── What N is, and what it is not ───────────────────────────────────────────
#
#   The N reported is the smallest one *found*, not the smallest that exists.
#   Simon's method returns whatever isotropic vector its reduction lands on;
#   that vector is exact and fully verified, but it is not the output of any
#   minimisation.
#
#   N can be lowered, at a price, by presenting the same lattice to the
#   algorithm repeatedly in different bases. For U ∈ GL₆(ℤ),
#
#       y (U F U') y' = (yU) F (yU)',
#
#   so an isotropic y for U F U' pulls back to the isotropic v = yU for F. The
#   lattice is unchanged; only the coordinates the reduction works in differ,
#   and with them the vector it lands on. Each random U is another draw at the
#   same problem, and `--trials` sets how many are taken. The default is none:
#   one presentation per form, one square reported. In practice the smallest
#   N after k draws falls roughly like 1/(k+1), so k draws cost k times the
#   work for one decimal digit per tenfold increase.
#
#   N = 1 is the smallest value possible, so when restarts are on they stop
#   as soon as it is reached.
#
# ─── The three routes to an isotropic vector ─────────────────────────────────
#
#   Three routes are tried in order, and the output records which one produced
#   each witness.
#
#     1. `lll_gram_indef_isotropic` (Hecke) — the indefinite LLL reduction of
#        Simon's 2005 paper, a direct port of his qfsolve.gp. It is cheap and
#        needs no factorisation of det F, but it is not a complete solver: it
#        returns an isotropic vector if the reduction happens to find one, and
#        otherwise the empty vector — a miss, not a proof of anisotropy.
#        Theorem 1.8 of the paper, which makes the reduction a solver in
#        rank ≤ 6, requires det = ±1, and det F = -2¹⁰p² is far from that. On
#        forms with det A = 16p² this route essentially never succeeds, and
#        the failure cannot be repaired by minimising F first as §2–3 of the
#        paper prescribe: the rational quadratic space of F has Hasse
#        invariant -1 at both 2 and p, so no rescaling of it is isometric to a
#        unimodular senary space.
#
#     2. `family_isotropic_rows` — the workhorse. This is the construction of
#        Simon's later note, as Hecke implements it in `_isotropic_subspace`,
#        carried out with what is already known about the forms of this file
#        supplied rather than rediscovered from the Gram matrix: that F is
#        isotropic, that the only primes dividing det F are 2 and p, and that
#        one auxiliary lattice serves every form of the same genus. See
#        `Family` and `family_isotropic_rows`.
#
#     3. `is_isotropic_with_vector` (Hecke) — the general solver, kept as a
#        safety net. Route 2 asserts every structural expectation it has and
#        raises rather than guess when one fails; the presentation is then
#        handed to route 3, which assumes nothing and, by Meyer, always
#        succeeds. The closing summary names any form that needed it.
#
# ─── Verification ────────────────────────────────────────────────────────────
#
#   Nothing is taken on trust from any solver. Every witness reported —
#   including every intermediate one — passes through `verify_witness`, which
#   recomputes in exact integer arithmetic
#
#       gcd(v) = 1,   v F v' = 0,   z ≠ 0,   gcd(x) = 1,   Q_A(x) = z².
#
#   A witness failing any of these aborts the run. The claims in the output
#   file are therefore exact, independently of the reduction that produced
#   them.
#
# ─── Input ───────────────────────────────────────────────────────────────────
#
#   The forms are read from a text file `RHI2_<prefix>.txt` or
#   `RHI2_<prefix>_<count>.txt` in the working directory, where `<prefix>` is
#   determined by p (see `rhi2_filename_prefix`). The file lists forms in
#   blocks
#
#       Type i
#       q(A,θ) = c₁₁*x1^2 + c₁₂*x1*x2 + … + c₅₅*x5^2
#
#   with the full cross coefficients c_ij = 2A[i,j] for i ≠ j. Any lines after
#   a line beginning `p = <p> (IKO` are ignored. The forms are expected to
#   have det A = 16p²; a form that does not is reported and processed anyway.
#
# ─── Output ──────────────────────────────────────────────────────────────────
#
#   Results go to `simon_p<prefix>_t<trials>_s<seed>.txt`, one line per form,
#
#       Type i: N
#
#   where N is the result of the run on F itself, before any change of basis,
#   or `none` if that run yielded nothing. When `--trials` is set the line is
#   followed by two more,
#
#         random N = N₁ N₂ … Nₖ
#         best N   = N*   (presentation j of k+1, d distinct)
#
#   giving the N of every random presentation in the order drawn (`-` for a
#   presentation that yielded nothing), and the smallest N over all
#   presentations, the direct run counted as presentation 1. The file ends
#   with the smallest and largest N seen and the forms attaining them.
#
#   Nothing else is printed per form: z = ±N, Q_A(x) = N², and det A = 16p²
#   throughout. The witness x itself is not printed. Anomalies — a
#   determinant other than 16p², a form solved by route 1, a form that needed
#   route 3, a solver error — are reported on the terminal at the end of the
#   run with the form numbers involved, so that a clean run is visibly clean.
#
# ─── Running ─────────────────────────────────────────────────────────────────
#
#   julia simon.jl <p> [--dir <dir>] [--first <i>] [--last <j>]
#                      [--trials <n>] [--steps <n>] [--coeff-bound <C>]
#                      [--seed <s>] [--no-fallback] [--no-stop-at-one]
#
#   <p> may be a plain integer or an arithmetic expression over + - * ^ ( )
#   evaluated in exact integer arithmetic, e.g. "2^721*3^176-1".
#
#   `--first` and `--last` restrict the run to a range of forms. Form i is
#   form i of the file whatever range it is computed in, and carries the same
#   seed, so ranges are independent runs that may be done in any order, on
#   any number of machines, and concatenated. Ranges are the unit of
#   parallelism: within one process the forms run one after another, because
#   each form reseeds Julia's global random number generator (see
#   `simon_scan`), and they are also how memory is bounded, since Hecke's
#   lattice machinery retains memory per presentation for the life of the
#   process. The full list of flags is at the end of this file.

using Oscar
using Random
using Printf

const ZZ = Oscar.ZZ
const QQ = Oscar.QQ

# ══════════════════════════════════════════════════════════════════════════════
# Two overrides evaluated into Hecke
# ══════════════════════════════════════════════════════════════════════════════
#
# Hecke 0.39.22, the version loaded by Oscar at the time of writing, narrows p
# to a 64-bit machine integer in two places on the way to a representative of
# the genus of the auxiliary lattice R of route 2. For p ≥ 2^63 both routes 2
# and 3 then raise InexactError on every presentation and every form would be
# reported as `none`. The two methods below are the library's own, with the
# value in question kept as a ZZRingElem; neither changes the mathematics.
#
# Each is applied only if a probe with a prime just above 2^63 still raises
# InexactError, so that once the installed Hecke has the fix this section does
# nothing.

# Hecke src/QuadForm/Quad/Spaces.jl, `_quadratic_form_with_invariants`, with
# the diagonal entries carried as ZZRingElem instead of Int (`D = ones(ZZRingElem, k)`
# in place of `D = ones(Int, k)`). Evaluated inside Hecke so that it replaces
# the method with the overflow.
let p = next_prime(Oscar.ZZ(2)^63), needed = false
  try
    Hecke._quadratic_form_with_invariants(4, Oscar.ZZ(1), [Oscar.ZZ(2), p], 4)
  catch e
    e isa InexactError || rethrow()
    needed = true
  end
  needed && Base.eval(Hecke, quote
  function _quadratic_form_with_invariants(dim::Int, det::ZZRingElem,
                                           finite::Vector{ZZRingElem}, negative::Int)
  #{Computes a quadratic form of dimension Dim and determinant Det that has Hasse invariants -1 at the primes in Finite.
   #The number of negative entries of the real signature is given in Negative}
    @hassert :Lattice 1 dim >= 1
    @hassert :Lattice 1 !iszero(det)
    @hassert :Lattice 1 negative in 0:dim

    sign(det) != (-1)^(negative % 2) && error("Real place information does not match the sign of the determinant")

    if dim == 1
      !isempty(finite) && error("Impossible Hasse invariants")
      return matrix(QQ, 1, 1, ZZRingElem[det])
    end

    finite = unique(finite)
    @hassert :Lattice 1 all(is_prime, finite)

    if dim == 2
      ok = all(p -> !is_local_square(-det, p), finite)

      if !ok
        #q = ZZRingElem[p for p in finite if is_local_square(-det, p)][1]
        if is_local_square(-det, q)
          error("A binary form with determinant $det must have Hasse invariant +1 at the prime $q")
        end
      end
    end

    # product formula check

    !iseven((negative % 4 >= 2 ? 1 : 0) + length(finite)) && error("The number of places (finite or infinite) with Hasse invariant -1 must be even")

    # reduce the number of bad primes
    det = squarefree_part(det)

    local det0::ZZRingElem
    local finite0::Vector{ZZRingElem}

    dim0 = dim
    det0 = det
    finite0 = copy(finite)
    negative0 = negative

    #// Pad with ones
    k = max(0, dim - max(3, negative))
    D = ones(ZZRingElem, k)
    dim = dim - k

    local PP::Vector{ZZRingElem}

    #// Pad with minus ones
    if dim >= 4
      @hassert :Lattice 1 dim == negative
      k = dim - 3
      d = (-1)^k
      f = (k % 4 >= 2) ? Set(ZZRingElem[2]) : Set(ZZRingElem[])
      PP = append!(ZZRingElem[p for (p, e) in factor(2 * det)], finite)
      unique!(PP)
      finite = ZZRingElem[ p for p in PP if hilbert_symbol(d, -det, p) * (p in f ? -1 : 1) * (p in finite ? -1 : 1) == -1]
      unique!(finite)
      D = append!(D, ZZRingElem[-1 for i in 1:k])
      det = isodd(k) ? -det : det
      dim = 3
      negative = 3
    end

    # ternary case
    if dim == 3
      #// The primes at which the form is anisotropic
      PP = append!(ZZRingElem[p for (p, e) in factor(2 * det)], finite)
      unique!(PP)
      filter!(p -> hilbert_symbol(-1, -det, p) != (p in finite ? -1 : 1), PP)
      #// Find some a such that for all p in PP: -a*Det is not a local square
      #// TODO: Find some smaller a?! The approach below is very lame.
      a = prod(p for p in PP if det % p != 0; init = one(ZZ))
      if negative == 3
        a = -a
        negative = 2
      end

      PP = append!(ZZRingElem[p for (p, e) in factor(2 * det * a)], finite)
      unique!(PP)
      finite = ZZRingElem[ p for p in PP if hilbert_symbol(a, -det, p) * (p in finite ? -1 : 1) == -1]
      det = squarefree_part(det * a)
      dim = 2
      push!(D, a)
    end

    #// The binary case
    a = _find_quaternion_algebra(QQFieldElem(-det)::QQFieldElem, finite::Vector{ZZRingElem}, negative == 2 ? PosInf[inf] : PosInf[])
    Drat = map(QQ, D)
    push!(Drat, a)
    push!(Drat, squarefree_part(ZZ(det * a)))
    M = diagonal_matrix(Drat)

    _, _, d, f, n = _quadratic_form_invariants(M)

    @hassert :Lattice 1 dim0 == length(Drat)
    @hassert :Lattice 1 d == det0
    @hassert :Lattice 1 issetequal(collect(keys(f)), finite0)
    @hassert :Lattice 1 n[1][2] == negative0
    return M
  end
  end)
end

# Hecke src/QuadForm/Quad/NormalForm.jl, `_sqrt`. The modulus p was narrowed
# to Int before being handed to Nemo.sqrtmod_pk, which accepts any integer
# type; keep it a ZZRingElem. The two methods are added beside the old ones
# and, being more specific in the first argument, take precedence.
let p = next_prime(Oscar.ZZ(2)^63), needed = false
  try
    Hecke._sqrt(residue_ring(Oscar.ZZ, p)[1](4), p)
  catch e
    e isa InexactError || rethrow()
    needed = true
  end
  needed && Base.eval(Hecke, quote
  function _sqrt(d::ZZModRingElem, p)
    R = parent(d)
    v, pp = is_perfect_power_with_data(R.n)
    return _sqrt(d, pp, Int(v))
  end
  function _sqrt(d::ZZModRingElem, p::ZZRingElem, prec::Int)
    R = parent(d)
    rt = Nemo.sqrtmod_pk(lift(d), p, prec)
    return ZZModRingElem(rt, R)
  end
  end)
end

# ══════════════════════════════════════════════════════════════════════════════
# Options
# ══════════════════════════════════════════════════════════════════════════════

"""
    Options(; ...)

Parameters for one `simon_scan` over a single quinary form.

| Field          | Default      | Meaning                                                     |
|:---------------|:-------------|:------------------------------------------------------------|
| `trials`       | `0`          | Number of random presentations tried after the direct run.  |
| `steps`        | `12`         | Elementary row operations composed into each random U.      |
| `coeff_bound`  | `2`          | Bound C on the multiplier in a transvection row operation.  |
| `seed`         | `20260904`   | Seed of the run's random number generator.                  |
| `fallback`     | `true`       | Use routes 2 and 3 when the indefinite LLL misses.          |
| `stop_at_one`  | `true`       | Stop a form's restarts once N = 1 is reached.               |

`steps` and `coeff_bound` control how far a random presentation travels from
the identity. A U built from few, small operations leaves the Gram matrix
close to F, so the reduction tends to retrace the direct run and return the
same vector; a U built from many, large operations inflates the entries of
U F U', which costs reduction time and tends to produce a *larger* isotropic
vector, since the reduction has further to travel back. The defaults are
Simon's own order of magnitude.

`fallback` should be left on: route 1 alone essentially never succeeds on
forms with det A = 16p² (see the file header), so turning it off discards
every witness. Turn it off only to *measure* the hit rate of the indefinite
LLL on a family whose answer is not already known; a presentation whose
indefinite LLL misses is then recorded as `:miss`, which is not a claim that
the presentation is anisotropic.

`trials` defaults to 0: one presentation per form, one square reported. Each
further presentation is another draw at the same problem (see the file
header).

`stop_at_one`: N ≥ 1 always, so once a primitive x with Q_A(x) = 1 is in hand
no further presentation can improve on it. Set it to `false` only when the
*spread* of N is itself the object of study and a complete set of `trials`
draws is wanted for every form.
"""
Base.@kwdef struct Options
    trials::Int       = 0
    steps::Int        = 12
    coeff_bound::Int  = 2
    seed::Int         = 20260904
    fallback::Bool    = true
    stop_at_one::Bool = true
end

# ══════════════════════════════════════════════════════════════════════════════
# Core arithmetic
#
# Everything below is exact integer arithmetic on FLINT `ZZMatrix` /
# `ZZRingElem` values. For large p the entries of A already exceed 64 bits,
# and the entries of U F U' after a dozen row operations exceed them by much
# more, so any narrowing to a machine integer would silently overflow.
# ══════════════════════════════════════════════════════════════════════════════

"""
    senary_form(A) -> ZZMatrix

Build F = 2A ⊥ ⟨-2⟩ from the quinary Gram matrix A, as a 6×6 `ZZMatrix`.

The scaling by 2 makes F even, which is what Hecke's lattice machinery works
with: A ⊥ ⟨-1⟩ would already be integral and would serve for the reduction
alone, but `is_isotropic_with_vector` would rescale by 2 itself. Doing it once
here keeps all routes on the same F, at the cost of a factor 2⁶ in the
determinant.
"""
function senary_form(A::ZZMatrix)
    nrows(A) == 5 && ncols(A) == 5 || error("expected a 5×5 quinary Gram matrix, got $(nrows(A))×$(ncols(A))")
    return block_diagonal_matrix([2 * A, matrix(ZZ, 1, 1, [ZZ(-2)])])
end

"""
    qvalue(A, x) -> ZZRingElem

The quinary form Q_A(x) = x'Ax, for `x` a 1×5 row `ZZMatrix`.
"""
qvalue(A::ZZMatrix, x::ZZMatrix) = (x * A * transpose(x))[1, 1]

"""
    fvalue(F, v) -> ZZRingElem

The senary form v F v', for `v` a 1×6 row `ZZMatrix`.
"""
fvalue(F::ZZMatrix, v::ZZMatrix) = (v * F * transpose(v))[1, 1]

"""
    primitive_row(v) -> ZZMatrix

Clear denominators and divide out the content, turning a nonzero rational row
vector into the unique (up to sign) primitive integer vector on the same line.

Accepts a rational row matrix, a `Vector{QQFieldElem}` or an integer row
matrix, since the solvers return different shapes: the indefinite LLL returns
an integer row matrix, while `is_isotropic_with_vector` returns a rational
vector in the coordinates of the ambient quadratic space.
"""
function primitive_row(v::ZZMatrix)
    g = content(v)
    iszero(g) && error("cannot normalise the zero vector to a primitive one")
    return divexact(v, g)
end

# `denominator(::QQMatrix)` is the common denominator of all entries, so
# `denominator(v) * v` is integral and the coercion to ZZ is exact.
primitive_row(v::QQMatrix) = primitive_row(change_base_ring(ZZ, denominator(v) * v))

primitive_row(v::Vector{QQFieldElem}) = primitive_row(matrix(QQ, 1, length(v), v))

# ══════════════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════════════

"""
    verify_witness(A, F, v) -> (x, z)

Check, in exact integer arithmetic, every claim this script makes about the
isotropic row vector `v = (x, z)`, and return the quinary part `x` (a 1×5
`ZZMatrix`) together with `z`. Throws on any failure.

The five checks are

    gcd(v) = 1,   v F v' = 0,   z ≠ 0,   gcd(x) = 1,   Q_A(x) = z²,

of which the last two are, by the argument in the file header, consequences of
the first three for this particular F. They are checked anyway rather than
inferred: they are the properties actually claimed in the output, they cost a
gcd and a matrix product, and a solver returning a vector for the wrong Gram
matrix — the one failure mode a basis-change bug produces — is caught here and
nowhere else.
"""
function verify_witness(A::ZZMatrix, F::ZZMatrix, v::ZZMatrix)
    (nrows(v) == 1 && ncols(v) == 6) ||
        error("witness must be a 1×6 row vector, got $(nrows(v))×$(ncols(v))")
    isone(content(v))    || error("witness is not primitive: gcd = $(content(v))")
    iszero(fvalue(F, v)) || error("witness is not isotropic: v F v' = $(fvalue(F, v))")

    x = v[1:1, 1:5]
    z = v[1, 6]

    iszero(z)            && error("witness has z = 0, so it carries no square")
    isone(content(x)) || error("quinary part is not primitive: gcd = $(content(x))")
    q = qvalue(A, x)
    q == z^2 || error("Q_A(x) = $q does not match z² = $(z^2)")

    return x, z
end

# ══════════════════════════════════════════════════════════════════════════════
# Random presentations of the same lattice
# ══════════════════════════════════════════════════════════════════════════════

"""
    random_unimodular(rng, n, steps, C) -> ZZMatrix

A random element of GLₙ(ℤ), built as a product of `steps` elementary row
operations drawn uniformly from three kinds: adding a multiple a of row j to
row i with 1 ≤ |a| ≤ C, negating a row, and swapping two rows.

Each of the three is unimodular, so the product is too, and this is asserted
before returning. Composing elementary operations rather than sampling entries
directly is what keeps the result unimodular by construction.

The multiplier `a` is resampled to 1 when it comes out 0, since a = 0 makes the
operation the identity and silently costs a step.
"""
function random_unimodular(rng::AbstractRNG, n::Int, steps::Int, C::Int)
    U = identity_matrix(ZZ, n)
    for _ in 1:steps
        # A row operation needs two *distinct* rows. Drawing j from 1:n-1 and
        # then shifting it past i gives a uniform draw from the n-1 rows other
        # than i.
        i = rand(rng, 1:n)
        j = rand(rng, 1:n-1)
        j >= i && (j += 1)

        typ = rand(rng, 1:3)
        if typ == 1
            a = rand(rng, -C:C)
            a == 0 && (a = 1)
            for k in 1:n
                U[i, k] += a * U[j, k]
            end
        elseif typ == 2
            for k in 1:n
                U[i, k] = -U[i, k]
            end
        else
            for k in 1:n
                U[i, k], U[j, k] = U[j, k], U[i, k]
            end
        end
    end
    is_unimodular(U) || error("row operations produced a non-unimodular matrix (det = $(det(U)))")
    return U
end

# ══════════════════════════════════════════════════════════════════════════════
# What p settles before any form is read
# ══════════════════════════════════════════════════════════════════════════════

"""
    Family(p)

The part of the isotropy computation that `p` alone decides, built once per run
and handed to every presentation of every form.

`is_isotropic_with_vector` is a solver for an arbitrary rational quadratic
space. Given only a Gram matrix it must rediscover four things that are already
known here, and `Family` supplies them instead.

  1. *That F is isotropic at all.* Hecke decides this from the Hasse invariants
     of F, which needs 2·det F factored. Here F has rank 6 and signature (5,1),
     so Meyer's theorem settles it with no computation, and the test is simply
     not performed.

  2. *Which primes divide det F.* Only 2 and p do, since det F = -2¹⁰p². Hecke
     calls `prime_divisors` on det F, and again on the determinant of the
     rank-10 lattice built in `family_isotropic_rows`. Both are replaced by
     `bad_primes`. For p of hundreds of digits this is the difference between
     a factorisation that is free — p is an argument to this script — and one
     that has to rediscover p.

  3. *The lattice glued to M.* The construction completes the maximal even
     overlattice M of F to something unimodular by gluing on a negative
     definite quaternary lattice R carrying the opposite discriminant form. R
     depends only on the genus of M, hence only on the genus of A, which is
     the same for every form with det A = 16p² of the file: one R serves every
     form and every presentation. `glue` caches it, keyed by that genus rather
     than assumed from it, so a form of a different genus gets its own R
     instead of the wrong one.

  4. *How much of the isotropic subspace is needed.* See
     `family_isotropic_rows`.

`glue` is mutated as the run proceeds, so a `Family` belongs to one run and
must not be shared between concurrent ones.
"""
struct Family
    p::ZZRingElem
    bad_primes::Vector{ZZRingElem}
    glue::Dict{ZZGenus,ZZLat}
end

Family(p::Integer) = Family(ZZ(p), ZZRingElem[ZZ(2), ZZ(p)], Dict{ZZGenus,ZZLat}())

"""
    family_isotropic_rows(G, fam) -> Vector{QQMatrix}

The construction of Simon's "Quadratic equations in dimensions 4, 5 and more",
as Hecke implements it in `_isotropic_subspace`, specialised to this family.
Returns the rows of an isotropic subspace of the senary form with Gram matrix
`G`, in the coordinates of `G`.

The construction, unchanged: pass to a maximal even overlattice M of the
lattice of `G`; glue on a negative definite quaternary R whose discriminant
form is the opposite of M's, so that the sum has an even unimodular
overlattice MM of signature (5,5); find a totally isotropic subspace of MM,
which is the regime of the 2005 paper since MM is unimodular; and intersect it
back with M.

Two things are done differently.

*The isotropic subspace is grown one dimension at a time.* Hecke builds a
*maximal* one, of dimension 5, because that is what a dimension count
guarantees to meet M: 5 + rank M = 11 > 10 = rank MM, and in general nothing
smaller will do. Building it costs five rounds of reduce, complete to a
hyperbolic plane, and pass to the orthogonal complement. Here the subspace is
extended one vector at a time and tested against M after each, so the rounds
that a dimension count only guarantees are not spent when they are not needed.
In practice the first isotropic vector already lies in M. The loop still runs
to five, and still terminates for the same reason Hecke's recursion does.

*Every row is returned, not just the first.* The intersection with M is
totally isotropic, so each of its basis vectors is a separate witness, and
`simon_presented` keeps the best.

Every structural expectation is asserted rather than assumed: signature,
parity, unimodularity of the glued lattice, and isotropy of what M meets. A
form outside this family therefore raises here instead of being solved by
accident, and `isotropic_rows` falls back to Hecke's general routine for it.
"""
function family_isotropic_rows(G::ZZMatrix, fam::Family)
    q   = quadratic_space(QQ, change_base_ring(QQ, G))
    sig = signature_tuple(q)
    sig == (5, 0, 1) ||
        error("expected signature (5,0,1) for F = 2A ⊥ ⟨-2⟩, got $sig")

    L = Hecke.lattice(q)
    iseven(L) || error("F = 2A ⊥ ⟨-2⟩ should be an even lattice")

    # A maximal even overlattice of L, at the two primes that can divide det F.
    # The argument-free `maximal_even_lattice` would factor det L to find them.
    M = L
    for r in fam.bad_primes
        M = maximal_even_lattice(M, r)
    end
    M = lll(M)

    # A reduced basis of M may already begin with isotropic vectors.
    GM = gram_matrix(M)
    if iszero(GM[1, 1])
        i = 1
        while iszero(GM[1:i+1, 1:i+1])
            i += 1
        end
        BM = basis_matrix(M)
        return QQMatrix[BM[j:j, :] for j in 1:i]
    end

    # Glue M to a negative definite quaternary lattice carrying the opposite
    # discriminant form. The signature is forced: MM is even unimodular, so its
    # signature difference must be divisible by 8, and (5 - 1) - a ≡ 0 (mod 8)
    # with a ≥ 0 minimal gives a = 4.
    D   = rescale(discriminant_group(M), -1)
    gen = genus(D, (0, 4))
    R   = get!(() -> representative(gen), fam.glue, gen)

    LL, inj = direct_sum(M, R)
    MM = LL
    for r in fam.bad_primes
        MM = maximal_even_lattice(MM, r)
    end
    isone(abs(det(MM))) ||
        error("the glued rank-10 lattice is not unimodular: det = $(det(MM))")

    VV    = ambient_space(MM)
    cur   = MM
    Hrows = zero_matrix(QQ, 0, degree(MM))

    for _ in 1:5
        cur = lll(cur)
        Gc  = gram_matrix(cur)
        iso = Hecke._isotropic_subspace_unimodular_gram_no_lll(Gc)
        Hrows = vcat(Hrows, iso * basis_matrix(cur))

        met = preimage(inj[1], Hecke.lattice(VV, Hrows))
        if rank(met) > 0
            iszero(gram_matrix(met)) ||
                error("the subspace M meets is not isotropic")
            B = basis_matrix(met)
            return QQMatrix[B[i:i, :] for i in 1:nrows(B)]
        end

        # Complete the new isotropic vectors to hyperbolic planes and pass to
        # the orthogonal complement, again even unimodular and of rank two
        # less. This is Hecke's recursion, written as a loop so that the
        # intersection with M can be tested between rounds.
        C   = solve(change_base_ring(ZZ, Gc * transpose(iso)),
                    identity_matrix(ZZ, nrows(iso)); side = :left)
        cur = orthogonal_submodule(cur,
                  Hecke.lattice(ambient_space(cur),
                                vcat(iso, C) * basis_matrix(cur)))
    end

    error("the isotropic subspace of the glued lattice never met M")
end

# ══════════════════════════════════════════════════════════════════════════════
# Simon's algorithm on one presentation
# ══════════════════════════════════════════════════════════════════════════════

"""
    isotropic_rows(G, fam; fallback=true) -> (rows, route)

Find isotropic vectors of the senary form with Gram matrix `G`, returning them
as a vector of 1×6 row `ZZMatrix`es together with a symbol saying which route
produced them:

  `:lll`       route 1, `lll_gram_indef_isotropic`, succeeded.
  `:family`    the indefinite LLL missed and route 2, `family_isotropic_rows`,
               supplied the witness.
  `:complete`  route 2 raised — a form outside the family — and route 3,
               Hecke's general `is_isotropic_with_vector`, was used instead.
               Guaranteed to succeed (F is indefinite of rank 6, so Meyer's
               theorem applies), at the cost of computing the genus of G from
               scratch.
  `:miss`      the indefinite LLL missed and `fallback` was off. Not a claim
               that G is anisotropic.
  `:error`     the indefinite LLL raised and `fallback` was off, or both of
               the other routes also failed. Recorded rather than propagated,
               so one bad presentation does not abandon a form.

The indefinite LLL is called with `base = true`. That flag is documented as
controlling the *first* return value, which is discarded here, but it also
settles the shape of the third: on its "singular principal minor" branch the
underlying `_quadratic_form_solve_triv` returns the solution as a column when
`base = false`, and as a row — matching every other branch — when
`base = true`. A singular leading minor is entirely possible for a random
presentation of F, so `base = true` is what keeps the returned shape uniform.
"""
function isotropic_rows(G::ZZMatrix, fam::Family; fallback::Bool = true)
    sol = try
        lll_gram_indef_isotropic(G; base = true)[3]
    catch e
        e isa InterruptException && rethrow()
        nothing
    end

    if sol isa MatElem
        return [primitive_row(sol)], :lll
    end

    errored = sol === nothing
    fallback || return (ZZMatrix[], errored ? :error : :miss)

    rows = try
        family_isotropic_rows(G, fam)
    catch e
        e isa InterruptException && rethrow()
        # `maxlog` because a systematic failure would otherwise print once per
        # presentation, tens of thousands of times, while the run reverts to
        # route 3. The closing summary is the reliable record of how often
        # this happened.
        @warn "the specialised construction failed on one presentation; " *
              "falling back to is_isotropic_with_vector" exception = e maxlog = 5
        nothing
    end
    rows === nothing || return ([primitive_row(r) for r in rows], :family)

    q = quadratic_space(QQ, change_base_ring(QQ, G))
    ok, w = try
        is_isotropic_with_vector(q)
    catch e
        e isa InterruptException && rethrow()
        false, QQFieldElem[]
    end
    ok || return (ZZMatrix[], :error)

    return [primitive_row(w)], :complete
end

"""
    simon_presented(A, F, U, fam; fallback=true) -> NamedTuple

Run Simon's algorithm on the presentation U F U' of the lattice and pull the
result back to F.

If y is isotropic for U F U', then v = yU satisfies

    v F v' = y (U F U') y' = 0,

so v is isotropic for F itself — in the *original* coordinates, which is what
makes the witness comparable across presentations and checkable against the
original A. The pullback is by the identity above, not by inverting U, so no
rational arithmetic enters.

Returns `(N, v, route, secs)`, where `N = |z|` and `v` is the verified witness
attaining it, or `N = nothing` and `v = nothing` when the presentation yielded
nothing (`route` then being `:miss` or `:error`). `secs` is wall-clock time in
the solver, excluding verification.
"""
function simon_presented(A::ZZMatrix, F::ZZMatrix, U::ZZMatrix, fam::Family;
                         fallback::Bool = true)
    FU = U * F * transpose(U)

    t0 = time()
    rows, route = isotropic_rows(FU, fam; fallback)
    secs = time() - t0

    bestN = nothing
    bestV = nothing
    for y in rows
        # y is isotropic for U F U'; v = yU is isotropic for F.
        v = primitive_row(y * U)
        _, z = verify_witness(A, F, v)
        N = abs(z)
        if bestN === nothing || N < bestN
            bestN = N
            bestV = v
        end
    end

    return (N = bestN, v = bestV, route = route, secs = secs)
end

# ══════════════════════════════════════════════════════════════════════════════
# One form
# ══════════════════════════════════════════════════════════════════════════════

"""
    simon_scan(A, fam; options=Options(), seed=options.seed) -> NamedTuple

Run the direct presentation and then `options.trials` random ones on the single
quinary form `A`, and return

    (bestN, bestX, bestZ, bestV, directN, route_of_best,
     Ns, routes, improvements, total_secs, trials_run)

where `Ns` is the vector of every N obtained, in the order obtained (the direct
run first), and `routes` the matching route per presentation, including the
presentations that yielded nothing.

`bestX` is the primitive quinary witness and `bestZ` its z, so that
Q_A(bestX) = bestZ² = bestN². The final witness is re-verified before being
returned.

`seed` pins the whole scan, and does so by reseeding Julia's **global** random
number generator — a deliberate side effect, and the only way to make the run
reproducible. There are two sources of randomness: the random presentations,
which could be driven from a private generator, and `is_isotropic_with_vector`,
which draws from the global one internally when it searches for a genus
representative. Seeding globally and drawing the presentations from the same
global stream pins both at once.
"""
function simon_scan(A::ZZMatrix, fam::Family;
                    options::Options = Options(),
                    seed::Integer    = options.seed)
    Random.seed!(seed)
    rng = Random.default_rng()

    F = senary_form(A)

    Ns     = Union{Nothing,ZZRingElem}[]
    routes = Symbol[]

    direct = simon_presented(A, F, identity_matrix(ZZ, 6), fam; fallback = options.fallback)
    push!(Ns, direct.N)
    push!(routes, direct.route)

    bestN, bestV, bestRoute = direct.N, direct.v, direct.route
    total_secs   = direct.secs
    improvements = 0
    trials_run   = 0

    if !(options.stop_at_one && bestN == 1)
        for _ in 1:options.trials
            trials_run += 1
            U = random_unimodular(rng, 6, options.steps, options.coeff_bound)
            r = simon_presented(A, F, U, fam; fallback = options.fallback)

            push!(Ns, r.N)
            push!(routes, r.route)
            total_secs += r.secs

            if r.N !== nothing && (bestN === nothing || r.N < bestN)
                bestN, bestV, bestRoute = r.N, r.v, r.route
                improvements += 1
            end

            options.stop_at_one && bestN == 1 && break
        end
    end

    bestX, bestZ = bestV === nothing ? (nothing, nothing) : verify_witness(A, F, bestV)

    return (bestN = bestN, bestX = bestX, bestZ = bestZ, bestV = bestV,
            directN = direct.N, route_of_best = bestRoute,
            Ns = Ns, routes = routes, improvements = improvements,
            total_secs = total_secs, trials_run = trials_run)
end

simon_scan(A::AbstractMatrix{<:Integer}, fam::Family; kwargs...) =
    simon_scan(matrix(ZZ, A), fam; kwargs...)

# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

"""
    spread_summary(res) -> NamedTuple

Condense one form's presentations into the numbers that say how much the choice
of basis mattered:

    (found, attempted, minN, medN, maxN, distinct, n_lll, n_family,
     n_complete, n_miss, n_error)

`medN` is the lower median of the N that were found (the ⌈m/2⌉-th of m sorted
values), so that it is always one of the N actually obtained and an exact
integer.

`distinct` is the count of distinct N over the successful presentations:
`distinct = 1` says every basis led the reduction back to the same square, so
restarting bought nothing on this form, while a large `distinct` says the
answer depends on the presentation and the best N found is likely still far
from the best that exists.
"""
function spread_summary(res)
    found = ZZRingElem[N for N in res.Ns if N !== nothing]
    sort!(found)
    m = length(found)
    return (found      = m,
            attempted  = length(res.Ns),
            minN       = m == 0 ? nothing : found[1],
            medN       = m == 0 ? nothing : found[cld(m, 2)],
            maxN       = m == 0 ? nothing : found[m],
            distinct   = length(unique(found)),
            n_lll      = count(==(:lll),      res.routes),
            n_family   = count(==(:family),   res.routes),
            n_complete = count(==(:complete), res.routes),
            n_miss     = count(==(:miss),     res.routes),
            n_error    = count(==(:error),    res.routes))
end

"""
    fmt_elapsed(secs) -> String

Wall-clock time as "1.23s", "4m 05s" or "2h 13m 07s", whichever fits.
"""
function fmt_elapsed(secs::Float64)
    secs < 60 && return @sprintf("%.2fs", secs)
    s = round(Int, secs)
    h, rem = divrem(s, 3600)
    m, sec = divrem(rem, 60)
    h > 0 && return @sprintf("%dh %02dm %02ds", h, m, sec)
    return @sprintf("%dm %02ds", m, sec)
end

"""
    form_list(v; limit=12) -> String

The form numbers in `v` as "3, 17, 204", truncated to `limit` of them with a
count of the rest. Used by the summary to name the forms behind an anomaly
without letting a systematic one fill the terminal.
"""
function form_list(v::Vector{Int}; limit::Int = 12)
    length(v) <= limit && return join(v, ", ")
    return join(v[1:limit], ", ") * ", … (" * string(length(v)) * " forms in all)"
end

# ══════════════════════════════════════════════════════════════════════════════
# Input
#
# The forms are read from `RHI2_<prefix>_<count>.txt` or `RHI2_<prefix>.txt`,
# in blocks headed `Type i` and carrying a line `q(A,θ) = …` (or `q(ExE,θ) = …`)
# with the form written out as a polynomial. See the file header.
# ══════════════════════════════════════════════════════════════════════════════

"""
    rhi2_filename_prefix(pZ) -> String

The `<prefix>` in `RHI2_<prefix>_*.txt`: the decimal value of `p` for primes of
25 bits or fewer, and `<bits>bit_<hash>` beyond that, where `<hash>` is the
low 32 bits of Julia's `hash(string(p))` in hexadecimal. The file name of a
large prime is thus short while still determined by p.
"""
function rhi2_filename_prefix(pZ::BigInt)::String
    s    = string(pZ)
    bits = ndigits(pZ, base = 2)
    if bits <= 25
        return s
    end
    digest = string(hash(s) & 0xffffffff; base = 16, pad = 8)
    return "$(bits)bit_$(digest)"
end
rhi2_filename_prefix(p::Integer) = rhi2_filename_prefix(BigInt(p))

"""
    parse_gram_matrix(q_str) -> Matrix{BigInt}

Parse a quinary form string "c₁₁*x1^2 + c₁₂*x1*x2 + … + c₅₅*x5^2" into the
symmetric 5×5 Gram matrix A with Q_A(x) = x'Ax. A leading `q(A,θ) =` or
`q(ExE,θ) =` label is stripped if present.

  Diagonal terms:     coefficient of `xi^2`        → `A[i,i]`.
  Off-diagonal terms: coefficient of `xi*xj`, i≠j  → `A[i,j] = A[j,i] = coeff/2`
                      (the file writes the full cross coefficient 2·A[i,j], so
                      an odd one means the file is not of this shape and is
                      rejected rather than rounded).

Coefficients are parsed as `BigInt`: for large p the Gram entries themselves
exceed 64 bits.
"""
function parse_gram_matrix(q_str::AbstractString)
    A = zeros(BigInt, 5, 5)
    s = replace(q_str, r"^q\((A|ExE),θ\)\s*=\s*" => "")

    for m in eachmatch(r"([+-]?\s*\d+)\s*\*\s*x(\d)\^2", s)
        i       = parse(Int, m.captures[2])
        A[i, i] = parse(BigInt, replace(m.captures[1], " " => ""))
    end
    for m in eachmatch(r"([+-]?\s*\d+)\s*\*\s*x(\d)\s*\*\s*x(\d)", s)
        coeff = parse(BigInt, replace(m.captures[1], " " => ""))
        i     = parse(Int, m.captures[2])
        j     = parse(Int, m.captures[3])
        i == j && continue
        iseven(coeff) || error("odd off-diagonal coefficient $coeff for x$i*x$j in \"$q_str\" — expected 2*A[i,j]")
        half    = coeff ÷ 2
        A[i, j] = half
        A[j, i] = half
    end
    return A
end

"""
    read_rhi2_file(p; dir=".") -> (Vector{Matrix{BigInt}}, path)

Locate `RHI2_<prefix>_<count>.txt` or `RHI2_<prefix>.txt` in `dir` and return
one 5×5 Gram matrix per "Type i" block, together with the path read. It is an
error for `dir` to hold neither.

Each pending `q(A,θ) = …` block is committed when the *next* "Type i" header
is seen (or immediately, at a `ways to represent` line, in files that carry
one), with a final commit after the loop for the last block.

Parsing stops at the line `p = <p> (IKO …`, so trailing statistics are
ignored. That line spells out the literal decimal prime, so the match uses the
decimal value rather than the hashed prefix.
"""
function read_rhi2_file(p::Integer; dir::AbstractString = ".")
    pZ     = BigInt(p)
    prefix = rhi2_filename_prefix(pZ)
    candidates = filter(readdir(dir; join = true)) do f
        occursin(Regex("RHI2_$(prefix)(_\\d+)?\\.txt\$"), f)
    end
    isempty(candidates)    && error("No RHI2_$(prefix)*.txt found in \"$dir\"")
    length(candidates) > 1 && @warn "Multiple matches; using $(candidates[1])"

    all_lines = readlines(candidates[1])
    stop_idx  = findfirst(l -> occursin(Regex("^p\\s*=\\s*$(pZ)\\s+\\(IKO"), l), all_lines)
    lines     = stop_idx === nothing ? all_lines : all_lines[1:stop_idx-1]

    matrices  = Matrix{BigInt}[]
    current_q = nothing
    for line in lines
        if occursin(r"^Type\s+\d+", line)
            current_q !== nothing && push!(matrices, parse_gram_matrix(current_q))
            current_q = nothing
        elseif occursin(r"^q\((A|ExE),θ\)\s*=", line)
            current_q = line
        elseif occursin(r"^ways to represent", line) && current_q !== nothing
            push!(matrices, parse_gram_matrix(current_q))
            current_q = nothing
        end
    end
    current_q !== nothing && push!(matrices, parse_gram_matrix(current_q))

    isempty(matrices) &&
        error("No form blocks parsed from $(candidates[1]) — expected lines beginning " *
              "\"q(A,θ) =\" or \"q(ExE,θ) =\"")
    return matrices, candidates[1]
end

# ══════════════════════════════════════════════════════════════════════════════
# Whole file
# ══════════════════════════════════════════════════════════════════════════════

"""
    process_rhi2(p; dir=".", options=Options(), first_form=1, last_form=nothing) -> String

Run `simon_scan` on the forms `first_form:last_form` of the RHI2 file for `p`,
write the results to `simon_p<prefix>_t<trials>_s<seed>.txt` in `dir`, and
return that path. The seed is in the name because it is what makes a run
reproducible: two runs of the same file that differ only in seed are different
experiments and must not overwrite one another. A run over part of the file
gets `_f<first>-<last>` as well.

`first_form` and `last_form` exist so that a large file can be covered by
several runs instead of one. Form numbering is always numbering *within the
file*, so form i is form i whatever range it was computed in — and, by the
seeding rule below, carries the same seed and produces the same witness. The
ranges of a file may therefore be run in any order, on any number of machines,
and concatenated; nothing is shared between them but the input. Separate
processes are also how memory is bounded: Hecke's lattice machinery retains
memory per presentation for the life of the process, so a single process
walking a long file with many trials grows without bound, while a process per
range returns its memory when it exits.

The determinant of each form is checked against 16p². A mismatch is reported
as a warning and the form is still processed: such a form falls outside the
specialised route, not outside the mathematics, and `isotropic_rows` hands it
to Hecke's general solver instead — so it still gets a witness, and the
closing summary names it.

One `Family` is built here, from p, and shared by every form of the run. Its
`glue` cache is the run's own record of the assumption that all these forms lie
in one genus: the summary reports how many distinct glued lattices were built,
and for a file of forms with det A = 16p² that number should be 1.

Form i is scanned under the seed `options.seed + i - 1`, rather than all forms
sharing one continuing random stream. This keeps the forms independent of one
another: form i reproduces on its own, and it reproduces *identically* whatever
`trials` is set to, since nothing form i sees depends on how many draws the
earlier forms consumed. That is what makes two runs at different `trials`
comparable form by form.
"""
function process_rhi2(p::Integer;
                      dir::AbstractString          = ".",
                      options::Options             = Options(),
                      first_form::Int              = 1,
                      last_form::Union{Nothing,Int} = nothing)
    pZ     = BigInt(p)
    pdisp  = rhi2_filename_prefix(pZ)

    @info "Reading RHI2 file for p = $pZ …"
    matrices, srcfile = read_rhi2_file(pZ; dir)
    nforms = length(matrices)
    @info "  Found $nforms form(s) in $(basename(srcfile))."

    lo = first_form
    hi = last_form === nothing ? nforms : last_form
    1 <= lo <= hi <= nforms ||
        error("form range $lo:$hi is not within 1:$nforms")
    partial = (lo, hi) != (1, nforms)

    fam     = Family(pZ)
    suffix  = partial ? "_f$(lo)-$(hi)" : ""
    outpath = joinpath(dir, "simon_p$(pdisp)_t$(options.trials)_s$(options.seed)$(suffix).txt")
    t_start = time()

    open(outpath, "w") do io
        println(io, "=" ^ 78)
        println(io, "Simon isotropic-vector search for primitively represented squares")
        println(io, "p = $pZ")
        println(io, "forms = $(hi - lo + 1) of $nforms (form $lo to form $hi)   " *
                    "source = $(basename(srcfile))")
        println(io, "trials = $(options.trials)   steps = $(options.steps)   " *
                    "coeff_bound = $(options.coeff_bound)   seed = $(options.seed)")
        println(io, "fallback = $(options.fallback)   stop_at_one = $(options.stop_at_one)")
        if options.trials > 0
            # The file carries more than a list of N, so it says what.
            println(io, "Type i: N        N found on F itself, before any change of basis")
            println(io, "  random N = …   one N per random presentation, in the order drawn (- = none)")
            println(io, "  best N   = …   the smallest of them all, the direct run counted as presentation 1")
        end
        println(io, "=" ^ 78)
        flush(io)

        best_over_all  = nothing
        best_form      = 0
        worst_over_all = nothing
        worst_form     = 0
        n_one         = 0
        n_none        = 0
        all_secs      = 0.0
        # Forms worth naming individually: a determinant outside the genus, or
        # a witness that did not come from the specialised route.
        odd_det       = Int[]
        used_lll      = Int[]
        used_complete = Int[]
        had_error     = Int[]

        for fi in lo:hi
            A = matrix(ZZ, matrices[fi])
            d = det(A)
            if d != 16 * pZ^2
                @warn "form $fi: det A = $d, expected 16p² = $(16 * pZ^2)"
            end

            res = simon_scan(A, fam; options, seed = options.seed + fi - 1)
            s   = spread_summary(res)
            all_secs += res.total_secs

            if res.bestN === nothing
                n_none += 1
            else
                res.bestN == 1 && (n_one += 1)
                # `s.minN` and `s.maxN` are the extremes over this form's
                # presentations. At trials = 0 there is one presentation, so
                # both equal the direct N; with restarts they are the smallest
                # and largest N seen in any presentation of the form. One rule
                # covers both cases.
                if best_over_all === nothing || s.minN < best_over_all
                    best_over_all = s.minN
                    best_form     = fi
                end
                if worst_over_all === nothing || s.maxN > worst_over_all
                    worst_over_all = s.maxN
                    worst_form     = fi
                end
            end

            # The form's line: the N of the run on F itself, before any random
            # change of basis. At the default it is the whole answer, and with
            # restarts it is what they had to beat.
            options.trials > 0 && println(io, "")
            println(io, "Type $fi: $(res.directN === nothing ? "none" : res.directN)")

            # With restarts, every N is printed in the order drawn, then the
            # smallest with the presentation that produced it (the direct run
            # being presentation 1) and the number of distinct N seen.
            # `res.Ns` holds the direct N first, then one entry per random
            # presentation actually run.
            if options.trials > 0
                shown = [N === nothing ? "-" : string(N) for N in res.Ns[2:end]]
                println(io, "  random N = " * (isempty(shown) ? "(none run)" : join(shown, " ")))
                if res.bestN === nothing
                    println(io, "  best N   = none")
                else
                    j = findfirst(==(res.bestN), res.Ns)
                    println(io, "  best N   = $(res.bestN)   " *
                                "(presentation $j of $(s.attempted), $(s.distinct) distinct)")
                end
            end
            d == 16 * pZ^2 || push!(odd_det, fi)
            s.n_lll      > 0 && push!(used_lll,      fi)
            s.n_complete > 0 && push!(used_complete, fi)
            s.n_error    > 0 && push!(had_error,     fi)
            flush(io)

            # Progress. The spread is only worth printing when more than one
            # presentation was run.
            @info "  form $fi of $lo:$hi: N = $(res.bestN === nothing ? "none" : res.bestN)" *
                  (s.attempted > 1 ?
                      "  (spread $(s.minN === nothing ? "-" : s.minN)…$(s.maxN === nothing ? "-" : s.maxN), " *
                      "$(s.distinct) distinct over $(s.attempted))" : "") *
                  "  $(fmt_elapsed(res.total_secs))"
        end

        # The extremes of N over the forms processed, with the form attaining
        # each.
        if best_over_all !== nothing
            println(io, "")
            println(io, "=" ^ 78)
            println(io, "min N = $best_over_all   (Type $best_form)")
            println(io, "max N = $worst_over_all   (Type $worst_form)")
            println(io, "=" ^ 78)
        end

        # The run's totals and checks go to the terminal rather than the
        # file: the file is a list of N and nothing else. Each check is silent
        # when it has nothing to say, so a clean run prints one line and a run
        # with an oddity names the forms responsible.
        @info "Finished forms $lo:$hi of $nforms" *
              "  smallest N = $(best_over_all === nothing ? "none" : "$best_over_all (form $best_form)")" *
              "  witnesses = $(hi - lo + 1 - n_none)/$(hi - lo + 1)" *
              "  N = 1 on $n_one form(s)" *
              "  solver $(fmt_elapsed(all_secs)), wall $(fmt_elapsed(time() - t_start))"
        isempty(odd_det) ||
            @warn "det A ≠ 16p² on form(s) $(form_list(odd_det))"
        length(fam.glue) == 1 ||
            @warn "$(length(fam.glue)) glued lattices built, expected 1 — more than one genus present"
        isempty(used_lll) ||
            @info "the indefinite LLL alone solved form(s) $(form_list(used_lll))"
        isempty(used_complete) ||
            @warn "form(s) $(form_list(used_complete)) fell outside the specialised route " *
                  "and needed Hecke's general solver"
        isempty(had_error) ||
            @warn "solver error on form(s) $(form_list(had_error))"
    end

    return outpath
end

# ══════════════════════════════════════════════════════════════════════════════
# Parsing <p> from the command line
# ══════════════════════════════════════════════════════════════════════════════

"""
    parse_prime_expr(s) -> BigInt

Parse `s` as a plain non-negative integer, or as an arithmetic expression over
non-negative integer literals and `+ - * ^ ( )` (standard precedence, unary `-`
allowed), evaluated entirely in BigInt — e.g. `"2^721*3^176-1"`.

Parsing is delegated to Julia's own parser via `Meta.parse`; the resulting
expression tree is then walked and evaluated node by node in BigInt. Julia's
`eval` is never called, and a node is evaluated only after being confirmed to
be an integer literal or one of the four permitted operators, so the string
cannot run arbitrary code. Doing the arithmetic by hand also keeps explicit
control of the types — in particular `^` always takes an `Int` exponent.
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

# ══════════════════════════════════════════════════════════════════════════════
# Command-line entry point
# ══════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    let
        SCRIPT = basename(@__FILE__)

        function usage()
            println(stderr, """
Usage: julia $(SCRIPT) <p> [--dir <dir>] [--first <i>] [--last <j>]
                       [--trials <n>] [--steps <n>] [--coeff-bound <C>]
                       [--seed <s>] [--no-fallback] [--no-stop-at-one]

  <p>                 prime whose RHI2 file is processed (required).
                      Either a plain integer (e.g. 23), or an arithmetic
                      expression over + - * ^ ( ) evaluated exactly
                      (e.g. 2^721*3^176-1).
  --dir <dir>         input/output directory  (default: script directory)
  --first <i>         first form of the file to process  (default: 1)
  --last <j>          last form of the file to process   (default: the last)
                      Form i is form i of the file whatever range it is
                      computed in, and carries the same seed, so ranges are
                      independent runs that may be done in any order, on any
                      number of machines, and concatenated. Output for a
                      partial range is written to a file naming that range.
  --trials <n>        extra random presentations per form, after the direct
                      run  (default: 0, i.e. one presentation and one square).
                      Each extra presentation is another draw at the same
                      problem; the smallest N after n draws falls roughly
                      like 1/(n+1).
  --steps <n>         elementary row operations composed into each random
                      U ∈ GL₆(ℤ)  (default: 12)
  --coeff-bound <C>   bound on the multiplier in a transvection row operation
                      (default: 2)
  --seed <s>          seed of the random number generator, so a run
                      reproduces exactly  (default: 20260904)
  --no-fallback       do not try routes 2 and 3 when the indefinite LLL
                      misses; record the presentation as a miss instead.
                      NOT a speed setting: on forms with det A = 16p² the
                      indefinite LLL alone essentially never succeeds, so
                      this discards every witness. Use it only to measure
                      that hit rate on a family whose answer is not already
                      known.
  --no-stop-at-one    keep running all --trials presentations even after
                      N = 1 has been reached. N = 1 is the smallest N can be,
                      so this only ever costs time — pass it when the spread
                      of N is itself what is being measured.

One process runs its forms one after another: each form reseeds the global
random number generator, so forms cannot run concurrently *within* a process
without losing reproducibility. Ranges in separate processes share nothing
and are the way to use more cores.

Examples:
  julia $(SCRIPT) 11
  julia $(SCRIPT) 2^11*3^24-1                      # whole file, one process
  julia $(SCRIPT) 1187 --trials 30 --steps 12 --coeff-bound 2
  julia $(SCRIPT) 23 --trials 100 --no-stop-at-one
  julia $(SCRIPT) 2^721*3^176-1 --dir ../other_dir
  julia $(SCRIPT) 23 --trials 50 --no-fallback     # measure the LLL hit rate

  # Ranges are for runs with many trials, where memory would otherwise grow:
  # 20 ranges of 500 with 30 trials each, eight ranges running at a time.
  seq 0 19 | xargs -P 8 -I% sh -c 'julia $(SCRIPT) 2^11*3^24-1 --trials 30 --first \$((%*500+1)) --last \$((%*500+500))'
""")
            exit(1)
        end

        length(ARGS) < 1 && usage()

        local p = parse_prime_expr(ARGS[1])
        p > 0 || error("parsed prime expression \"$(ARGS[1])\" evaluated to $p, which is not a positive integer")

        local dir         = dirname(abspath(@__FILE__))
        local first_form  = 1
        local last_form   = nothing
        local trials      = 0
        local steps       = 12
        local coeff_bound = 2
        local seed        = 20260904
        local fallback    = true
        local stop_at_one = true

        local i = 2
        while i <= length(ARGS)
            flag = ARGS[i]
            if flag == "--dir"
                i + 1 > length(ARGS) && (println(stderr, "Error: --dir requires an argument"); usage())
                dir = ARGS[i+1]; i += 2
            elseif flag == "--first"
                i + 1 > length(ARGS) && (println(stderr, "Error: --first requires an argument"); usage())
                first_form = tryparse(Int, ARGS[i+1])
                (first_form === nothing || first_form < 1) &&
                    (println(stderr, "Error: --first requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--last"
                i + 1 > length(ARGS) && (println(stderr, "Error: --last requires an argument"); usage())
                last_form = tryparse(Int, ARGS[i+1])
                (last_form === nothing || last_form < 1) &&
                    (println(stderr, "Error: --last requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--trials"
                i + 1 > length(ARGS) && (println(stderr, "Error: --trials requires an argument"); usage())
                trials = tryparse(Int, ARGS[i+1])
                (trials === nothing || trials < 0) &&
                    (println(stderr, "Error: --trials requires a non-negative integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--steps"
                i + 1 > length(ARGS) && (println(stderr, "Error: --steps requires an argument"); usage())
                steps = tryparse(Int, ARGS[i+1])
                (steps === nothing || steps < 1) &&
                    (println(stderr, "Error: --steps requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--coeff-bound"
                i + 1 > length(ARGS) && (println(stderr, "Error: --coeff-bound requires an argument"); usage())
                coeff_bound = tryparse(Int, ARGS[i+1])
                (coeff_bound === nothing || coeff_bound < 1) &&
                    (println(stderr, "Error: --coeff-bound requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--seed"
                i + 1 > length(ARGS) && (println(stderr, "Error: --seed requires an argument"); usage())
                seed = tryparse(Int, ARGS[i+1])
                seed === nothing &&
                    (println(stderr, "Error: --seed requires an integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--no-fallback"
                fallback = false; i += 1
            elseif flag == "--no-stop-at-one"
                stop_at_one = false; i += 1
            else
                println(stderr, "Error: unknown argument \"$flag\""); usage()
            end
        end

        isdir(dir) || (println(stderr, "Error: directory not found: \"$dir\""); exit(1))

        (last_form === nothing || first_form <= last_form) ||
            (println(stderr, "Error: --first $first_form is after --last $last_form"); exit(1))

        opts = Options(; trials, steps, coeff_bound, seed, fallback, stop_at_one)
        @info "Starting" p dir first_form last_form trials steps coeff_bound seed fallback stop_at_one
        outpath = process_rhi2(p; dir, options = opts, first_form, last_form)
        println("Done → $outpath")
    end
end
