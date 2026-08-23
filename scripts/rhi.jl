using Oscar

"""
    check_p_11mod12(p_big) -> BigInt

Validate that `p_big` is (probably) prime and congruent to 11 mod 12,
returning it unchanged. The RHI construction in `rhi2` assumes both: p ≡ 3
(mod 4) is what makes `B_{p,∞}` the algebra ramified at p and ∞ with the
reduced-norm form used here, and p ≡ 11 (mod 12) is the stronger convention
shared with `polz_all.jl` and every prime under `data/`, so the scripts accept
the same inputs. Mirrors `check_p_11mod12` in `scripts/polz_random.jl`.
"""
function check_p_11mod12(p_big::BigInt)::BigInt
    is_probable_prime(ZZ(p_big)) || error("p must be prime; got p=$p_big")
    mod(p_big, 12) == 11 ||
        error("p must be 11 mod 12; got p mod 12 = $(mod(p_big, 12))")
    return p_big
end

"""
    default_output_filename(p_big, N, prefix)

Given a prime `p_big` (as a BigInt) and a number `N`, returns the filename used
for both the polarization input file and the RHI2 output file, matching the
naming convention used by the polarization generator:

  - If `p_big` has 24 or fewer bits, the file is named "<prefix>_<p>_<N>.txt".
  - Otherwise (big primes, 25+ bits), the file is named
    "<prefix>_<bits>bit_<digest>_<N>.txt", where `bits` is the bit-length of
    `p_big` and `digest` is the low 32 bits of `hash(string(p_big))` in hex.

`prefix` is either "polz" (input) or "RHI2" (output).
"""
function default_output_filename(p_big::BigInt, N::Integer, prefix::String)::String
    bits = ndigits(p_big, base = 2)
    if bits <= 24
        return "$(prefix)_$(string(p_big))_$(N).txt"
    end
    s = string(p_big)
    digest = string(hash(s) & 0xffffffff; base = 16, pad = 8)
    return "$(prefix)_$(bits)bit_$(digest)_$(N).txt"
end

# Accept either "# p = 11" or "p = 11" headers. Compiled once at module
# load instead of on every file_reader call.
const _POL_PRIME_RE = r"^\s*#?\s*p\s*=\s*(\d+)"
const _POL_DATA_RE = r"^\s*\[\d+\]\s+(.+)$"

"""
    _parse_polarization_value(s) -> BigInt

Parse a single polarization coordinate as a `BigInt`. Polarization
coordinates (`u0, v0, w0, x0, y0, z0`) come from a bounded
classification search and almost always fit in `Int128`; the prime
`p` itself is handled separately and can be much larger. Values are
tried as `Int128` first and only parsed as `BigInt` if that fails,
which is cheaper for the common case while still handling arbitrarily
large values correctly.
"""
function _parse_polarization_value(s::AbstractString)::BigInt
    v = tryparse(Int128, s)
    return v === nothing ? parse(BigInt, s) : BigInt(v)
end

"""
    file_reader(filename) -> (prime, polarizations)

Reads a polarization input file named per `default_output_filename(p, N, "polz")`
(i.e. "polz_<p>_<N>.txt" for primes of 24 bits or fewer, or
"polz_<bits>bit_<digest>_<N>.txt" for larger primes). Each non-blank line is one of:

  - a prime header, accepted as either "# p = <p>" or "p = <p>"
  - a comment line starting with `#` (skipped)
  - a data line of 6 whitespace-separated values (optionally prefixed with an
    index like "[1]"), the `u0, v0, w0, x0, y0, z0` polarization coordinates

Returns the parsed `prime` (or `nothing` if no header line was present) and a
`Vector` of 6-element `Vector{BigInt}` polarizations, in file order.
"""
function file_reader(filename::String)
    prime = nothing
    polarizations = Vector{Vector{BigInt}}()

    for line in eachline(filename)
        raw = strip(line)
        if isempty(raw)
            continue
        end

        m = match(_POL_PRIME_RE, raw)
        if m !== nothing
            # p itself can be cryptographic size, so always go straight to BigInt.
            prime = parse(BigInt, m.captures[1])
            continue
        end

        if startswith(raw, "#")
            continue
        end

        data = match(_POL_DATA_RE, raw)
        if data !== nothing
            raw = data.captures[1]
        end

        nums = split(raw)
        if length(nums) >= 6
            # Non-numeric lines (e.g. the "order basis: [1, i, ...]" header,
            # which also splits into 6 fields) land here and are skipped.
            # Catch only parse failures -- a bare `catch` would also swallow
            # InterruptException, making Ctrl-C during a large read a no-op.
            try
                values = [_parse_polarization_value(nums[i]) for i in 1:6]
                push!(polarizations, values)
            catch err
                err isa ArgumentError || rethrow()
                continue
            end
        end
    end

    return prime, polarizations
end


"""
    rhi2(p, param) -> NamedTuple{(:coeff_matrix, :det)} | Nothing

Given `Bp = (-1, -p | Q)` and polarization `param = [u0, v0, w0, x0, y0, z0]`,
compute the coefficient matrix of the 5-ary refined Humbert invariant
that does not represent 1.

Returns `nothing` if the reduced lattice represents 1 (`minimum(L) <= 1`),
which is the one legitimate way a well-formed polarization can fail to
yield the invariant. This is a normal outcome, not an error: measured
over `data/`, it accounts for ~2% of polarizations at p=227 and ~0.4%
at p=1619, and none at all at 50 bits and above.

Every other failure -- the form not being positive semidefinite, or the
LLL-reduced Gram matrix not having rank 5 with a zero last row, or the
5x5 block not being positive definite -- means the construction itself
is broken for this `p`, not that this polarization is uninteresting.
Those raise rather than returning `nothing`; none of them occurred in
8000 polarizations sampled across `data/polz_small_100k` and
`data/polz_big_10k`.

On success, returns a named tuple `(coeff_matrix, det)`:

  - `coeff_matrix`: the `ZZMatrix` coefficient matrix.
  - `det`: the determinant of the Gram matrix `coeff_matrix .÷ 2`,
    i.e. the determinant of the LLL-reduced 5x5 lattice before it is
    doubled into `coeff_matrix`.

For any valid polarization this determinant equals `2^4 * p^2`,
independent of the polarization coordinates; it is computed and
checked against that closed form on every successful call as a
correctness check on the construction, not a per-polarization
validity filter, and a mismatch raises. Because `p` is fixed for an entire `all_rhi2(p, N)`
run, `det` is identical across every valid polarization in that run,
so it carries no discriminating information for bucketing candidates
(see `all_rhi2`); it is returned here for logging/sanity purposes.
"""
function rhi2(p::Integer, param::AbstractVector{<:Integer})
    u0, v0, w0, x0, y0, z0 = param
    p2 = p * p

    A = zero_matrix(QQ, 6, 6)

    # Fill diagonal
    A[1, 1] = 2 * v0^2
    A[2, 2] = 2 * u0^2
    A[3, 3] = 8 * w0^2 + 8 * w0 * z0 + 2 * z0^2 + 8
    A[4, 4] = 8 * x0^2 + 8 * x0 * y0 + 2 * y0^2 + 8
    A[5, 5] = 2 * x0^2 + 2 * x0 * y0 * p + 2 * x0 * y0 + QQ(1, 2) * y0^2 * p2 + y0^2 * p + QQ(1, 2) * y0^2 + 2 * p + 2
    A[6, 6] = 2 * w0^2 - 2 * w0 * z0 * p + 2 * w0 * z0 + QQ(1, 2) * z0^2 * p2 - z0^2 * p + QQ(1, 2) * z0^2 + 2 * p + 2

    # Fill upper triangular off-diagonals
    A[1, 2] = 2 * u0 * v0 - 4
    A[1, 3] = -4 * v0 * w0 - 2 * v0 * z0
    A[1, 4] = -4 * v0 * x0 - 2 * v0 * y0
    A[1, 5] = -2 * v0 * x0 - v0 * y0 * p - v0 * y0
    A[1, 6] = -2 * v0 * w0 + v0 * z0 * p - v0 * z0

    A[2, 3] = -4 * u0 * w0 - 2 * u0 * z0
    A[2, 4] = -4 * u0 * x0 - 2 * u0 * y0
    A[2, 5] = -2 * u0 * x0 - u0 * y0 * p - u0 * y0
    A[2, 6] = -2 * u0 * w0 + u0 * z0 * p - u0 * z0

    A[3, 4] = 8 * w0 * x0 + 4 * w0 * y0 + 4 * x0 * z0 + 2 * y0 * z0
    A[3, 5] = 4 * w0 * x0 + 2 * w0 * y0 * p + 2 * w0 * y0 + 2 * x0 * z0 + y0 * z0 * p + y0 * z0
    A[3, 6] = 4 * w0^2 - 2 * w0 * z0 * p + 4 * w0 * z0 - z0^2 * p + z0^2 + 4

    A[4, 5] = 4 * x0^2 + 2 * x0 * y0 * p + 4 * x0 * y0 + y0^2 * p + y0^2 + 4
    A[4, 6] = 4 * w0 * x0 + 2 * w0 * y0 - 2 * x0 * z0 * p + 2 * x0 * z0 - y0 * z0 * p + y0 * z0

    A[5, 6] = 2 * w0 * x0 + w0 * y0 * p + w0 * y0 - x0 * z0 * p + x0 * z0 - QQ(1, 2) * y0 * z0 * p2 + QQ(1, 2) * y0 * z0

    # Mirror the upper triangular part to the lower triangular part
    for i = 2:6
        for j = 1:(i-1)
            A[i, j] = A[j, i]
        end
    end

    # Check A is positive semidefinite. The RHI construction guarantees
    # this for every polarization of a prime p = 11 mod 12, so a failure
    # here is a broken computation (or a p violating that congruence),
    # not a polarization to skip over.
    V = quadratic_space(QQ, A)
    D = diagonal(V)
    if !all(>=(0), D)
        error("rhi2: Gram matrix is not positive semidefinite (diagonal = $D) for p = $p, param = $param")
    end

    Azz = map_entries(x -> ZZ(x//2), A)
    AA = lll_gram(Azz)

    # The form has rank 5 in a 6-dimensional space, and lll_gram sorts the
    # zero vector into the last row. Both are structural, so -- as above --
    # a failure means the construction is broken rather than that this
    # polarization should be skipped.
    if rank(AA) != 5 || !is_symmetric(AA) || any(!iszero, AA[6, :])
        error("rhi2: LLL-reduced Gram matrix is not rank 5 with a zero last row " *
              "(rank = $(rank(AA)), symmetric = $(is_symmetric(AA))) for p = $p, param = $param")
    end

    # Work with top-left 5×5 submatrix
    B = @view AA[1:5, 1:5]

    L = integer_lattice(; gram = B)

    # Positive definiteness of the rank-5 block is structural, like the
    # checks above.
    if !is_positive_definite(L)
        error("rhi2: rank-5 block is not positive definite for p = $p, param = $param")
    end

    # This one is the genuine per-polarization filter: the refined Humbert
    # invariant we want is the form that does *not* represent 1. A lattice
    # with minimum 1 is a normal, expected outcome, so return nothing.
    if minimum(L) <= 1
        return nothing
    end

    C = B .* 2
    det_B = det(B)

    # Structural invariant of this construction: for any valid
    # polarization the Gram matrix's determinant is exactly 2^4 * p^2,
    # independent of the polarization coordinates. A violation means
    # something is wrong with the computation itself. Raised rather than
    # `@assert`ed so it cannot be compiled out under --check-bounds=no or
    # a future -O setting that elides assertions.
    expected_det = 16 * BigInt(p)^2
    if BigInt(det_B) != expected_det
        error("rhi2: det(Gram matrix) = $(det_B), expected 2^4*p^2 = $(expected_det) for p = $p, param = $param")
    end

    return (coeff_matrix = C, det = det_B)
end


"""
    poly_form(M)

Given the coefficient matrix M of a quadratic form, this function computes the polynomial form using f = 1/2 * (X^t * M * X).

"""
function poly_form(M::ZZMatrix)
    n = number_of_rows(M)
    R, x = polynomial_ring(ZZ, n)
    f = R(0)
    # Use the formula: f = 1/2 * Σᵢ M[i,i]*x[i]^2 + Σ₍ᵢ<ⱼ₎ M[i,j]*x[i]*x[j]
    for i = 1:n
        f += (M[i, i] ÷ 2) * x[i]^2
        for j = (i+1):n
            f += M[i, j] * x[i] * x[j]
        end
    end
    return f
end


"""
    condensed_prime_repr(p_big)

Returns a short, human-readable representation of a prime for console/log
output. Small primes (24 bits or fewer) are shown in full. Large primes are
shown as their bit-length and digest, matching the identifiers used in
filenames, instead of printing the full decimal expansion.
"""
function condensed_prime_repr(p_big::BigInt)::String
    bits = ndigits(p_big, base = 2)
    if bits <= 24
        return string(p_big)
    end
    s = string(p_big)
    digest = string(hash(s) & 0xffffffff; base = 16, pad = 8)
    return "$(bits)-bit prime (digest $(digest))"
end

"""
    _mat_big(M) -> Matrix{BigInt}

Convert a Nemo/Hecke matrix (`ZZMatrix` or similar) to `Matrix{BigInt}`.
Used instead of a plain `Matrix{Int}` conversion throughout this file
because entries built from a large `p` can exceed `Int64`.
"""
_mat_big(M) = Matrix{BigInt}([BigInt(M[i, j]) for i in 1:nrows(M), j in 1:ncols(M)])

_det_big_mat(T::Matrix{BigInt}) = BigInt(det(matrix(ZZ, T)))

# ============================================================
# Bucketing invariants
# ============================================================
# `all_rhi2` deduplicates candidate forms up to isometry. Testing every
# pair of candidates with the exact (and expensive) Plesken-Souvignier
# isometry search would be quadratic in the number of candidates, so
# candidates are first bucketed on a cheap, genuine isometry invariant
# (a quantity that is provably equal for isometric lattices, so
# bucketing on it can never produce a false negative). Only forms
# sharing this value are ever compared with the exact isometry test.
#
# The key used here is the *minimum-anchored* theta profile
# (`theta_anchored` below): the lattice minimum `m` together with the
# vector counts at squared lengths `m, m+1, ..., m+Tmax`. Both parts
# are genuine isometry invariants, so the no-false-negative property
# is preserved.
#
# The anchoring is what makes the key useful. An *absolute* cutoff --
# counting vectors of squared length `1..Tmax` for a small fixed
# `Tmax` -- is worthless on this data, because `rhi2` only accepts
# lattices with `minimum(L) > 1` and `det(Gram) = 16*p^2` pushes the
# minimum up with `p`. Measured on the corpora under `data/`:
#
#     data/polz_small_100k, p=227    min(L) ~ 5..16      1467 candidates -> 9 distinct keys
#     data/polz_small_100k, p=1619   min(L) larger       1495 candidates -> 3 distinct keys
#     data/polz_big_10k, 50-bit      min(L) ~ 1e6        1500 candidates -> 1 distinct key
#     data/polz_big_10k, 100-bit     min(L) ~ 1e12       every key all-zero
#
# i.e. at 50 bits and above every candidate hashed to the same
# all-zero key and the bucketing did no work at all. Anchoring at the
# minimum instead gives 1077 / 1474 / 799 distinct keys on the same
# three samples, for the same cost. End to end on 3000 polarizations
# at p=227 that is 1_311 exact isometry calls instead of 1_264_425,
# for an identical class count.
#
# The determinant of the Gram matrix is also a genuine isometry
# invariant, but is *not* used as a bucket key here: `rhi2` shows it
# equals `16*p^2` for every valid polarization, and `p` is fixed for
# the duration of one `all_rhi2(p, N)` run, so every candidate in a
# given run shares the same determinant. Bucketing on a value that
# never varies within a run does no discriminating work -- it would
# only add a constant key to every bucket lookup and hash. The Smith
# normal form of the Gram matrix is constant within a run for the same
# reason (measured: 1 distinct value across every sample above).
#
# Local genus symbols are not used either: a genus symbol at `p` is
# expensive to compute for the large primes this file supports (see
# `parse_big_prime`). The kissing number needs no separate key -- it
# is the first entry of the anchored profile by construction.

"""
    theta_anchored(L, Tmax::Int) -> Tuple{ZZRingElem, Vector{Int}}

`(m, [r_m, r_{m+1}, ..., r_{m+Tmax}])`, where `m = minimum(L)` and
`r_k` is the number of vectors of `L` of squared length `k`, computed
via Hecke's Fincke-Pohst `short_vectors`. `short_vectors` returns
vectors up to sign, so each is counted twice; `r_m` is therefore the
kissing number.

Isometric lattices have the same minimum and the same theta series, so
both components are genuine isometry invariants and this is safe to
bucket on. Unlike an absolute `1..Tmax` window it stays informative
however large the entries of the Gram matrix grow -- see the
"Bucketing invariants" note above for the measurements motivating it.
"""
function theta_anchored(L, Tmax::Int)
    # Hecke returns the minimum as a QQFieldElem. Every Gram matrix reaching
    # here is a ZZMatrix, so all norms are integral; narrowing to ZZRingElem
    # keeps the bucket key exactly hashable and matches the counts' index type.
    m = ZZ(minimum(L))
    counts = zeros(Int, Tmax + 1)
    for entry in short_vectors(L, m, m + Tmax)
        n = Int(ZZ(entry[2]) - m)
        if 0 <= n <= Tmax
            counts[n+1] += 2
        end
    end
    return (m, counts)
end

function _inv_unimodular_bigint(T::Matrix{BigInt})
    Tzz = matrix(ZZ, T)
    d = det(Tzz)
    @assert d == 1 || d == -1 "matrix is not unimodular; det=$d"
    return _mat_big(inv(Tzz))
end

"""
    _verify_transport(T, F, G) -> Bool

Check that `T`, as returned by Hecke's `isometry`, conjugates one Gram
matrix to the other, trying each transport convention (row/column,
`G->F`/`F->G`, `T^{-1}`) in turn with early return. Operates on
`Vector{ZZMatrix}`/`Matrix{BigInt}` throughout so it stays correct for
arbitrarily large entries.
"""
function _verify_transport(T::Matrix{BigInt}, F::Vector{ZZMatrix}, G::Vector{ZZMatrix})
    Fb = [_mat_big(A) for A in F]
    Gb = [_mat_big(A) for A in G]
    Tt = transpose(T)
    idx = eachindex(Fb)
    all(k -> T * Gb[k] * Tt == Fb[k], idx) && return true
    all(k -> T * Fb[k] * Tt == Gb[k], idx) && return true
    all(k -> Tt * Gb[k] * T == Fb[k], idx) && return true
    all(k -> Tt * Fb[k] * T == Gb[k], idx) && return true
    if _det_big_mat(T) == 1 || _det_big_mat(T) == -1
        Ti = _inv_unimodular_bigint(T)
        Tit = transpose(Ti)
        all(k -> Ti * Fb[k] * Tit == Gb[k], idx) && return true
    end
    return false
end

_diagmax(A::ZZMatrix) = maximum(A[i, i] for i in 1:min(nrows(A), ncols(A)))

"""
    _choose_input_output(F, G) -> (Fin, Gout)

Hecke's isometry setup is cheaper when it starts from the lattice with
the smaller-normed diagonal, so return that one first.
"""
function _choose_input_output(F::Vector{ZZMatrix}, G::Vector{ZZMatrix})
    bound_f, bound_g = _diagmax(F[1]), _diagmax(G[1])
    bound_g < bound_f ? (G, F) : (F, G)
end

function _try_setup(FF::Vector{ZZMatrix}, GG::Vector{ZZMatrix}; depth::Int=0, bacher_depth::Int=0)
    fl, CF, CG = Hecke._try_iso_setup_small(FF, GG; depth=depth, bacher_depth=bacher_depth)
    fl && return CF, CG
    return Hecke._iso_setup(FF, GG; depth=depth, bacher_depth=bacher_depth)
end

"""
    _hecke_isometric(A, B; depth=0, bacher_depth=0, verify=true) -> Bool

Exact isometry test between two positive-definite Gram matrices via
Hecke's small-setup Plesken-Souvignier pipeline (`Hecke.isometry`).
This bypasses Oscar's high-level `is_isometric`/
`is_isometric_with_isometry` wrapper for `ZZLat`, which for the
definite case funnels into the same underlying `_try_iso_setup_small`/
`_iso_setup`/`Hecke.isometry` core called directly here, but adds
overhead this function avoids:

  1. The wrapper unconditionally re-runs `lll_gram_with_transform` on
     both operands on every call, with no caching. `A` and `B` here
     are already LLL-reduced (they come straight out of `rhi2`'s
     `lll_gram`), so re-reducing them is wasted work that grows more
     expensive as `p`, and hence the matrix entries, grow.
  2. The wrapper always treats its first argument as the enumeration
     "input" side, with no equivalent of `_choose_input_output`'s
     smaller-diagonal heuristic, so it can end up doing the expensive
     vector-sum/Bacher-polynomial setup on the larger-normed side.
  3. The wrapper builds full `ZZLat`/`quadratic_space` wrapper objects
     around each Gram matrix instead of operating on bare `ZZMatrix`
     values.

Stays in `ZZMatrix`/`BigInt` throughout, so there is no `Int` overflow
risk for large primes. `verify` double-checks any transform Hecke
returns against the original Gram matrices.
"""
function _hecke_isometric(A::ZZMatrix, B::ZZMatrix; depth::Int=0, bacher_depth::Int=0, verify::Bool=true)
    Fin, Gout = _choose_input_output(ZZMatrix[A], ZZMatrix[B])
    CF, CG = _try_setup(Fin, Gout; depth=depth, bacher_depth=bacher_depth)
    b, raw_transform = Hecke.isometry(CF, CG)
    if Bool(b) && verify
        _verify_transport(_mat_big(raw_transform), Fin, Gout) || error("RHI2 isometry verification failed")
    end
    return Bool(b)
end

"""
    all_rhi2(p, N; verify=true, Tmax=40)

Given a prime `p` congruent to 11 mod 12, read the polarizations for
`p` from the `N`-indexed "polz" input file (see
`default_output_filename`), compute the 5-ary refined Humbert
invariant coefficient matrix for each via `rhi2`, and write one entry
per isometry-class representative found to the matching "RHI2" output
file, along with summary statistics (polarizations checked, RHI's
computed, rejection counts by reason, duplicate counts per type,
elapsed time).

Errors if the input file's header prime does not match `p`, rather
than silently producing an empty result.

Isometry classes are found in two stages:
1. An exact hash lookup on the flattened Gram matrix entries catches
   identical forms in O(1).
2. Remaining candidates are bucketed on `theta_anchored(L, Tmax)`
   -- a genuine isometry invariant -- and `_hecke_isometric` (an
   exact Plesken-Souvignier isometry test) is only run between forms
   sharing a bucket. The Gram matrix determinant is also a genuine
   isometry invariant, but is constant across an entire run (see
   `rhi2`'s docstring: it equals `16*p^2` for fixed `p`), so it adds
   no discriminating power as a bucket key and is not used as one.

`verify` is forwarded to `_hecke_isometric` to double-check every
isometry transform Hecke returns against the original Gram matrices
before trusting it.

`Tmax` sets how many squared lengths above the lattice minimum
`theta_anchored` profiles. The default of 40 was chosen by measurement
across `data/polz_small_100k` and `data/polz_big_10k` -- see the
"Bucketing invariants" note above. Smaller values degrade gracefully
(they only make buckets coarser, never incorrect) but cost real time:
at `Tmax=5` the same p=227 sample yields 196 buckets instead of 1077.

Limitations:
- This function is single-threaded by design. Its hot paths (`rhi2`'s
  `lll_gram`/`is_positive_definite`/`minimum`, `theta_anchored`'s
  `short_vectors`, `det`, and `_hecke_isometric`'s `Hecke.isometry`)
  all call into FLINT via Nemo/Hecke, whose C-level state is not
  documented as safe for concurrent calls from independent Julia
  OS-threads. To parallelize across polarizations, use `Distributed.jl`
  (separate OS processes, so no shared FLINT state) rather than
  `Threads.@threads`. Note the workload is also embarrassingly
  parallel *across primes*: one process per input file needs no
  coordination at all.
- Theta bucketing rules out false negatives but is not a full isometry
  invariant on its own; forms sharing a bucket still require the exact
  `_hecke_isometric` test, so a run with many candidates sharing the
  same anchored profile will still be slow.
"""
function all_rhi2(p::Integer, N::Integer; verify::Bool=true, Tmax::Int=40)
    start_time = time()
    p_big = BigInt(p)
    check_p_11mod12(p_big)
    pdisp = condensed_prime_repr(p_big)
    println("working with prime ", pdisp)

    input_filename = default_output_filename(p_big, N, "polz")
    output_filename = default_output_filename(p_big, N, "RHI2")

    # Read the input *before* opening the output for writing: `open(_, "w")`
    # truncates, so doing it the other way round leaves an empty RHI2 file
    # behind whenever the input is missing or unreadable.
    isfile(input_filename) ||
        error("all_rhi2: no polarization file $(input_filename) in $(pwd()). " *
              "Input paths are relative to the working directory, so run this " *
              "from the directory holding the polz_* files.")
    prime, params = file_reader(input_filename)
    count = length(params)

    prime === nothing &&
        error("all_rhi2: $(input_filename) has no `p = ...` header line.")
    prime == p_big ||
        error("all_rhi2: $(input_filename) is for p = $(condensed_prime_repr(prime)), " *
              "but was asked for p = $(pdisp).")

    file = open(output_filename, "w")
    try
        println(file, "p = ", p_big, "\n")

        idx = 0   # Counting unique forms.
        total = 0 # Total RHIs computed.
        represents_one = 0  # Polarizations rejected by rhi2's minimum(L) > 1 filter.

        # unique_forms[k] holds the data for the k-th unique form found so far:
        #   coeff_matrix - the ZZMatrix coefficient matrix
        #   theta        - theta_anchored(L, Tmax): the lattice minimum plus the
        #                  vector counts at the next Tmax squared lengths, used
        #                  as the bucket key (see the "Bucketing invariants"
        #                  note above).
        unique_forms = NamedTuple{(:coeff_matrix, :theta),
                                   Tuple{ZZMatrix,Tuple{ZZRingElem,Vector{Int}}}}[]

        # Maps a theta_anchored key to the list of indices (into unique_forms)
        # of unique forms sharing it. Only forms sharing a bucket are ever
        # compared via the expensive isometry call.
        theta_buckets = Dict{Tuple{ZZRingElem,Vector{Int}},Vector{Int}}()

        # Flattened Gram entries (as BigInt, since entries can exceed Int64 once
        # p is large) => index into unique_forms. An O(1) hash check that
        # catches identical forms outright.
        exact_lookup = Dict{NTuple{25,BigInt},Int}()

        pol_count = Dict{Int,Int}()

        for param in params
            res = rhi2(p_big, param)  # (coeff_matrix, det) or nothing.
            if res === nothing
                represents_one += 1
            else
                coeff_matrix = res.coeff_matrix
                total += 1
                is_unique = true

                # O(1) exact-equality check via hash lookup.
                exact_key = NTuple{25,BigInt}(vec(_mat_big(coeff_matrix)))
                existing_k = get(exact_lookup, exact_key, 0)
                if existing_k != 0
                    is_unique = false
                    pol_count[existing_k] = get(pol_count, existing_k, 1) + 1
                end

                local theta_val, bucket_key
                if is_unique
                    # theta_anchored as a zero-risk isometry-invariant
                    # filter -- see the note on unique_forms above.
                    L = integer_lattice(; gram = coeff_matrix .÷ 2)
                    theta_val = theta_anchored(L, Tmax)
                    bucket_key = theta_val

                    # Only compare against forms sharing the anchored theta
                    # profile; no isometric lattice can fall outside this
                    # bucket.
                    # Every candidate in the bucket still gets the exact
                    # isometry test below -- there is no further
                    # pre-filter, because the diagonal of an LLL-reduced
                    # Gram matrix is not itself a genuine isometry
                    # invariant (LLL-reduced bases are not canonical, so
                    # isometric lattices reduced from different starting
                    # bases can land on different diagonals).
                    candidates = get(theta_buckets, bucket_key, Int[])
                    for k in candidates
                        u = unique_forms[k]
                        # Direct Hecke Plesken-Souvignier isometry test,
                        # skipping Oscar's high-level is_isometric wrapper
                        # -- see _hecke_isometric's docstring.
                        if _hecke_isometric(coeff_matrix, u.coeff_matrix; verify=verify)
                            is_unique = false
                            pol_count[k] = get(pol_count, k, 1) + 1
                            break
                        end
                    end
                end

                if is_unique
                    idx += 1
                    println(file, "Type ", idx)
                    # s4 = param[4] >= 0 ? "+" : "-"
                    # s5 = param[5] >= 0 ? "+" : "-"
                    # s6 = param[6] >= 0 ? "+" : "-"
                    # a4 = abs(param[4])
                    # a5 = abs(param[5])
                    # a6 = abs(param[6])

                    # s4b = (-param[4]) >= 0 ? "+" : "-"
                    # s5b = (-param[5]) >= 0 ? "+" : "-"
                    # s6b = (-param[6]) >= 0 ? "+" : "-"
                    # a4b = abs(-param[4])
                    # a5b = abs(-param[5])
                    # a6b = abs(-param[6])

                    # println(
                    #     file,
                    #     "θ = [",
                    #     param[1],
                    #     "  ",
                    #     param[3],
                    #     s4,
                    #     a4,
                    #     "β₁",
                    #     s5,
                    #     a5,
                    #     "β₂",
                    #     s6,
                    #     a6,
                    #     "β₃]",
                    # )
                    # println(
                    #     file,
                    #     "    [",
                    #     param[3],
                    #     s4b,
                    #     a4b,
                    #     "β₁",
                    #     s5b,
                    #     a5b,
                    #     "β₂",
                    #     s6b,
                    #     a6b,
                    #     "β₃  ",
                    #     param[2],
                    #     "]",
                    # )

                    push!(unique_forms, (coeff_matrix = coeff_matrix, theta = theta_val))
                    push!(get!(theta_buckets, bucket_key, Int[]), idx)
                    exact_lookup[exact_key] = idx

                    q = poly_form(coeff_matrix)
                    println(file, "q(A,θ) = ", q, "\n")
                end
            end
        end

        println(file, "total polarizations checked: ", count)
        println(file, "total RHI's computed: ", total)
        println(file, "polarizations whose RHI represents 1 (skipped): ", represents_one, "\n")
        println(file, "polarization leading to same type: ", pol_count, "\n")

        end_time = time()
        elapsed_time = end_time - start_time
        hours = floor(elapsed_time / 3600)
        minutes = floor((elapsed_time % 3600) / 60)
        seconds = round(elapsed_time % 60)
        println(file, "Total run time: ", hours, " hrs ", minutes, " min ", seconds, " sec")
    catch
        # A run can be hours long; make a partial file self-identifying rather
        # than leaving one that merely looks short.
        println(file, "\n# ABORTED: run failed before completion; output above is partial.")
        rethrow()
    finally
        close(file)
    end
    println("saved data for prime ", pdisp)
    return nothing
end


"""
    parse_big_prime(s)

Parses a CLI argument representing a prime as a BigInt. Accepts plain integer
literals ("23") as well as simple arithmetic expressions in +, -, *, ^, and
parentheses ("2^721*3^176-1"), evaluated with BigInt arithmetic throughout so
large exponents don't overflow.
"""
function parse_big_prime(s::AbstractString)::BigInt
    str = strip(s)

    # Fast path: a plain (possibly signed) integer literal.
    if occursin(r"^[+-]?\d+$", str)
        return parse(BigInt, str)
    end

    # Otherwise, treat it as an arithmetic expression. Only allow digits,
    # whitespace, parentheses, and the operators + - * ^ for safety.
    if !occursin(r"^[\d\s\+\-\*\^\(\)]+$", str)
        error("Invalid characters in prime expression: $s")
    end

    expr = Meta.parse(str)

    # Rewrite the expression tree so every integer literal becomes a BigInt,
    # forcing all arithmetic (including ^) to use arbitrary precision.
    function bigify(e)
        if e isa Integer
            return BigInt(e)
        elseif e isa Expr
            return Expr(e.head, map(bigify, e.args)...)
        else
            return e
        end
    end

    value = eval(bigify(expr))
    return BigInt(value)
end

"""
    read_primes_file(filename) -> Vector{Tuple{Int,BigInt}}

Reads a `primes.txt`-style file: each non-blank, non-comment line holds
a bit-length and a prime expression (parsed with `parse_big_prime`, so
plain integers or arithmetic like `2^721*3^176-1` both work), e.g.

    # bits  p_expression
    50      2^11*3^24-1
    100     2^44*3^35-1

Returns `(bits, p)` pairs in file order. Lines starting with `#` (such
as the header) and blank lines are skipped.
"""
function read_primes_file(filename::String)
    entries = Vector{Tuple{Int,BigInt}}()
    for line in eachline(filename)
        raw = strip(line)
        if isempty(raw) || startswith(raw, "#")
            continue
        end
        parts = split(raw)
        length(parts) < 2 && continue
        bits = parse(Int, parts[1])
        p = parse_big_prime(parts[2])
        push!(entries, (bits, p))
    end
    return entries
end

"""
    main()

Parses CLI arguments `p` and `N` and runs all_rhi2(p, N).

Usage:
    julia rhi.jl <p> <N>

Examples:
    julia rhi.jl 23 50
    julia rhi.jl "2^721*3^176-1" 100

`p` may be a plain integer or a simple arithmetic expression (using +, -, *,
^, parentheses), evaluated with BigInt precision — useful for very large
primes. `N` selects which polarization file to read.

For small primes (24 bits or fewer), this reads "polz_<p>_<N>.txt" and
writes "RHI2_<p>_<N>.txt".
For large primes (25+ bits), this reads "polz_<bits>bit_<digest>_<N>.txt" and writes
"RHI2_<bits>bit_<digest>_<N>.txt", matching the naming convention used by the
polarization generator (digest = low 32 bits of hash(string(p)) in hex).
"""
function main()
    if length(ARGS) < 2
        println("Usage: julia rhi.jl <p> <N>")
        println("  p: prime (plain integer or arithmetic expression, e.g. \"2^721*3^176-1\")")
        println("  N: number (used to locate the polarization file and name the output file)")
        exit(1)
    end

    p = parse_big_prime(ARGS[1])
    N = parse(Int, ARGS[2])

    all_rhi2(p, N)
end

# ------------------------------------------------------------------
# Entry point.
#
# If CLI args are given (e.g. `julia rhi.jl "2^11*3^24-1" 10000`), run
# just that one prime via `main()`, which reads `p` and `N` from `ARGS`.
#
# Otherwise (no args), fall back to the batch driver: runs
# `all_rhi2(p, 100_000)` for every prime listed in `primes.txt`, in file
# order.
if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS)
        main()
    else
        for (bits, p) in read_primes_file("primes.txt")
            println("=== ", bits, "-bit prime ===")
            all_rhi2(p, 100_000)
        end
    end
end