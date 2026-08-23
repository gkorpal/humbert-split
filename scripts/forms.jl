using Oscar

# Hecke internals not re-exported by Oscar (leading-underscore, or
# `isometry` for simultaneous-packet Plesken-Souvignier search):
# Hecke._try_iso_setup_small, Hecke._iso_setup, Hecke.isometry.
const Hecke = Oscar.Hecke

# ------------------------------------
# Integer-matrix / Hecke conversion helpers
# ------------------------------------

_zzmat(A::Matrix{<:Integer}) = matrix(ZZ, A)
_zzpacket(F::AbstractVector{<:AbstractMatrix{<:Integer}}) = ZZMatrix[_zzmat(Matrix(A)) for A in F]
# BigInt, not Int: these hold LLL-reduced Gram-matrix entries whose
# size is bounded by the lattice's discriminant, not by the size of
# the raw polarization coordinates that fed into `rhi` -- but nothing
# in this file *proves* that bound stays under Int64's ~9.2e18, so per
# Oscar's own integer-type guidance (Int only where overflow is
# provably avoided) this must be BigInt, not Int.
_mat_big(M) = Matrix{BigInt}([BigInt(M[i, j]) for i in 1:nrows(M), j in 1:ncols(M)])
_big(A::Matrix{<:Integer}) = BigInt.(A)
_diagmax(A::Matrix{<:Integer}) = maximum(A[i, i] for i in 1:min(size(A, 1), size(A, 2)))
_det_big(A::Matrix{<:Integer}) = BigInt(det(_zzmat(BigInt.(A))))

function _inv_unimodular_bigint(T::Matrix{BigInt})
    Tzz = _zzmat(T)
    d = det(Tzz)
    @assert d == 1 || d == -1 "matrix is not unimodular; det=$d"
    return _mat_big(inv(Tzz))
end

# ----------------------------
# Exact Hecke isometry test
# ----------------------------

"""
    _verify_transport(T, F, G) -> Bool

Check that `T`, as returned by Hecke's `isometry`, conjugates one Gram
matrix to the other, trying each transport convention (row/column,
`G->F`/`F->G`, `T^{-1}`) in turn with early return.
"""
function _verify_transport(T::Matrix{BigInt}, F::AbstractVector{<:AbstractMatrix{<:Integer}}, G::AbstractVector{<:AbstractMatrix{<:Integer}})
    Fb = [_big(Matrix(A)) for A in F]
    Gb = [_big(Matrix(A)) for A in G]
    Tt = transpose(T)
    idx = eachindex(Fb)
    all(k -> T * Gb[k] * Tt == Fb[k], idx) && return true
    all(k -> T * Fb[k] * Tt == Gb[k], idx) && return true
    all(k -> Tt * Gb[k] * T == Fb[k], idx) && return true
    all(k -> Tt * Fb[k] * T == Gb[k], idx) && return true
    if _det_big(T) == 1 || _det_big(T) == -1
        Ti = _inv_unimodular_bigint(T)
        Tit = transpose(Ti)
        all(k -> Ti * Fb[k] * Tit == Gb[k], idx) && return true
    end
    return false
end

_verify_f1(T::Matrix{BigInt}, F::AbstractMatrix{<:Integer}, G::AbstractMatrix{<:Integer}) =
    _verify_transport(T, [F], [G])

"""
    _choose_input_output(F, G; minbound_direction=true) -> (Fin, Gout, direction, input_bound, output_bound)

Hecke's isometry setup (vector sums, Bacher polynomials) is cheaper
when it starts from the lattice with the smaller-normed diagonal, so
with `minbound_direction=true` that one is returned first.
"""
function _choose_input_output(F, G; minbound_direction::Bool=true)
    bF, bG = _diagmax(F[1]), _diagmax(G[1])
    if minbound_direction && bG < bF
        return G, F, :G_input, bG, bF
    else
        return F, G, :F_input, bF, bG
    end
end

function _try_setup(FF, GG; depth::Int=0, bacher_depth::Int=0)
    fl, CF, CG = Hecke._try_iso_setup_small(FF, GG; depth=depth, bacher_depth=bacher_depth)
    fl && return CF, CG, true
    CF, CG = Hecke._iso_setup(FF, GG; depth=depth, bacher_depth=bacher_depth)
    return CF, CG, false
end

"""
    _hecke_isometric_core(Fin, Gout; depth, bacher_depth, verify, verify_fn, error_msg) -> NamedTuple

Run Hecke's small-setup-then-`isometry` pipeline on `Fin`/`Gout` (each a
vector of integer matrices, already ordered by `_choose_input_output`),
then verify the raw transform via `verify_fn`, raising `error_msg` on
failure.
"""
function _hecke_isometric_core(Fin::AbstractVector{<:AbstractMatrix{<:Integer}}, Gout::AbstractVector{<:AbstractMatrix{<:Integer}};
                               depth::Int, bacher_depth::Int, verify::Bool,
                               verify_fn, error_msg::String)
    tsetup = time()
    CF, CG, setup_small = _try_setup(_zzpacket(Fin), _zzpacket(Gout); depth=depth, bacher_depth=bacher_depth)
    setup_time = time() - tsetup
    tiso = time()
    b, rawT = Hecke.isometry(CF, CG)
    iso_time = time() - tiso
    if Bool(b) && verify
        verify_fn(_mat_big(rawT), Fin, Gout) || error(error_msg)
    end
    return (b=Bool(b), setup_small=setup_small, setup_time=setup_time, iso_time=iso_time,
            total_time=setup_time + iso_time)
end

"""
    hecke_f1_isometric(F1, G1; depth=0, bacher_depth=0, verify=true, minbound_direction=true) -> NamedTuple

Exact isometry test between two Gram matrices via Hecke's small-setup
Plesken-Souvignier pipeline and `Hecke.isometry`. Returns a `NamedTuple`
whose `b` field is `true` iff `F1` and `G1` are isometric; `verify`
double-checks any transform Hecke returns against the input matrices.
`F1`/`G1` may hold any integer type (`Int`, `BigInt`, ...) -- entries
are only ever narrowed to `Int` where that is provably safe (loop
indices, dictionary bookkeeping), never for the Gram-matrix values
themselves.

This bypasses Oscar's high-level `is_isometric`/
`is_isometric_with_isometry` wrapper for `ZZLat`, which for the
definite case funnels into the same `_try_iso_setup_small`/
`_iso_setup`/`Hecke.isometry` core called here, but adds overhead this
function avoids:

  1. The wrapper unconditionally re-runs `lll_gram_with_transform` on
     both operands on every call, with no caching. The Gram matrices
     fed in here are already LLL-reduced (they come straight out of
     [`rhi`](@ref)'s `lll_gram`), so re-reducing them is wasted work.
  2. The wrapper always treats its first argument as the enumeration
     "input" side, with no equivalent of `_choose_input_output`'s
     smaller-diagonal heuristic, so it can end up doing the expensive
     vector-sum/Bacher-polynomial setup on the larger-normed side.
  3. The wrapper builds full `ZZLat`/`quadratic_space` wrapper objects
     around each Gram matrix instead of operating on bare `ZZMatrix`
     values.
"""
function hecke_f1_isometric(F1::AbstractMatrix{<:Integer}, G1::AbstractMatrix{<:Integer};
                            depth::Int=0, bacher_depth::Int=0,
                            verify::Bool=true, minbound_direction::Bool=true)
    Fin, Gout, direction, input_bound, output_bound =
        _choose_input_output([F1], [G1]; minbound_direction=minbound_direction)
    core = _hecke_isometric_core(Fin, Gout; depth=depth, bacher_depth=bacher_depth, verify=verify,
                                 verify_fn=(T, F, G) -> _verify_f1(T, F[1], G[1]),
                                 error_msg="F1 isometry verification failed")
    return merge((direction=direction, input_bound=input_bound, output_bound=output_bound), core)
end

# ============================================================
# Cheap isometry-invariant key
# ============================================================
# (LocalGenus_2, LocalGenus_p, ThetaSeriesInitials, min, kissing number)
# is a hashable isometry invariant: forms with different keys are
# certifiably non-isometric, so bucketing candidates on this key and
# only running `hecke_f1_isometric` within a bucket never discards a
# true match.
#
# The determinant of the Gram matrix is also a genuine isometry
# invariant, but is deliberately *not* part of the key: `rhi` asserts
# it equals `16*p^2` for every valid polarization, and `p` is fixed for
# the duration of one `all_rhi(p)` run, so every candidate in a run
# shares it. Bucketing on a value that never varies within a run does
# no discriminating work -- it would only add a constant component to
# every hash and lookup.

const _THETA_FALLBACK_WARNED = Ref(false)

"""
    theta_initials(L, Tmax::Int) -> NTuple{Tmax,Int}

`(r_1,...,r_Tmax)`, `r_k` = number of vectors of `L` of squared length
`k` (first `Tmax` theta-series coefficients), via Hecke's Fincke-Pohst
`short_vectors`. `short_vectors` returns vectors up to sign, so each
is counted twice. Isometric lattices have identical theta series, so
this is a genuine isometry invariant.
"""
function theta_initials(L, Tmax::Int)
    counts = zeros(Int, Tmax)
    sv = try
        short_vectors(L, Tmax)
    catch err
        if !_THETA_FALLBACK_WARNED[]
            _THETA_FALLBACK_WARNED[] = true
            @warn "short_vectors(L, Tmax) failed; falling back to short_vectors(L, 0, Tmax) for the rest of this run." exception=(err, catch_backtrace())
        end
        short_vectors(L, 0, Tmax)
    end
    for entry in sv
        n = Int(entry[2])
        if 1 <= n <= Tmax
            counts[n] += 2
        end
    end
    return Tuple(counts)
end

"""
    f1_invariant_key(F1::AbstractMatrix{<:Integer}, Tmax::Int, p::Integer) -> NamedTuple

Cheap, hashable isometry invariant combining `LocalGenus_{2,p}`,
`theta_initials`, `minimum`, `kissing_number`. Local genus symbols are
stored via `canonical_symbol` (not `string(g)`) so equal symbols hash
equal. `p::Integer` (not `p::Int`) so this can never `MethodError`
against `all_rhi`/`rhi`, which both accept any `Integer` for `p`.
"""
function f1_invariant_key(F1::AbstractMatrix{<:Integer}, Tmax::Int, p::Integer)
    L = integer_lattice(; gram = _zzmat(Matrix(F1)))
    g2 = canonical_symbol(genus(L, 2))
    gp = canonical_symbol(genus(L, p))
    theta = theta_initials(L, Tmax)
    # `theta` already answers both "what is the minimal norm" and "how
    # many vectors attain it" whenever that minimum is <= Tmax: its
    # first nonzero entry sits at index `minimum(L)` and its value is
    # `kissing_number(L)`. Reading them off `theta` saves two further
    # `short_vectors` enumerations per candidate; the explicit calls
    # are only needed for a lattice whose minimum exceeds `Tmax`, where
    # `theta` is all zeros and carries no information at all.
    m_idx = findfirst(!iszero, theta)
    m, kappa = m_idx === nothing ? (Int(minimum(L)), Int(kissing_number(L))) :
                                   (Int(m_idx), theta[m_idx])
    return (genus2 = g2, genusp = gp, theta = theta, minv = m, kappa = kappa)
end

# Accept either "# p = 11" or "p = 11" headers; data lines are
# "[k] u v w x y z".
const _POL_PRIME_RE = r"^\s*#?\s*p\s*=\s*(\d+)"
const _POL_DATA_RE = r"^\s*\[\d+\]\s+(.+)$"

"""
    ReadPolarizationsReport

Diagnostics from [`read_polarizations`](@ref), so callers (notably
[`all_rhi`](@ref)) can explain *why* zero polarizations were read
instead of just reporting a zero count.

- `data_lines_seen`: lines that looked like polarization data (matched
  the `[k] ...` / bare-numbers shape with >= 6 tokens), whether or not
  they parsed successfully.
- `parse_failures`: of those, how many failed to parse as `BigInt`.
- `sample_failures`: up to 3 `(line_number, raw_text, error)` triples
  for the first failures seen, to make the cause obvious at a glance.
- `short_lines`: lines that had a `[k]` marker (or survived the blank/
  comment/header filters) but fewer than 6 whitespace-separated
  tokens, so they couldn't possibly be a polarization.
"""
struct ReadPolarizationsReport
    data_lines_seen::Int
    parse_failures::Int
    sample_failures::Vector{Tuple{Int,String,String}}
    short_lines::Int
end

"""
    read_polarizations(filename) -> (prime, polarizations, report)

Read a polarization file of the form `polarizations_p<prime>.txt`,
where each line holds six `rhi_param` values, and return `(prime,
polarizations, report)`, where `report::ReadPolarizationsReport`
explains any lines that failed to parse.

Values are parsed as `BigInt`. Polarization coordinates come from a
bounded classification search but their *size* is not bounded by the
search depth in any fixed-width sense: coordinates observed so far
range from single digits (small `p`, e.g. `p = 11`) to over a hundred
digits (`p = 227`), so no fixed-width integer type (`Int`/`Int64`,
`Int128`) is safe in general -- only `BigInt` is. Parsing with a
fixed-width type does not error loudly on these files: an overflow
inside a bare `try/catch` silently drops the line, so a whole file of
too-large coordinates parses as "0 polarizations" with no indication
of why.
"""
function read_polarizations(filename::String)
    prime = nothing
    polarizations = Vector{Vector{BigInt}}()

    data_lines_seen = 0
    parse_failures = 0
    short_lines = 0
    sample_failures = Tuple{Int,String,String}[]

    for (line_number, line) in enumerate(eachline(filename))
        raw = strip(line)
        if isempty(raw)
            continue
        end

        m = match(_POL_PRIME_RE, raw)
        if m !== nothing
            # BigInt, not Int: the header is whatever the file says, and
            # parsing it with a fixed-width type would overflow on a file
            # written for a large prime instead of reporting the mismatch.
            prime = parse(BigInt, m.captures[1])
            continue
        end

        if startswith(raw, "#")
            continue
        end

        data = match(_POL_DATA_RE, raw)
        if data === nothing
            # Not a recognized "[k] ..." data line (e.g. a free-form
            # header/comment like "order basis: [...]"); not data at
            # all, so skip it without counting it toward any report
            # field.
            continue
        end
        raw = data.captures[1]

        nums = split(raw)
        if length(nums) >= 6
            data_lines_seen += 1
            try
                values = [parse(BigInt, nums[i]) for i in 1:6]
                push!(polarizations, values)
            catch err
                parse_failures += 1
                if length(sample_failures) < 3
                    push!(sample_failures, (line_number, raw, sprint(showerror, err)))
                end
                continue
            end
        else
            # Had a "[k] ..." marker but not enough tokens after it.
            short_lines += 1
        end
    end

    report = ReadPolarizationsReport(data_lines_seen, parse_failures, sample_failures, short_lines)
    return prime, polarizations, report
end

"""
    rhi(p, param)

Given `Bp = (-1, -p | Q)` and polarization `param = [u0, v0, w0, x0, y0, z0]`,
compute the coefficient matrix of the 5-ary refined Humbert invariant.
Return the `ZZMatrix` coefficient matrix, or `nothing` if `param` does
not yield a positive-definite form.

Every form that survives the checks below has Gram-matrix determinant
exactly `2^4 * p^2`, independent of the polarization coordinates. That
is a structural invariant of the construction rather than a property
some polarizations happen to have, so a violation means the
computation itself is wrong; it is therefore `@assert`ed on every
successful call instead of being treated as a rejected polarization.
(`all_rhi` runs `rhi` inside a `try`, so a violation is reported as a
counted compute failure with the offending `param` rather than
aborting a whole sweep.)
"""
function rhi(p::Integer, param::AbstractVector{<:Integer})
    u0, v0, w0, x0, y0, z0 = param
    p2 = p * p

    gram = zero_matrix(QQ, 6, 6)

    # Fill diagonal
    gram[1, 1] = 2 * v0^2
    gram[2, 2] = 2 * u0^2
    gram[3, 3] = 8 * w0^2 + 8 * w0 * z0 + 2 * z0^2 + 8
    gram[4, 4] = 8 * x0^2 + 8 * x0 * y0 + 2 * y0^2 + 8
    gram[5, 5] = 2 * x0^2 + 2 * x0 * y0 * p + 2 * x0 * y0 + QQ(1, 2) * y0^2 * p2 + y0^2 * p + QQ(1, 2) * y0^2 + 2 * p + 2
    gram[6, 6] = 2 * w0^2 - 2 * w0 * z0 * p + 2 * w0 * z0 + QQ(1, 2) * z0^2 * p2 - z0^2 * p + QQ(1, 2) * z0^2 + 2 * p + 2

    # Fill upper triangular off-diagonals
    gram[1, 2] = 2 * u0 * v0 - 4
    gram[1, 3] = -4 * v0 * w0 - 2 * v0 * z0
    gram[1, 4] = -4 * v0 * x0 - 2 * v0 * y0
    gram[1, 5] = -2 * v0 * x0 - v0 * y0 * p - v0 * y0
    gram[1, 6] = -2 * v0 * w0 + v0 * z0 * p - v0 * z0

    gram[2, 3] = -4 * u0 * w0 - 2 * u0 * z0
    gram[2, 4] = -4 * u0 * x0 - 2 * u0 * y0
    gram[2, 5] = -2 * u0 * x0 - u0 * y0 * p - u0 * y0
    gram[2, 6] = -2 * u0 * w0 + u0 * z0 * p - u0 * z0

    gram[3, 4] = 8 * w0 * x0 + 4 * w0 * y0 + 4 * x0 * z0 + 2 * y0 * z0
    gram[3, 5] = 4 * w0 * x0 + 2 * w0 * y0 * p + 2 * w0 * y0 + 2 * x0 * z0 + y0 * z0 * p + y0 * z0
    gram[3, 6] = 4 * w0^2 - 2 * w0 * z0 * p + 4 * w0 * z0 - z0^2 * p + z0^2 + 4

    gram[4, 5] = 4 * x0^2 + 2 * x0 * y0 * p + 4 * x0 * y0 + y0^2 * p + y0^2 + 4
    gram[4, 6] = 4 * w0 * x0 + 2 * w0 * y0 - 2 * x0 * z0 * p + 2 * x0 * z0 - y0 * z0 * p + y0 * z0

    gram[5, 6] = 2 * w0 * x0 + w0 * y0 * p + w0 * y0 - x0 * z0 * p + x0 * z0 - QQ(1, 2) * y0 * z0 * p2 + QQ(1, 2) * y0 * z0

    # Mirror the upper triangular part to the lower triangular part
    for i = 2:6
        for j = 1:(i-1)
            gram[i, j] = gram[j, i]
        end
    end

    # Check gram is positive semidefinite
    space = quadratic_space(QQ, gram)
    diag_values = diagonal(space)
    if !all(>=(0), diag_values)
        println("Not semi-pd")
        return nothing
    end

    half_gram = map_entries(x -> ZZ(x // 2), gram)
    reduced = lll_gram(half_gram)

    # Check rank, symmetry, and last row
    if rank(reduced) != 5 || !is_symmetric(reduced) || any(!iszero, reduced[6, :])
        println("non sym")
        return nothing
    end

    # Work with top-left 5×5 submatrix
    top_left = @view reduced[1:5, 1:5]

    lat = integer_lattice(; gram = top_left)
    if is_positive_definite(lat)
        # Structural invariant of the construction -- see the docstring.
        det_gram = det(top_left)
        expected_det = 16 * BigInt(p)^2
        @assert BigInt(det_gram) == expected_det "rhi: det(Gram matrix) = $(det_gram), expected 2^4*p^2 = $(expected_det) for p = $p, param = $param"
        return top_left .* 2
    end
    return nothing
end

"""
    poly_form(M)

Given the coefficient matrix `M` of a quadratic form, return its
polynomial form `f = (1/2) * xᵀMx`.
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
    all_rhi(p; Tmax=6, verify=true, dir=dirname(@__FILE__), N=nothing)

For a prime `p ≡ 11 (mod 12)`, read the polarization file for `p`,
compute the refined Humbert invariant for each polarization via
[`rhi`](@ref), and write each isometry class found to a matching
`RHI_*` file. `dir` defaults to the directory holding this script, so
runs work from any working directory.

`N` selects which polarization file is read, matching the two naming
conventions in `data/`:
- `N === nothing` (default): the exhaustive classification files,
  `<dir>/polarizations_p<p>.txt` -> `<dir>/RHI_<p>.txt`.
- `N::Integer`: the random-sample files of `N` polarizations produced
  by `polz_random.jl`, `<dir>/polz_<p>_<N>.txt` ->
  `<dir>/RHI_<p>_<N>.txt`.

Isometry classes are found in two stages:
1. An exact hash lookup on the Gram matrix entries catches identical
   forms.
2. `f1_invariant_key` (local genus at `2` and `p`, theta series
   initials, minimum, kissing number) buckets the remaining candidates;
   `hecke_f1_isometric` is then only run between forms sharing a key.

`Tmax` sets `f1_invariant_key`'s theta-series cutoff (4:10 is the
recommended range). `verify` is forwarded to
`hecke_f1_isometric`, which checks any transform Hecke's `isometry`
returns against the original Gram matrices.

If nothing gets computed, `all_rhi` says which of the following
happened, rather than just reporting a bare zero count:

- The polarization file was missing or unreadable, or its `p = ...`
  header didn't match `p`. Like `all_rhi2` in `rhi.jl`, the input is
  read *before* the output file is opened, so these three cases report
  on stdout and return without writing (and so without truncating) the
  `RHI_*` file. Unlike `all_rhi2` they do not throw, so a sweep over
  many primes carries on past one bad file.
- Its polarization lines failed to parse (with sample offending lines
  and the parse error -- this is what happens if a fixed-width integer
  type is ever reintroduced into `read_polarizations` and a coordinate
  is too large for it), or individual polarizations raised an error
  inside `rhi`/isometry-checking and were skipped. These are reported
  both on stdout and in the `RHI_*` file, which is written as usual.

Limitations:
- This function is single-threaded by design. Its hot paths (`rhi`'s
  `lll_gram`/`is_positive_definite`/`det`, `f1_invariant_key`'s
  `genus`/`short_vectors`, and `hecke_f1_isometric`'s
  `Hecke.isometry`) all call into FLINT via Nemo/Hecke, whose C-level
  state is not documented as safe for concurrent calls from
  independent Julia OS-threads. To parallelize across polarizations,
  use `Distributed.jl` (separate OS processes, so no shared FLINT
  state) rather than `Threads.@threads`.
- The invariant key rules out false negatives but is not a complete
  isometry invariant, so forms sharing a key still need the exact
  `hecke_f1_isometric` test; a run with many candidates sharing a key
  is still slow.
"""
function all_rhi(p::Integer; Tmax::Int=6, verify::Bool=true, dir::AbstractString=dirname(@__FILE__),
                 N::Union{Nothing,Integer}=nothing)
    start_time = time()
    println("working with prime ", p)

    # `nothing` => the exhaustive `polarizations_p<p>.txt` files; an integer
    # => the `polz_<p>_<N>.txt` random samples of N polarizations. The output
    # name carries the same suffix so the two runs never overwrite each other.
    stem = N === nothing ? string(p) : "$(p)_$(N)"
    filename = joinpath(dir, "RHI_$(stem).txt")
    pol_filename = joinpath(dir, N === nothing ? "polarizations_p$(p).txt" : "polz_$(p)_$(N).txt")

    # Read the input *before* opening the output for writing, as `all_rhi2`
    # in rhi.jl does: `open(_, "w")` truncates, so doing it the other way
    # round destroys a previous good `RHI_*` file and leaves an all-but-empty
    # one behind whenever the input turns out to be missing, unreadable, or
    # for a different prime. Unlike `all_rhi2`, the three fatal cases below
    # report and return rather than throwing, so a sweep over many primes
    # carries on past one bad file.
    prime = nothing
    params = Vector{Vector{BigInt}}()
    report = nothing
    read_error = nothing
    try
        prime, params, report = read_polarizations(pol_filename)
    catch err
        read_error = sprint(showerror, err)
    end
    count = length(params)

    fatal = if read_error !== nothing
        "ERROR: could not read '$(pol_filename)': $(read_error)"
    elseif prime === nothing
        "ERROR: '$(pol_filename)' has no 'p = <n>' header; nothing was read."
    elseif prime != p
        "ERROR: '$(pol_filename)' header says p = $(prime), but all_rhi was called with p = $(p); skipping."
    else
        nothing
    end
    if fatal !== nothing
        println(fatal)
        println("       '$(filename)' was not written; nothing was computed for p = ", p, ".")
        return nothing
    end

    open(filename, "w") do file
        println(file, "p = ", p, "\n")

        idx = 0   # Counting unique forms.
        total = 0 # Total RHIs computed.

        unique_forms = Vector{ZZMatrix}()
        unique_bigints = Vector{Matrix{BigInt}}()    # unique_forms[k] as Matrix{BigInt} (see note on _mat_big)
        key_buckets  = Dict{Any,Vector{Int}}()       # f1_invariant_key(...) => indices into unique_forms
        exact_lookup = Dict{NTuple{25,BigInt},Int}() # flattened Gram entries => index into unique_forms
        pol_count = Dict{Int,Int}()

        # Explain up front, both on stdout and in the output file, why
        # there may be zero (or fewer than expected) polarizations to
        # process -- instead of only ever reporting a bare zero count.
        # The file-level failures are already handled above, before this
        # file was opened; what is left is per-line parse trouble.
        if count == 0
            msg = "WARNING: 0 of $(report.data_lines_seen) polarization line(s) in '$(pol_filename)' parsed successfully " *
                  "($(report.parse_failures) failed to parse as BigInt, $(report.short_lines) had too few fields)."
            println(msg)
            println(file, msg)
            for (lineno, raw, err) in report.sample_failures
                example = "  line $(lineno): $(first(raw, 60))$(length(raw) > 60 ? "..." : "") -> $(err)"
                println(example)
                println(file, example)
            end
            println(file)
        elseif report !== nothing && report.parse_failures > 0
            msg = "NOTE: $(report.parse_failures) of $(report.data_lines_seen) polarization line(s) in '$(pol_filename)' " *
                  "failed to parse and were skipped."
            println(msg)
            println(file, msg, "\n")
        end

        compute_failures = 0
        compute_failure_samples = Tuple{Vector{BigInt},String}[]

        for param in params
          try
            gram = rhi(p, param)
            if gram !== nothing
                total += 1

                gram_big = _mat_big(gram)
                exact_key = NTuple{25,BigInt}(vec(gram_big))
                is_unique = true

                existing_k = get(exact_lookup, exact_key, 0)
                if existing_k != 0
                    is_unique = false
                    pol_count[existing_k] = get(pol_count, existing_k, 1) + 1
                end

                key = nothing
                if is_unique
                    key = f1_invariant_key(gram_big, Tmax, p)
                    for k in get(key_buckets, key, Int[])
                        res = hecke_f1_isometric(gram_big, unique_bigints[k]; verify=verify)
                        if res.b
                            is_unique = false
                            pol_count[k] = get(pol_count, k, 1) + 1
                            break
                        end
                    end
                end

                if is_unique
                    idx += 1
                    println(file, "Type ", idx)
                    # Polarization printout suppressed -- only the
                    # forms themselves should be written to file.
                    # s4 = param[4] >= 0 ? "+" : "-"
                    # s5 = param[5] >= 0 ? "+" : "-"
                    # s6 = param[6] >= 0 ? "+" : "-"
                    # a4 = abs(param[4])
                    # a5 = abs(param[5])
                    # a6 = abs(param[6])
                    #
                    # s4b = (-param[4]) >= 0 ? "+" : "-"
                    # s5b = (-param[5]) >= 0 ? "+" : "-"
                    # s6b = (-param[6]) >= 0 ? "+" : "-"
                    # a4b = abs(-param[4])
                    # a5b = abs(-param[5])
                    # a6b = abs(-param[6])
                    #
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

                    push!(unique_forms, gram)
                    push!(unique_bigints, gram_big)
                    push!(get!(key_buckets, key, Int[]), idx)
                    exact_lookup[exact_key] = idx

                    q = poly_form(gram)
                    println(file, "q(A,θ) = ", q, "\n")
                end
            end
          catch err
            compute_failures += 1
            if length(compute_failure_samples) < 3
                push!(compute_failure_samples, (param, sprint(showerror, err)))
            end
            continue
          end
        end

        if compute_failures > 0
            msg = "NOTE: $(compute_failures) of $(count) polarization(s) raised an error during rhi()/isometry " *
                  "computation and were skipped (not counted in totals below)."
            println(msg)
            println(file, msg)
            for (bad_param, err) in compute_failure_samples
                example = "  param $(bad_param) -> $(err)"
                println(example)
                println(file, example)
            end
            println(file)
        end

        println(file, "total polarizations checked: ", count)
        println(file, "total RHI's computed: ", total, "\n")
        println(file, "polarization leading to same type: ", pol_count, "\n")

        elapsed_time = time() - start_time
        hours = floor(elapsed_time / 3600)
        minutes = floor((elapsed_time % 3600) / 60)
        seconds = round(elapsed_time % 60)
        println(file, "Total run time: ", hours, " hrs ", minutes, " min ", seconds, " sec")
    end
    println("saved data for prime ", p)
    return nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# Entry point.
#
# Usage: julia forms.jl [<p>] [--N K] [--tmax T] [--dir D] [--no-verify]
#
#   <p>          prime ≡ 11 (mod 12) below 1620; with no <p>, sweep the
#                whole 743:12:1620 range (the previous default behaviour).
#   --N K        read the K-polarization random sample `polz_<p>_<K>.txt`
#                and write `RHI_<p>_<K>.txt`; without it, read the
#                exhaustive `polarizations_p<p>.txt` and write `RHI_<p>.txt`.
#   --tmax T     theta-series cutoff for the invariant key (default 6).
#   --dir D      input/output directory (default: directory of this script).
#   --no-verify  skip re-checking the transforms Hecke's `isometry` returns.
#
# This block only runs when the file is executed directly. Loading it with
# `include("forms.jl")` -- to drive `all_rhi`/`rhi` interactively, or to
# sweep a different range by hand -- defines the functions and stops there,
# matching how `prim.jl`/`prim_rm.jl` in this directory behave.
# ══════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    let
        function usage()
            println(stderr, """
Usage: julia forms.jl [<p>] [--N K] [--tmax T] [--dir D] [--no-verify]

  <p>          prime below 1620; if omitted, every prime in 743:12:1620
               is processed in turn
  --N K        use the K-polarization sample file polz_<p>_<K>.txt
               (output: RHI_<p>_<K>.txt); without it, the exhaustive
               polarizations_p<p>.txt is read (output: RHI_<p>.txt)
  --tmax T     theta-series cutoff for the invariant key (default 6)
  --dir D      input/output directory (default: script directory)
  --no-verify  skip verifying the transforms Hecke's isometry returns

Examples:
  julia forms.jl 1319                  # reads polarizations_p1319.txt
  julia forms.jl 1319 --N 100000       # reads polz_1319_100000.txt
  julia forms.jl 1319 --dir ./data --tmax 8
  julia forms.jl                       # sweep 743:12:1620
""")
            exit(1)
        end

        local p = nothing   # Union{Nothing,Int}; nothing => sweep the range
        local i = 1
        if !isempty(ARGS) && !startswith(ARGS[1], "--")
            p = tryparse(Int, ARGS[1])
            (p === nothing || p < 2 || p >= 1620 || !is_probable_prime(ZZ(p))) &&
                (println(stderr, "Error: <p> must be a prime less than 1620, got \"$(ARGS[1])\""); usage())
            i = 2
        end

        local dir    = dirname(@__FILE__)
        local Tmax   = 6
        local verify = true
        local N      = nothing  # Union{Nothing,Int}; nothing => polarizations_p<p>.txt

        while i <= length(ARGS)
            flag = ARGS[i]
            if flag == "--tmax"
                i + 1 > length(ARGS) && (println(stderr, "Error: --tmax requires an argument"); usage())
                Tmax = tryparse(Int, ARGS[i+1])
                (Tmax === nothing || Tmax < 1) &&
                    (println(stderr, "Error: --tmax requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--N"
                i + 1 > length(ARGS) && (println(stderr, "Error: --N requires an argument"); usage())
                N = tryparse(Int, ARGS[i+1])
                (N === nothing || N < 1) &&
                    (println(stderr, "Error: --N requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--dir"
                i + 1 > length(ARGS) && (println(stderr, "Error: --dir requires an argument"); usage())
                dir = ARGS[i+1]; i += 2
            elseif flag == "--no-verify"
                verify = false; i += 1
            else
                println(stderr, "Error: unknown argument \"$flag\""); usage()
            end
        end

        isdir(dir) || (println(stderr, "Error: directory not found: \"$dir\""); exit(1))

        if p === nothing
            for q in 743:12:1620
                if is_probable_prime(ZZ(q))
                    all_rhi(q; Tmax = Tmax, verify = verify, dir = dir, N = N)
                end
            end
        else
            all_rhi(p; Tmax = Tmax, verify = verify, dir = dir, N = N)
        end
    end
end