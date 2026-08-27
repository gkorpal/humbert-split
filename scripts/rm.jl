# For each quinary (5×5) positive-definite integral quadratic form
#
#     Q_A(x) = x' A x,   x ∈ ℤ⁵,   A ∈ Mat_{5×5}(ℤ) symmetric, A > 0,
#
# read out of an RHI_{p}.txt file, decide which square-free positive
# integers n ≤ D are *primitively* represented by Q_A, i.e. Q_A(x) = n for
# some x ∈ ℤ⁵ with gcd(x₁,…,x₅) = 1.
#
#
# ─── Algorithm ────────────────────────────────────────────────────────────
#
#   squarefree_list(D)           Sieve [1, D] for square-free integers by
#                                 striking out multiples of p² for every
#                                 p ≤ √D (no primality test needed: striking
#                                 multiples of a composite p² is redundant
#                                 with, and hence harmless alongside, the
#                                 strikes already done at its prime factors).
#   build_form(A)                 Validate A, build the Oscar lattice, and
#                                 LLL-reduce it once via
#                                 `lll_gram_with_transform` — this is the
#                                 one expensive step per form, so it is paid
#                                 only once rather than once per n.
#   primitively_represents(F, n)  Enumerate lattice vectors of norm exactly
#                                 n in the LLL-reduced lattice via Hecke's
#                                 `short_vectors_iterator`, and return the
#                                 first primitive one found (if any),
#                                 back-transformed to the original
#                                 coordinates and verified against Q_A.
#   scan_form(F, ns)               [primitively_represents(F, n) for n in ns],
#                                 collecting the n's that are represented.
#
# ─── Launch ────────────────────────────────────────────────────────────────
#
#   julia rm.jl <p> --D <D> [--dir <dir>] [--max-nodes <n>]
#
#   <p>            a prime less than 1620; the level of the RHI_{p}.txt
#                  file to read (required).
#   --D <D>        upper bound; every square-free 1 ≤ n ≤ D is tested
#                  against each form read from the file (required).
#   --dir <dir>    directory containing RHI_{p}.txt (default: directory
#                  of this script).
#   --max-nodes <n>  safety cap on the number of candidates
#                  `short_vectors_iterator` may examine per (form, n) pair
#                  before giving up (default: 300000). If this is hit, the
#                  result for that n is reported as not-found but flagged
#                  as truncated via a warning, since the search was not
#                  exhaustive.
#
# Example:
#   julia rm.jl 11 --D 100

using Oscar

const ZZ = Oscar.ZZ

"""
    zz_dot(a::ZZMatrix, b::ZZMatrix) -> ZZRingElem

Dot product of two n×1 FLINT `ZZMatrix` column vectors, computed with a
plain element-wise loop over Oscar's own `ZZRingElem` arithmetic (`+`, `*`,
indexing). Used only in the
per-candidate hot loop of `primitively_represents`, where avoiding a fresh
1×1 matrix allocation per candidate (as `transpose(xm) * Ax` would incur)
matters; `qvalue` below uses the built-in matrix-product form instead,
since it isn't called from that loop.
"""
function zz_dot(a::ZZMatrix, b::ZZMatrix)
    n = nrows(a)
    n == nrows(b) || error("dimension mismatch")
    s = ZZ(0)
    @inbounds for i in 1:n
        s += a[i, 1] * b[i, 1]
    end
    return s
end

# ══════════════════════════════════════════════════════════════════════════════
# Square-free sieve
# ══════════════════════════════════════════════════════════════════════════════

"""
    squarefree_list(D::Integer) -> Vector{Int}

Return every square-free integer in `1:D`, in increasing order.

Sieve: start with all of `1:D` marked square-free, then for every
`p = 2, …, ⌊√D⌋`, strike out all multiples of `p²`. `p` need not be prime —
striking multiples of `p²` for composite `p` only re-strikes numbers already
struck at `p`'s prime factors, so the result is unaffected and no primality
test is needed.
"""
function squarefree_list(D::Integer)
    D = Int(D)
    D >= 1 || return Int[]
    sf  = trues(D)
    lim = isqrt(D)
    for p in 2:lim
        p2 = p * p
        for m in p2:p2:D
            sf[m] = false
        end
    end
    return findall(sf)
end

# ══════════════════════════════════════════════════════════════════════════════
# Quadratic-form evaluation / primitivity 
# ══════════════════════════════════════════════════════════════════════════════

"""
    qvalue(A_zz, xm) -> ZZRingElem
    qvalue(A_zz, x)  -> BigInt

Evaluate the quadratic form Q_A(x) = x'Ax via two FLINT matrix products
(`A·x` then `x'·(A·x)`), extracting the resulting 1×1 matrix's sole entry.
Built entirely from Oscar's own `ZZMatrix` arithmetic (`*`, `transpose`,
indexing) — no external linear-algebra dependency. This is a general-purpose
evaluator; the actual per-candidate hot loop in `primitively_represents`
below inlines its own scratch-buffer version of the same computation for
speed, via `zz_dot`.
"""
function qvalue(A_zz::ZZMatrix, xm::ZZMatrix)
    Ax = A_zz * xm
    return (transpose(xm) * Ax)[1, 1]
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

`gcd(x₁, …, xₙ)`, short-circuiting as soon as the running gcd reaches 1.
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

# Fast primitivity check on FLINT ZZRingElem vectors (as returned by
# short_vectors_iterator), avoiding Int -> ZZ boxing.
function _isprimitive_zz(x::AbstractVector{ZZRingElem})
    g = ZZ(0)
    @inbounds for a in x
        g = gcd(g, a)
        isone(g) && return true
    end
    return false
end

# ══════════════════════════════════════════════════════════════════════════════
# Form: validated lattice + one-time LLL reduction
# ══════════════════════════════════════════════════════════════════════════════

"""
    Form

Pre-computed, immutable data for one positive-definite 5×5 integral
quadratic form Q_A(x) = x'Ax, ready for repeated primitive-representation
queries against many targets n.

Fields
──────
  A       Original Gram matrix, `Matrix{BigInt}`.
  A_zz    Same matrix as a FLINT `ZZMatrix`, for the hot loop in
          `primitively_represents` (and for `qvalue`).
  U       Unimodular LLL transform such that `x = U·z` maps reduced
          coordinates `z` back to original coordinates `x`, as `Matrix{BigInt}`.
  U_zz    Same, as a FLINT `ZZMatrix`.
  Lred    The LLL-reduced lattice (in `z`-coordinates), as an Oscar `ZZLat`.
"""
struct Form
    A::Matrix{BigInt}
    A_zz::ZZMatrix
    U::Matrix{BigInt}
    U_zz::ZZMatrix
    Lred::ZZLat
end

"""
    build_form(A; lll_delta=0.99, lll_eta=0.501) -> Form

Validate a 5×5 symmetric, positive-definite integer Gram matrix `A`, build
the corresponding Oscar lattice, and LLL-reduce it once via
`lll_gram_with_transform`. The returned `Form` can then be queried for any
number of targets `n` via `primitively_represents` without repeating the
reduction.
"""
function build_form(Ain::AbstractMatrix{<:Integer};
                    lll_delta::Float64 = 0.99, lll_eta::Float64 = 0.501)
    A = Matrix{BigInt}(Ain)
    size(A) == (5, 5) || error("expected a 5×5 Gram matrix")
    A == A'           || error("Gram matrix must be symmetric")

    A_zz = matrix(ZZ, A)
    L    = integer_lattice(; gram = A_zz)
    is_positive_definite(L) || error("Gram matrix must be positive definite")

    ctx        = LLLContext(lll_delta, lll_eta, :gram)
    Gred_zz, T = lll_gram_with_transform(A_zz, ctx)

    n = nrows(T)
    U = Matrix{BigInt}(undef, n, n)
    @inbounds for i in 1:n, j in 1:n
        U[i, j] = BigInt(T[i, j])
    end

    Lred = integer_lattice(; gram = Gred_zz)
    return Form(A, A_zz, U, T, Lred)
end

# ══════════════════════════════════════════════════════════════════════════════
# Primitive representation test
# ══════════════════════════════════════════════════════════════════════════════

"""
    primitively_represents(F::Form, n; max_nodes=300_000) -> NamedTuple

Decide whether `Q_A` primitively represents the positive integer `n`.

Enumerates lattice vectors `z` of norm exactly `n` in the LLL-reduced
lattice `F.Lred` via `short_vectors_iterator(F.Lred, n, n)`. For each `z`:

  1. Check `gcd(z) = 1` in reduced coordinates (fast `ZZRingElem` path).
     Since `U` is unimodular, `gcd(z) = 1 ⟺ gcd(U·z) = 1`, so this single
     check suffices — no need to re-check after back-transforming.
  2. Back-transform `x = U·z`.
  3. Verify `Q_A(x) = n` (sanity check; should always hold since `Lred` is
     an isometric image of the original lattice under `U`).

Returns a named tuple `(represented, witness, nodes, truncated)`:
  `represented` — `true` iff a primitive representation was found.
  `witness`     — the vector `x` if found, else `nothing`.
  `nodes`       — number of candidates examined.
  `truncated`   — `true` if the search was aborted at `max_nodes` before
                  exhausting all candidates of norm `n` (in which case
                  `represented = false` does *not* prove non-representability).
"""
function primitively_represents(F::Form, n::Integer; max_nodes::Int = 300_000)
    n >= 1 || error("n must be a positive integer")
    target = ZZ(n)
    iter   = short_vectors_iterator(F.Lred, n, n)
    nn     = nrows(F.U_zz)

    zm = zero_matrix(ZZ, nn, 1)   # reusable column vector for U·z
    xm = zero_matrix(ZZ, nn, 1)   # reusable column vector for the hot loop below
    Ax = zero_matrix(ZZ, nn, 1)   # reusable scratch for A·x
    x  = Vector{BigInt}(undef, nn)

    nodes = 0
    for (z_zz, _) in iter
        nodes += 1
        if nodes > max_nodes
            return (represented = false, witness = nothing,
                    nodes = nodes, truncated = true)
        end

        _isprimitive_zz(z_zz) || continue

        @inbounds for i in 1:nn; zm[i, 1] = z_zz[i]; end
        Oscar.mul!(xm, F.U_zz, zm)                 # x = U·z
        @inbounds for i in 1:nn; x[i] = BigInt(xm[i, 1]); end

        Oscar.mul!(Ax, F.A_zz, xm)                 # Ax = A·x
        if zz_dot(xm, Ax) == target
            return (represented = true, witness = copy(x),
                    nodes = nodes, truncated = false)
        end
    end

    return (represented = false, witness = nothing,
            nodes = nodes, truncated = false)
end

"""
    scan_form(F::Form, ns; max_nodes=300_000) -> Vector{Int}

For each `n` in `ns` (an iterable of positive integers, e.g. from
`squarefree_list`), test `primitively_represents(F, n)` and return the
sublist of `ns` that are primitively represented, in the order given.
Emits a `@warn` for any `n` whose search was truncated at `max_nodes`
(the result for that `n` is then a lower bound, not a proof of absence).
"""
function scan_form(F::Form, ns; max_nodes::Int = 300_000)
    represented = Int[]
    for n in ns
        r = primitively_represents(F, n; max_nodes = max_nodes)
        r.truncated &&
            @warn "search for n=$n truncated at max_nodes=$max_nodes; " *
                  "absence from the result is not proven"
        r.represented && push!(represented, Int(n))
    end
    return represented
end

# ══════════════════════════════════════════════════════════════════════════════
# RHI file parsing (as in rm.jl, specialised to p < 1620: the filename
# prefix is always just the plain decimal value of p — no hashed-name
# scheme is needed for such small primes).
# ══════════════════════════════════════════════════════════════════════════════

"""
    parse_gram_matrix(q_str) -> Matrix{BigInt}

Parse a quinary form string of the form
    "c₁₁*x1^2 + c₁₂*x1*x2 + … + c₅₅*x5^2"
into the symmetric 5×5 Gram matrix A such that Q_A(x) = x'Ax.

  Diagonal terms:     coefficient of `xi^2`        → `A[i,i]`.
  Off-diagonal terms: coefficient of `xi*xj` (i≠j) → `A[i,j] = A[j,i] = coeff/2`.
                      (The input uses the full coefficient `2·A[i,j]` for
                      cross terms.)
"""
function parse_gram_matrix(q_str::AbstractString)
    A = zeros(BigInt, 5, 5)
    s = replace(q_str, r"^q\(A,θ\)\s*=\s*" => "")

    for m in eachmatch(r"([+-]?)\s*(\d*)\s*\*?\s*x(\d)\^2", s)
        sign, digits, idxstr = m.captures
        i   = parse(Int, idxstr)
        mag = isempty(digits) ? BigInt(1) : parse(BigInt, digits)
        A[i, i] = (sign == "-") ? -mag : mag
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
    read_rhi_file(p; dir=".") -> Vector{Matrix{BigInt}}

Locate the file matching `RHI_{p}.txt` in `dir` (`p` is the plain decimal
prime — always short enough here, since `p < 1620`) and return one 5×5 Gram
matrix per "Type N" block, exactly as `rm.jl`'s `read_rhi_file` does for
"short" primes.
"""
function read_rhi_file(p::Integer; dir::AbstractString = ".")
    prefix = string(Int(p))
    candidates = filter(readdir(dir; join = true)) do f
        occursin(Regex("RHI_$(prefix).txt\$"), f)
    end
    isempty(candidates)    && error("No RHI_$(prefix).txt found in \"$dir\"")
    length(candidates) > 1 && @warn "Multiple matches; using $(candidates[1])"

    all_lines = readlines(candidates[1])
    stop_idx  = findfirst(l -> occursin(Regex("^p\\s*=\\s*$(prefix)\\s+\\(IKO"), l),
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

    isempty(matrices) && error("No form blocks parsed from $(candidates[1])")
    return matrices
end

# ══════════════════════════════════════════════════════════════════════════════
# Driver: run everything for one prime and write results to reps_p.txt
# ══════════════════════════════════════════════════════════════════════════════

"""
    check_prim(p, D; dir=".", max_nodes=300_000) -> String

For the prime `p`, read the quadratic forms from `RHI_p.txt` in `dir`,
test every square-free `1 ≤ n ≤ D` against each form for primitive
representability, and write the results to `reps_p.txt` in `dir` — one
line at a time, flushed as soon as it's produced, so the file reflects
progress even if the run is interrupted or a later form errors out.
Also emits `@info`/`@warn` progress messages to the console. Returns the
path to the output file.

Can be called directly from the REPL or a loaded script (e.g. to sweep
over many primes), or via this file's command-line entry point.
"""
function check_prim(p::Integer, D::Integer;
                     dir::AbstractString = ".", max_nodes::Int = 300_000)
    @info "Starting" p D dir max_nodes

    outfile = joinpath(dir, "reps_$(p).txt")
    io = open(outfile, "w")

    try
        println(io, "p = $p")
        println(io, "D = $D")
        println(io, "max_nodes = $max_nodes")
        println(io)
        flush(io)

        ns = squarefree_list(D)
        @info "$(length(ns)) square-free integer(s) in [1, $D] to test"
        println(io, "$(length(ns)) square-free integer(s) in [1, $D] to test")
        println(io, ns)
        println(io)
        flush(io)

        forms = read_rhi_file(p; dir = dir)
        @info "Read $(length(forms)) quadratic form(s) from RHI_$(p).txt"

        for (idx, A) in enumerate(forms)
            println(io, "Type $idx:")

            F = build_form(A)
            represented = scan_form(F, ns; max_nodes = max_nodes)

            println(io, "primitively represented square-free n ≤ $D (",
                    length(represented), " of ", length(ns), ")")
            println(io, represented)
            println(io)
            flush(io)
        end
    finally
        close(io)
    end

    @info "Done. Results written to $outfile"
    return outfile
end

# ══════════════════════════════════════════════════════════════════════════════
# Command-line entry point
#
#   julia rm.jl <p> --D <D> [--dir <dir>] [--max-nodes <n>]
#
#   <p>              prime less than 1620, the level of the RHI_{p}.txt
#                    file to read (required).
#   --D <D>          test every square-free 1 ≤ n ≤ D (required).
#   --dir <dir>      input directory (default: directory of this script).
#   --max-nodes <n>  per-(form,n) safety cap on candidates examined by the
#                    exact search (default: 300000).
#
# This block only runs when the file is executed directly (`julia rm.jl
# ...`). Loading the file with `include("rm.jl")` — e.g. to sweep over
# several primes in a loop, or to drive it interactively — skips this
# block entirely and just defines all the functions above directly (no
# module), so `check_prim` (and everything else) is available to call by
# hand:
#
#   include("rm.jl")
# ══════════════════════════════════════════════════════════════════════════════
if abspath(PROGRAM_FILE) == @__FILE__
    let
        function usage()
            println(stderr, """
Usage: julia rm.jl <p> --D <D> [--dir <dir>] [--max-nodes <n>]

  <p>              prime less than 1620; level of the RHI_{p}.txt file
                    to read (required).
  --D <D>          test every square-free integer 1 ≤ n ≤ D against each
                    form in the file (required).
  --dir <dir>      input directory (default: script directory)
  --max-nodes <n>  per-(form,n) cap on candidates examined by the exact
                    search before giving up (default: 300000)

Examples:
  julia rm.jl 11 --D 100
  julia rm.jl 23 --D 200 --dir ./data
  julia rm.jl 7  --D 500 --max-nodes 1000000
""")
            exit(1)
        end

        length(ARGS) < 1 && usage()

        local p = tryparse(Int, ARGS[1])
        (p === nothing || p < 2 || p >= 1620 || !is_probable_prime(ZZ(p))) &&
            (println(stderr, "Error: <p> must be a prime less than 1620, got \"$(ARGS[1])\""); usage())

        local dir       = dirname(@__FILE__)
        local D         = nothing   # Union{Nothing,Int}
        local max_nodes = 300_000

        local i = 2
        while i <= length(ARGS)
            flag = ARGS[i]
            if flag == "--D"
                i + 1 > length(ARGS) && (println(stderr, "Error: --D requires an argument"); usage())
                D = tryparse(Int, ARGS[i+1])
                (D === nothing || D < 1) &&
                    (println(stderr, "Error: --D requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            elseif flag == "--dir"
                i + 1 > length(ARGS) && (println(stderr, "Error: --dir requires an argument"); usage())
                dir = ARGS[i+1]; i += 2
            elseif flag == "--max-nodes"
                i + 1 > length(ARGS) && (println(stderr, "Error: --max-nodes requires an argument"); usage())
                max_nodes = tryparse(Int, ARGS[i+1])
                (max_nodes === nothing || max_nodes < 1) &&
                    (println(stderr, "Error: --max-nodes requires a positive integer, got \"$(ARGS[i+1])\""); usage())
                i += 2
            else
                println(stderr, "Error: unknown argument \"$flag\""); usage()
            end
        end

        D === nothing && (println(stderr, "Error: --D <D> is required"); usage())
        isdir(dir) || (println(stderr, "Error: directory not found: \"$dir\""); exit(1))

        check_prim(p, D; dir = dir, max_nodes = max_nodes)
    end
end

# # uncomment to run
# for p in 599:12:690
#     if is_probable_prime(ZZ(p))
#         check_prim(p,100)
#     end
# end