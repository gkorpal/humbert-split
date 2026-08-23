using Oscar

# Hecke internals not re-exported by Oscar (leading-underscore, or
# `isometry` for simultaneous-packet Plesken-Souvignier search):
# Hecke._try_iso_setup_small, Hecke._iso_setup, Hecke.isometry.
const Hecke = Oscar.Hecke

# Quaternion/order conventions:
# B=(-1,-p)/Q, i^2=-1, j^2=-p, k=ij=-ji. QElt(A0,A1,A2,A3) = (A0+A1*i+A2*j+A3*k)/2.
# Maximal order O = Z<1, i, (i+j)/2, (1+k)/2>. Algebra basis: 1,i,j,k.
# Packet basis on O^2: first 4 vectors on coord 1, next 4 on coord 2.
#
# Map from Algorithm polz(p, Vmax, Umax, Tmax) to this file:
#   theta=[u alpha; conj(alpha) v]        -> Hermitian2x2(u, alpha, v)
#   candidate enum (v, alpha')            -> enumerate_principal_candidates, _candidate_from_uv
#   F_i <- AuxiliaryTraceGram(theta, a_i) -> canonical_packet_forms
#   (F1',P)<-LLLGramWithTransform; F_i'   -> reduce_packet_by_lll
#   key <- (LocalGenus, ThetaSeriesInitials, min, kappa) -> f1_invariant_key
#   ExactIsometric                        -> hecke_f1_isometric
#   ExactSimultaneousIsometric            -> hecke_packet_isometric
#   "for j in Packets with matching key" + isNew -> _find_f1_class_keyed
#     plus the per-candidate loop in _extend_to_Amax!
#   while-loop growing Vmax/Umax until |Reps|=H(p) -> classify_prime_lll_until_target

struct QElt
    A0::Int
    A1::Int
    A2::Int
    A3::Int
end

_qkey(x::QElt) = (x.A0, x.A1, x.A2, x.A3)

Base.show(io::IO, q::QElt) = print(io, "(", q.A0, ", ", q.A1, ", ", q.A2, ", ", q.A3, ")")
Base.:(==)(x::QElt, y::QElt) = _qkey(x) == _qkey(y)
Base.hash(x::QElt, h::UInt) = hash(_qkey(x), h)
Base.:+(x::QElt, y::QElt) = QElt(x.A0 + y.A0, x.A1 + y.A1, x.A2 + y.A2, x.A3 + y.A3)
Base.:-(x::QElt, y::QElt) = QElt(x.A0 - y.A0, x.A1 - y.A1, x.A2 - y.A2, x.A3 - y.A3)
Base.:-(x::QElt) = QElt(-x.A0, -x.A1, -x.A2, -x.A3)

const ZEROQ = QElt(0, 0, 0, 0)

one_quaternion() = QElt(2, 0, 0, 0)
qi() = QElt(0, 2, 0, 0)
qj() = QElt(0, 0, 2, 0)
qk() = QElt(0, 0, 0, 2)

qconj(x::QElt) = QElt(x.A0, -x.A1, -x.A2, -x.A3)
trd(x::QElt) = x.A0

function _halfdiv(n::Int)
    iseven(n) || error("nonintegral quaternion coordinate: numerator=$n")
    return n ÷ 2
end

function nrd(x::QElt, p::Int)
    num = x.A0*x.A0 + x.A1*x.A1 + p*(x.A2*x.A2 + x.A3*x.A3)
    num % 4 == 0 || error("nonintegral reduced norm numerator=$num")
    return num ÷ 4
end

function qmul(x::QElt, y::QElt, p::Int)
    a0,a1,a2,a3 = x.A0, x.A1, x.A2, x.A3
    b0,b1,b2,b3 = y.A0, y.A1, y.A2, y.A3
    return QElt(
        _halfdiv(a0*b0 - a1*b1 - p*a2*b2 - p*a3*b3),
        _halfdiv(a0*b1 + a1*b0 + p*(a2*b3 - a3*b2)),
        _halfdiv(a0*b2 + a2*b0 - a1*b3 + a3*b1),
        _halfdiv(a0*b3 + a3*b0 + a1*b2 - a2*b1),
    )
end

const ORDER_BASIS_O = QElt[
    QElt(2,0,0,0),       # 1
    QElt(0,2,0,0),       # i
    QElt(0,1,1,0),       # (i+j)/2
    QElt(1,0,0,1),       # (1+k)/2
]
order_basis_O() = ORDER_BASIS_O

const ALGEBRA_BASIS_E = QElt[
    QElt(2,0,0,0),       # 1
    QElt(0,2,0,0),       # i
    QElt(0,0,2,0),       # j
    QElt(0,0,0,2),       # k
]
algebra_basis_E() = ALGEBRA_BASIS_E

const O2_BASIS8 = NTuple{2,QElt}[[(b, ZEROQ) for b in ORDER_BASIS_O]; [(ZEROQ, b) for b in ORDER_BASIS_O]]
O2_basis8() = O2_BASIS8

has_integral_order_coordinates(q::QElt) = iseven(q.A0 - q.A3) && iseven(q.A1 - q.A2)

function order_coordinates(q::QElt)
    has_integral_order_coordinates(q) || error("quaternion does not lie in the fixed maximal order: $q")
    z = q.A3
    y = q.A2
    w = (q.A0 - z) ÷ 2
    x = (q.A1 - y) ÷ 2
    return (w, x, y, z)
end

function certify_basis_conventions(p::Int=11)
    B = order_basis_O()
    for a in B, b in B
        has_integral_order_coordinates(qmul(a,b,p)) || return false
    end
    i, h, g = qi(), QElt(0,1,1,0), qj()
    qmul(i,i,p) == QElt(-2,0,0,0) || return false
    qmul(g,g,p) == QElt(-2p,0,0,0) || return false
    return true
end

# ------------------
# Hermitian forms
# ------------------

struct Hermitian2x2
    u::Int
    a::QElt
    v::Int
end

Base.show(io::IO, H::Hermitian2x2) = print(io, "Hermitian2x2(u=", H.u, ", a=", H.a, ", v=", H.v, ")")
Base.:(==)(X::Hermitian2x2, Y::Hermitian2x2) = (X.u == Y.u && X.v == Y.v && X.a == Y.a)
Base.hash(H::Hermitian2x2, h::UInt) = hash((H.u, _qkey(H.a), H.v), h)

function is_principal(H::Hermitian2x2, p::Int)
    return H.u > 0 && H.v > 0 && H.u*H.v - nrd(H.a,p) == 1
end

function normalize_hermitian(H::Hermitian2x2)
    if H.u > H.v
        return Hermitian2x2(H.v, qconj(H.a), H.u)
    elseif H.u == H.v
        ac = qconj(H.a)
        return _qkey(ac) < _qkey(H.a) ? Hermitian2x2(H.u, ac, H.v) : H
    else
        return H
    end
end

# Realises the pseudocode's
#   for v<-1 to Vmax: for alpha' in O/vO with Nrd(alpha')=-1 (mod v):
#     alpha<-ChooseLift(alpha',v); u<-(Nrd(alpha)+1)/v; if 0<u<=Umax: theta<-[u alpha; conj(alpha) v]
# by iterating every principal pair (u,v), u<=v<=Amax
# (enumerate_principal_candidates/_candidate_from_uv), pulling every
# order element alpha of exact reduced norm N=uv-1 from its norm shell
# (_order_elements_exact_norm) instead of looping mod-v residue classes.
# ChooseLift's reduction is _lemma13_residue_key, the stabilizer-orbit
# equivalence of Lemma 13 (Hashimoto-Ibukiyama, "On class numbers of
# positive definite binary quaternion hermitian forms").
# Amax plays both Vmax and Umax: normalize_hermitian forces u<=v, so
# bounding v<=Amax bounds u too.

const _SHELL_CACHE = Dict{Tuple{Int,Int},Vector{QElt}}()
const _PACKET_CACHE = Dict{Tuple{Int,Hermitian2x2},Vector{Matrix{Int}}}()
const _UNITS_CACHE = Dict{Int,Vector{QElt}}()
const _CANDIDATE_CACHE = Dict{Tuple{Int,Int,Int},Vector{Hermitian2x2}}()

function clear_caches!()
    empty!(_SHELL_CACHE)
    empty!(_PACKET_CACHE)
    empty!(_UNITS_CACHE)
    empty!(_CANDIDATE_CACHE)
    return nothing
end

# -----------------------------------------------------------------------
# `short_vectors`'s signature varies across Hecke/Oscar versions;
# _order_elements_exact_norm/theta_initials fall back to a slower path
# if it throws. These Refs make each fallback @warn once per process
# instead of spamming every shell.
const _SHELL_FALLBACK_WARNED = Ref(false)
const _THETA_FALLBACK_WARNED = Ref(false)

"""
    _units_norm1(p) -> Vector{QElt}

Norm-1 units of the fixed maximal order `O` (`U(O)` below). Cached per
`p`, since `_lemma13_residue_key` calls this once per order element,
per shell, per `(u,v)`, per stage.
"""
function _units_norm1(p::Int)
    return get!(_UNITS_CACHE, p) do
        out = QElt[]
        for A0 in -2:2, A1 in -2:2, A2 in -2:2, A3 in -2:2
            if A0*A0 + A1*A1 + p*(A2*A2 + A3*A3) == 4 &&
               iseven(A0 - A3) && iseven(A1 - A2)
                push!(out, QElt(A0,A1,A2,A3))
            end
        end
        sort!(unique(out), by=_qkey)
    end
end

"""
    _lemma13_residue_key(a::QElt, u::Int, v::Int, p::Int) -> NTuple{4,Int}

Canonical key for `(u,v)` fixed, identifying `α` up to the diagonal
stabilizer `diag(u₁,u₂) ∈ U(O)×U(O)`: `α ↦ ū₁·α·u₂` fixes `u,v` and
moves `α` within its orbit; keeping one `α` per orbit is Lemma 13 of
Hashimoto-Ibukiyama, *"On class numbers of positive definite binary
quaternion hermitian forms"*, replacing the pseudocode's `ChooseLift`.

The swap `g=[0 b; c 0]` sends `(u,α,v) ↦ (v,ᾱ,u)`; only a genuine
extra symmetry when `u==v` (reduces to `α ↦ ᾱ`), hence folded in only
under that guard -- for `u!=v` it belongs to a different shell.
"""
function _lemma13_residue_key(a::QElt, u::Int, v::Int, p::Int)
    units = _units_norm1(p)
    keys = NTuple{4,Int}[]
    for u1 in units, u2 in units
        x = qmul(qconj(u1), qmul(a, u2, p), p)
        push!(keys, (mod(x.A0, 2v), mod(x.A1, 2v), mod(x.A2, 2v), mod(x.A3, 2v)))
        if u == v
            xc = qconj(x)
            push!(keys, (mod(xc.A0, 2v), mod(xc.A1, 2v), mod(xc.A2, 2v), mod(xc.A3, 2v)))
        end
    end
    return minimum(keys)
end

"""
    _SHELL_LOG_THRESHOLD

Shells with `N` at or above this get an explicit "starting" / "done"
console line around the (still, in the worst case, expensive) short-
vector search below, so a long stretch with no output reads as "still
working, here's how far along" instead of a hang.
"""
const _SHELL_LOG_THRESHOLD = 2_000

const _ORDER_LATTICE_CACHE = Dict{Int,ZZLat}()

"""
    _order_norm_lattice(p) -> ZZLat

The order's reduced-norm form Nrd, doubled and expressed as an
integral Gram matrix in `order_basis_O()` coordinates `(w,x,y,z)`
(`q = w*1 + x*i + y*(i+j)/2 + z*(1+k)/2`), wrapped as a Hecke `ZZLat`
so `_order_elements_exact_norm` can use Hecke's short-vector
enumerator instead of a box scan.

`Nrd(q)=(A0^2+A1^2+p(A2^2+A3^2))/4` with `A0=2w+z,A1=2x+y,A2=y,A3=z`
works out to `w^2+x^2+wz+xy+q0*(y^2+z^2)`, `q0=(p+1)/4` (integer since
`p==11 mod 12`). Doubled to clear the 1/2 off-diagonal entries; the
resulting form is block-diagonal (after permuting to `(w,z,x,y)`) with
two copies of `[2 1; 1 2*q0]`, positive definite as `short_vectors` needs.

Cached per `p`: the Cholesky setup is the expensive part and is the
same lattice for every `N`.
"""
function _order_norm_lattice(p::Int)
    return get!(_ORDER_LATTICE_CACHE, p) do
        q0 = (p + 1) ÷ 4
        M = Int[2 0 0 1;
                0 2 1 0;
                0 1 2*q0 0;
                1 0 0 2*q0]
        integer_lattice(; gram = _zzmat(M))
    end
end

"""
    _order_elements_exact_norm_bruteforce(p, N) -> Vector{QElt}

Box-scan fallback for `_order_elements_exact_norm` if `short_vectors`
fails. Correct but O(N^2) per shell (`B ~ 2*sqrt(N)`), vs. the fast
path's anisotropic-shape-aware search. Warns once via `_SHELL_FALLBACK_WARNED`.
"""
function _order_elements_exact_norm_bruteforce(p::Int, N::Int)
    vals = QElt[]
    B = Int(ceil(2sqrt(N))) + 2
    for w in -B:B, x in -B:B, y in -B:B, z in -B:B
        q = QElt(2w + z, 2x + y, y, z)
        nrd(q,p) == N && push!(vals, q)
    end
    return sort!(unique(vals), by=_qkey)
end

"""
    _order_elements_exact_norm(p, N) -> Vector{QElt}

All maximal-order elements of exact reduced norm `N` (or `[ZEROQ]` for
`N=0`). Delegates the `N>0` case to Hecke's Fincke-Pohst-style
`short_vectors` on `_order_norm_lattice(p)` (asking for the exact
value `2N`, since that lattice's form is twice `Nrd`), which adapts
its search region via Cholesky, far faster than the O(N^2) bruteforce
box scan once `Amax` reaches the hundreds.

`short_vectors` returns vectors up to sign, so both `v` and `-v` are
added back to return every solution.
"""
function _order_elements_exact_norm(p::Int, N::Int)
    key = (p,N)
    haskey(_SHELL_CACHE, key) && return _SHELL_CACHE[key]

    vals = QElt[]
    if N == 0
        vals = QElt[ZEROQ]
    elseif N > 0
        loggable = N >= _SHELL_LOG_THRESHOLD
        t0 = time()
        if loggable
            println("    [$(_now())] p=$p shell N=$N: enumerating via Hecke short_vectors...")
            flush(stdout)
        end
        L = _order_norm_lattice(p)
        target = 2*N
        sv = try
            short_vectors(L, target, target)
        catch err
            if !_SHELL_FALLBACK_WARNED[]
                _SHELL_FALLBACK_WARNED[] = true
                @warn "short_vectors(L, target, target) failed; falling back to O(N^2) brute-force shell search for the rest of this run. Check installed Oscar/Hecke version." exception=(err, catch_backtrace())
            end
            vals = _order_elements_exact_norm_bruteforce(p, N)
            nothing
        end
        if sv !== nothing
            for (v, n) in sv
                w, x, y, z = Int(v[1]), Int(v[2]), Int(v[3]), Int(v[4])
                push!(vals, QElt(2w + z, 2x + y, y, z))
                push!(vals, QElt(-(2w + z), -(2x + y), -y, -z))
            end
            vals = sort!(unique(vals), by=_qkey)
        end
        if loggable
            println("    [$(_now())] p=$p shell N=$N: done in $(round(time()-t0; digits=1))s, found $(length(vals)) elements")
            flush(stdout)
        end
    end

    _SHELL_CACHE[key] = vals
    return vals
end

"""
    _candidate_from_uv(p, u, v) -> Vector{Hermitian2x2}

All principal candidates `θ = [u α; ᾱ v]` for fixed `u <= v`, i.e.
`Nrd(α) = uv-1`. Pulls every `α` of that norm from its shell
(`_order_elements_exact_norm`), reduces one-per-orbit via
`_lemma13_residue_key`, then keeps the principal ones.

Cached per `(p,u,v)` in `_CANDIDATE_CACHE`: `enumerate_principal_candidates`
re-calls this for every `(u,v)` on every stage, so without the cache
already-known `(u,v)` pairs would redo the `O(|units|^2)` reduction
from scratch each time.
"""
function _candidate_from_uv(p::Int, u::Int, v::Int)
    return get!(_CANDIDATE_CACHE, (p,u,v)) do
        N = u*v - 1
        by_residue = Dict{Any,QElt}()
        for a in _order_elements_exact_norm(p, N)
            k = _lemma13_residue_key(a, u, v, p)
            get!(by_residue, k, a)
        end

        out = Hermitian2x2[]
        for a in values(by_residue)
            H = normalize_hermitian(Hermitian2x2(u,a,v))
            is_principal(H,p) && push!(out, H)
        end
        sort!(unique(out), by = H -> (H.u, H.v, _qkey(H.a)))
    end
end

"""
    enumerate_principal_candidates(p; Amax) -> Vector{Hermitian2x2}

All principal candidates `θ` with `u <= v <= Amax`, i.e. the
pseudocode's `for v <- 1 to Vmax` loop with `Vmax = Umax = Amax` (see
the "Candidate enumeration" comment above `_lemma13_residue_key` for
why one bound suffices) collapsed into pairs `(u,v)` directly rather
than looping over residue classes `alpha' in O/vO`. `Amax` only bounds
candidate generation; the target-count stopping test `|Reps| = H(p)`
from the pseudocode is applied by the caller (`_extend_to_Amax!`), not
here.
"""
function enumerate_principal_candidates(p::Int; Amax::Int)
    Amax >= 1 || return Hermitian2x2[]
    reps = Dict{Tuple{Int,Int,NTuple{4,Int}},Hermitian2x2}()
    for v in 1:Amax, u in 1:v
        for H in _candidate_from_uv(p,u,v)
            reps[(H.u,H.v,_qkey(H.a))] = H
        end
    end
    return sort!(collect(values(reps)), by = H -> (H.u, H.v, _qkey(H.a)))
end

# Trace forms: AuxiliaryTraceGram(theta, a_i) = F_a(x,y) = Trd(h_H(a*x,y))
# for a=1,i,j,k. Definitions from Sarah Chisholm's PhD thesis (U. Calgary),
# Sec. 3.8.1, Lemma 97, citing Greenberg-Voight Lemma 6.2.

function _scale(q::QElt, c::Int)
    return QElt(c*q.A0, c*q.A1, c*q.A2, c*q.A3)
end

_leftmul_pair(a::QElt, x::NTuple{2,QElt}, p::Int) = (qmul(a,x[1],p), qmul(a,x[2],p))

"""
    _hermitian_value(H, x, y, p) -> QElt

`h_H(x,y) = x·H·ȳᵀ` for the Hermitian form with matrix
`H = [u a; ā v]` (`h_H` in the pseudocode's `F_a(x,y) = Trd(h_H(a·x,y))`),
evaluated on `O²`-vectors `x = (x1,x2)`, `y = (y1,y2)`:

    h_H(x,y) = u·x1·ȳ1 + x1·a·ȳ2 + x2·ā·ȳ1 + v·x2·ȳ2.
"""
function _hermitian_value(H::Hermitian2x2, x::NTuple{2,QElt}, y::NTuple{2,QElt}, p::Int)
    x1,x2 = x
    y1,y2 = y
    return _scale(qmul(x1, qconj(y1), p), H.u) +
           qmul(x1, qmul(H.a, qconj(y2), p), p) +
           qmul(x2, qmul(qconj(H.a), qconj(y1), p), p) +
           _scale(qmul(x2, qconj(y2), p), H.v)
end

"""
    canonical_packet_forms(H, p) -> [F1,F2,F3,F4]

Packet `(AuxiliaryTraceGram(θ,1), ..., AuxiliaryTraceGram(θ,k))`: for
each `a` in `algebra_basis_E()`, the 8x8 Gram matrix
`F_a[r,s] = Trd(h_H(a·B[r], B[s]))` over `B = O2_basis8()`. `F1`
feeds `reduce_packet_by_lll`.
"""
function canonical_packet_forms(H::Hermitian2x2, p::Int)
    key = (p,H)
    haskey(_PACKET_CACHE, key) && return _PACKET_CACHE[key]
    is_principal(H,p) || error("canonical_packet_forms expects a principal positive Hermitian form; got $H")

    B = O2_basis8()
    A = algebra_basis_E()
    F = Matrix{Int}[]
    for a in A
        G = zeros(Int, 8, 8)
        for r in 1:8, s in 1:8
            G[r,s] = trd(_hermitian_value(H, _leftmul_pair(a, B[r], p), B[s], p))
        end
        push!(F, G)
    end
    _PACKET_CACHE[key] = F
    return F
end

# ============================================================
# Hashimoto--Ibukiyama class numbers (the number of principal polarizations)
# ============================================================

function check_p_11mod12(p::Int)
    p > 0 || error("p must be positive; got p=$p")
    is_probable_prime(p) || error("p must be prime; got p=$p")
    p % 12 == 11 || error("p must be 11 mod 12; got p=$p")
    return true
end

"""
    The class numbers are taken from 'Supersingular curves of genus two and class numbers' by Tomoyoshi Ibukiyama, Toshiyuki Katsura, and Frans Oort.
 
    classNumbers(p) returns the class numbers h, h(h+1)/2 and H.
    H is the number of all polarizations, and h(h+1)/2 is the number of reducible polarizations.
"""

function classNumbers(p::Integer)
    if p == 2 || p == 3
        return Int(1), Int(1), Int(0)
    elseif p == 5
        return Int(2), Int(1), Int(1)
    end

    # Compute H.
    a = 1 - jacobi_symbol(-1, p)
    b = 1 - jacobi_symbol(-2, p)
    c = 1 - jacobi_symbol(-3, p)
    d = (p % 5 == 4) ? (4//5) : 0//1

    H =
        ((p - 1) * (p + 12) * (p + 23))//2880 +
        (a * (2p + 13))//96 +
        (c * (p + 11))//36 +
        (b//8) +
        ((a * c)//12) +
        d

    # Compute h.
    r12 = p % 12
    offset = if r12 == 1
        0
    elseif r12 == 5 || r12 == 7
        1
    elseif r12 == 11
        2
    else
        0
    end

    h = ((p - r12)//12 + offset)

    return Int(h), Int((h * (h + 1)) ÷ 2), Int(H)
end
targetH(p::Integer) = classNumbers(p)[3]

# ----------------------
# Output
# ----------------------

function _write_polarization_row(io, k::Int, H::Hermitian2x2, p::Int)
    w,x,y,z = order_coordinates(H.a)
    println(io, "[$k]\t$(H.u)\t$(H.v)\t$w\t$x\t$y\t$z\tdet1=$(is_principal(H,p))")
end

"""
    write_polarizations_file(path, reps, p) -> path

Atomically rewrite `path` (tmp+rename) with every row of `reps`. Full
O(n) rewrite; used for initial sync and once per completed Amax-stage.
Per-new-rep appends use `_append_polarization_row!` instead.
"""
function write_polarizations_file(path::AbstractString, reps::Vector{Hermitian2x2}, p::Int)
    tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, "p=$p")
        println(io, "# format: u v w x y z where a = w + x*i + y*(i+j)/2 + z*(1+k)/2")
        for (k,H) in enumerate(reps)
            _write_polarization_row(io, k, H, p)
        end
    end
    mv(tmp, path; force=true)
    return path
end

"""
    _append_polarization_row!(path, k, H, p) -> path

O(1) append of one row to `path`, vs. `write_polarizations_file`'s
O(n) rewrite. Only correct when `path` already holds exactly rows `1..k-1`.
"""
function _append_polarization_row!(path::AbstractString, k::Int, H::Hermitian2x2, p::Int)
    open(path, "a") do io
        _write_polarization_row(io, k, H, p)
    end
    return path
end

# ============================================================
# LLL-reduced packet isometry and dynamic batch classification
# ============================================================
# The functions below keep the original Hermitian2x2 representatives for
# output, but use a unimodular Z^8-basis change of the whole packet
# [F1,F2,F3,F4] before calling Hecke's Plesken--Souvignier routines.
# They also use exact F1-isometry as a safe gate:
#     packet isometry => F1-isometry.

# Progress-line stamp: seconds elapsed since this file was loaded.
const _T0 = time()
_now() = string(round(time() - _T0; digits=1), "s")

# -------------------------------
# Integer matrix / Hecke helpers
# -------------------------------

_zzmat(A::Matrix{<:Integer}) = matrix(ZZ, A)

_zzpacket(F::Vector{Matrix{Int}}) = ZZMatrix[_zzmat(A) for A in F]
_mat_int(M) = Matrix{Int}([Int(M[i,j]) for i in 1:nrows(M), j in 1:ncols(M)])
_mat_big(M) = Matrix{BigInt}([BigInt(M[i,j]) for i in 1:nrows(M), j in 1:ncols(M)])
_big(A::Matrix{Int}) = BigInt.(A)
_diagmax(A::Matrix{Int}) = maximum(A[i,i] for i in 1:min(size(A,1), size(A,2)))
_det_big(A::Matrix{<:Integer}) = BigInt(det(_zzmat(BigInt.(A))))

function _inv_unimodular_bigint(T::Matrix{BigInt})
    Tzz = _zzmat(T)
    d = det(Tzz)
    @assert d == 1 || d == -1 "matrix is not unimodular; det=$d"
    return _mat_big(inv(Tzz))
end

_inv_unimodular_int(P::Matrix{Int}) = Int.(_inv_unimodular_bigint(BigInt.(P)))

# -----------------------------------------
# LLL transport of the whole four-form packet
# -----------------------------------------

function _lll_gram_transform(G::Matrix{Int})
    Gzz = _zzmat(G)
    A, P = lll_gram_with_transform(Gzz)
    return _mat_int(A), _mat_int(P), :lll_gram_with_transform
end

function _detect_transport_convention(G::Matrix{Int}, Gred::Matrix{Int}, P::Matrix{Int})
    if P * G * transpose(P) == Gred
        return :row_PGPt, P
    end
    if transpose(P) * G * P == Gred
        return :col_PtGP, P
    end
    if _det_big(P) == 1 || _det_big(P) == -1
        Pinv = _inv_unimodular_int(P)
        if Pinv * G * transpose(Pinv) == Gred
            return :row_invP_G_invPt, Pinv
        end
        if transpose(Pinv) * G * Pinv == Gred
            return :col_invPt_G_invP, Pinv
        end
    end
    error("Could not detect LLL transport convention")
end

function _transport_packet(F::Vector{Matrix{Int}}, P::Matrix{Int}, conv::Symbol)
    length(F) == 4 || error("expected a 4-form packet")
    if conv == :row_PGPt || conv == :row_invP_G_invPt
        return [P * F[k] * transpose(P) for k in 1:4]
    elseif conv == :col_PtGP || conv == :col_invPt_G_invP
        return [transpose(P) * F[k] * P for k in 1:4]
    else
        error("unknown transport convention $conv")
    end
end

function reduce_packet_by_lll(F::Vector{Matrix{Int}})
    length(F) == 4 || error("expected a 4-form packet")
    old_bound = _diagmax(F[1])
    Gred, Praw, source = _lll_gram_transform(F[1])
    detP = _det_big(Praw)
    @assert detP == 1 || detP == -1 "LLL transform is not unimodular; det=$detP"
    conv, Puse = _detect_transport_convention(F[1], Gred, Praw)
    Fred = _transport_packet(F, Puse, conv)
    @assert Fred[1] == Gred "transported F1 does not match reported reduced Gram"
    return (Fred=Fred, Praw=Praw, Puse=Puse, convention=conv,
            source=source, old_bound=old_bound, new_bound=_diagmax(Fred[1]), detP=detP)
end

# ----------------------------
# Exact Hecke isometry helpers
# ----------------------------

"""
    _verify_transport(T, F, G) -> Bool

Check that `T`, as returned by Hecke's `isometry`, really conjugates
one Gram matrix (or entrywise, one packet of matrices) to the other,
trying each transport convention (row/column, `G->F`/`F->G`, `T^{-1}`)
in turn with early return.
"""
function _verify_transport(T::Matrix{BigInt}, F::Vector{Matrix{Int}}, G::Vector{Matrix{Int}})
    Fb = [_big(A) for A in F]
    Gb = [_big(A) for A in G]
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

_verify_all_forms(T::Matrix{BigInt}, F::Vector{Matrix{Int}}, G::Vector{Matrix{Int}}) =
    _verify_transport(T, F, G)  # 4-form packet

_verify_f1(T::Matrix{BigInt}, F::Matrix{Int}, G::Matrix{Int}) =
    _verify_transport(T, [F], [G])  # single form

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
    if fl
        return CF, CG, true
    end
    CF, CG = Hecke._iso_setup(FF, GG; depth=depth, bacher_depth=bacher_depth)
    return CF, CG, false
end

"""
    _hecke_isometric_core(Fin, Gout; depth, bacher_depth, verify, verify_fn, error_msg) -> NamedTuple

Run Hecke's small-setup-then-`isometry` pipeline on `Fin`/`Gout`
(each a `Vector{Matrix{Int}}`, length 4 for a packet or length 1 for a
lone F1, already ordered by `_choose_input_output`), then verify the
raw transform via `verify_fn` (`_verify_all_forms` or `_verify_f1`,
bound to the right arity), raising `error_msg` if verification fails.
"""
function _hecke_isometric_core(Fin::Vector{Matrix{Int}}, Gout::Vector{Matrix{Int}};
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
            total_time=setup_time+iso_time)
end

function hecke_packet_isometric(F::Vector{Matrix{Int}}, G::Vector{Matrix{Int}};
                                depth::Int=0, bacher_depth::Int=0,
                                verify::Bool=true, minbound_direction::Bool=true)
    Fin, Gout, direction, input_bound, output_bound = _choose_input_output(F, G; minbound_direction=minbound_direction)
    core = _hecke_isometric_core(Fin, Gout; depth=depth, bacher_depth=bacher_depth, verify=verify,
                                 verify_fn=_verify_all_forms, error_msg="packet isometry verification failed")
    return merge((direction=direction, input_bound=input_bound, output_bound=output_bound), core)
end

function hecke_f1_isometric(F1::Matrix{Int}, G1::Matrix{Int};
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
# Cheap isometry-invariant key (LocalGenus, ThetaSeriesInitials, min, kappa)
# ============================================================
# Algorithm polz buckets candidates by
#   (LocalGenus_{2,p}(F1'), ThetaSeriesInitials(F1',Tmax), min(F1'), kappa(F1'))
# and only runs the expensive exact isometry test between candidates
# sharing a key (mismatched key => certifiably not isometric).
# All four invariants map onto existing Oscar/Hecke calls:
#   LocalGenus_{2,p} -> genus(L,q) for q in (2,p) (B_p=(-1,-p|Q) ramifies
#     only at {2,p,infty}), serialized via canonical_symbol for Dict safety
#   ThetaSeriesInitials -> short_vectors (see theta_initials)
#   min -> minimum(L); kappa -> kissing_number(L)

"""
    theta_initials(L, Tmax::Int) -> NTuple{Tmax,Int}

`(r_1,...,r_Tmax)`, `r_k` = number of vectors of `L` of squared length
`k` (first `Tmax` theta-series coefficients), via Hecke's Fincke-Pohst
`short_vectors` (Fincke & Pohst, Math. Comp. 44 (1985), 463-471).
`short_vectors` returns vectors up to sign, so each is counted twice.
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
    f1_invariant_key(F1::Matrix{Int}, Tmax::Int, p::Int) -> NamedTuple

Cheap, hashable isometry invariant combining `LocalGenus_{2,p}`,
`theta_initials`, `minimum`, `kissing_number`. Different keys certify
non-isometric forms, so bucketing on this never discards a true match.

Local genus symbols are stored via `canonical_symbol` (not
`string(g)`) so equal symbols hash equal, matching this file's
hand-written `QElt` hash/`==`.
"""
function f1_invariant_key(F1::Matrix{Int}, Tmax::Int, p::Int)
    L = integer_lattice(; gram = _zzmat(F1))
    g2 = canonical_symbol(genus(L, 2))
    gp = canonical_symbol(genus(L, p))
    theta = theta_initials(L, Tmax)
    m = Int(minimum(L))
    kappa = Int(kissing_number(L))
    return (genus2 = g2, genusp = gp, theta = theta, minv = m, kappa = kappa)
end

# -----------------------------
# Dynamic classification state
# -----------------------------

mutable struct ClassificationState
    p::Int
    targetH::Int
    Tmax::Int
    Amax::Int
    candidates::Vector{Hermitian2x2}
    red_packets::Vector{Any}
    cand_keys::Vector{Any}                  # cheap invariant key per candidate index
    reps::Vector{Hermitian2x2}
    reps_idx::Vector{Int}
    f1_class_reps_idx::Vector{Int}
    f1_bucket_reps_idx::Vector{Vector{Int}}
    f1_keys::Vector{Any}                    # cheap invariant key per F1-class id (stable across Amax growth)
    f1_key_buckets::Dict{Any,Vector{Int}}   # key -> list of F1-class ids sharing that key
    old_bounds::Vector{Int}
    new_bounds::Vector{Int}
    packet_checked::Int
    packet_merged::Int
    f1_checked::Int
    f1_merged::Int
    f1_new_classes::Int
    key_bucket_hits::Int                    # candidates whose key matched >=1 existing class (had to test)
    key_bucket_misses::Int                  # candidates whose key matched no existing class (skipped entirely)
    enum_time::Float64
    packet_time::Float64
    reduce_time::Float64
    merge_time::Float64
    key_time::Float64
    packet_setup_time::Float64
    packet_iso_time::Float64
    f1_setup_time::Float64
    f1_iso_time::Float64
    slow_count::Int
end

function _empty_state(p::Int, H::Int; Tmax::Int=6)
    return ClassificationState(p, H, Tmax, 0, Hermitian2x2[], Any[], Any[],
                               Hermitian2x2[], Int[],
                               Int[], Vector{Int}[], Any[], Dict{Any,Vector{Int}}(),
                               Int[], Int[], 0, 0, 0, 0, 0, 0, 0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

"""
    _outfile_path(outdir, p) -> String

Results-file path for prime `p` in `outdir`.
"""
_outfile_path(outdir::AbstractString, p::Int) = joinpath(outdir, "polarizations_p$(p).txt")

_candidate_key(H::Hermitian2x2) = (H.u, H.v, _qkey(H.a))

function _old_to_new_index_map(old_cand::Vector{Hermitian2x2}, new_cand::Vector{Hermitian2x2})
    pos = Dict{Tuple{Int,Int,NTuple{4,Int}},Int}()
    for (j,H) in pairs(new_cand)
        pos[_candidate_key(H)] = j
    end
    return [pos[_candidate_key(H)] for H in old_cand]
end

function _next_Amax(A::Int; small_until::Int=12, medium_until::Int=40,
                    small_step::Int=1, medium_step::Int=2, large_step::Int=4)
    if A < small_until
        return A + small_step
    elseif A < medium_until
        return A + medium_step
    else
        return A + large_step
    end
end

function _ordered_new_indices(new_indices::Vector{Int}, old_bounds::Vector{Int}, new_bounds::Vector{Int}, order::Symbol)
    if order == :enumerator
        return new_indices
    elseif order == :small_bound
        return sort(new_indices; by=i -> (new_bounds[i], old_bounds[i], i))
    elseif order == :old_bound
        return sort(new_indices; by=i -> (old_bounds[i], new_bounds[i], i))
    else
        error("unknown process_new_order=$order")
    end
end

function _ordered_reps(reps_idx::Vector{Int}, bounds::Vector{Int}, i::Int, order::Symbol)
    if order == :discovery
        return reps_idx
    elseif order == :small_bound
        return sort(reps_idx; by=r -> (bounds[r], abs(bounds[i]-bounds[r]), r))
    else
        error("unknown rep_order=$order")
    end
end

"""
    _find_f1_class_keyed(red_packets, f1_class_reps_idx, f1_keys, f1_key_buckets,
                         i, key_i; kwargs...)

The "does F1' already match an existing class" half of the pseudocode's
    for j in 1..|Packets|: if key==Keys[j]: if ExactIsometric(F1',Packets[j]_1): ...
`f1_key_buckets` (key -> class ids) is an O(1) index for the `key==Keys[j]`
gate instead of a linear scan; mismatched key certifies non-isometric.

F1-isometric representatives are pre-grouped into classes (one rep per
class in `f1_class_reps_idx`), so testing `i` against one rep per class
is complete since F1-isometry is an equivalence relation. Once a class
`cid` matches, the caller still runs `ExactSimultaneousIsometric`
against every full-packet rep in `f1_bucket_reps_idx[cid]`, since
distinct reps can share F1' but differ in F2',F3',F4'.

Returns `(cid, isnewF1, stats)`: `isnewF1=false` with the matching
`cid` if found; `isnewF1=true` with `cid` one past the last class id
otherwise (new F1-class).
"""
function _find_f1_class_keyed(red_packets, f1_class_reps_idx::Vector{Int},
                              f1_keys::Vector{Any}, f1_key_buckets::Dict{Any,Vector{Int}},
                              i::Int, key_i;
                              depth::Int=0, bacher_depth::Int=0, verify::Bool=true)
    checked = 0
    merged = 0
    setup = 0.0
    iso = 0.0
    bucket = get(f1_key_buckets, key_i, Int[])
    for cid in bucket
        rep_i = f1_class_reps_idx[cid]
        checked += 1
        r = hecke_f1_isometric(red_packets[i][1], red_packets[rep_i][1];
                               depth=depth, bacher_depth=bacher_depth, verify=verify)
        setup += r.setup_time
        iso += r.iso_time
        if r.b
            merged += 1
            return cid, false, (checked=checked, merged=merged, setup=setup, iso=iso)
        end
    end
    return length(f1_class_reps_idx) + 1, true, (checked=checked, merged=merged, setup=setup, iso=iso)
end

function _extend_to_Amax!(st::ClassificationState, Amax_new::Int;
                          rep_order::Symbol=:small_bound,
                          process_new_order::Symbol=:small_bound,
                          stop_when_target::Bool=true,
                          depth::Int=0, bacher_depth::Int=0, verify::Bool=true,
                          print_every::Int=100, slow_threshold::Float64=2.0,
                          outfile::Union{Nothing,AbstractString}=nothing)
    p = st.p

    # Baselines for printing this call's incremental progress only.
    base_packet_checked = st.packet_checked
    base_key_hits = st.key_bucket_hits
    base_key_misses = st.key_bucket_misses

    # Enumerate candidates for Amax_new, reduce the unseen ones, and remap
    # the carryover from the previous Amax onto the new indexing.
    println("\n[$(_now())] p=$p extend Amax $(st.Amax) -> $Amax_new")
    st.enum_time += @elapsed (new_cand = enumerate_principal_candidates(p; Amax=Amax_new))
    println("[$(_now())] p=$p candidates=$(length(new_cand))")
    flush(stdout)

    old_to_new = isempty(st.candidates) ? Int[] : _old_to_new_index_map(st.candidates, new_cand)
    old_set = Set(old_to_new)
    red = Vector{Any}(undef, length(new_cand))
    ckeys = Vector{Any}(undef, length(new_cand))   # cheap invariant key per candidate, cached like red
    old_bounds = Vector{Int}(undef, length(new_cand))
    new_bounds = Vector{Int}(undef, length(new_cand))

    for old_i in eachindex(st.candidates)
        new_i = old_to_new[old_i]
        red[new_i] = st.red_packets[old_i]
        old_bounds[new_i] = st.old_bounds[old_i]
        new_bounds[new_i] = st.new_bounds[old_i]
        if isassigned(st.cand_keys, old_i)
            ckeys[new_i] = st.cand_keys[old_i]
        end
    end

    # Raw packets are cheap pure arithmetic (no Hecke/Oscar calls),
    # so kept local to this stage rather than in ClassificationState.
    built_raw = 0
    raw = Vector{Any}(undef, length(new_cand))
    st.packet_time += @elapsed for i in eachindex(new_cand)
        if !isassigned(red, i)
            raw[i] = canonical_packet_forms(new_cand[i], p)
            built_raw += 1
        end
    end

    built_red = 0
    st.reduce_time += @elapsed for i in eachindex(new_cand)
        if !isassigned(red, i)
            R = reduce_packet_by_lll(raw[i])
            red[i] = R.Fred
            old_bounds[i] = R.old_bound
            new_bounds[i] = R.new_bound
            built_red += 1
            if built_red == 1 || built_red % print_every == 0
                println("[$(_now())] p=$p reduced new packet $built_red old_bound=$(R.old_bound) new_bound=$(R.new_bound)")
                flush(stdout)
            end
        end
    end

    # Only the index-valued fields need remapping; `reps`, `f1_keys` and
    # `f1_key_buckets` hold values/class ids that survive re-indexing.
    reps_idx = isempty(st.reps_idx) ? Int[] : [old_to_new[i] for i in st.reps_idx]
    f1_class_reps_idx = isempty(st.f1_class_reps_idx) ? Int[] : [old_to_new[i] for i in st.f1_class_reps_idx]
    f1_bucket_reps_idx = Vector{Int}[]
    for bucket in st.f1_bucket_reps_idx
        push!(f1_bucket_reps_idx, [old_to_new[i] for i in bucket])
    end

    new_indices = [i for i in eachindex(new_cand) if !(i in old_set)]
    new_indices = _ordered_new_indices(new_indices, old_bounds, new_bounds, process_new_order)
    println("[$(_now())] p=$p reused=$(length(st.candidates)) built_raw=$built_raw built_red=$built_red process_new=$(length(new_indices))")
    flush(stdout)

    st.candidates = new_cand
    st.red_packets = red
    st.cand_keys = ckeys
    st.old_bounds = old_bounds
    st.new_bounds = new_bounds
    st.reps_idx = reps_idx
    st.f1_class_reps_idx = f1_class_reps_idx
    st.f1_bucket_reps_idx = f1_bucket_reps_idx

    tmerge = time()

    for (count, i) in pairs(new_indices)
        if stop_when_target && length(st.reps) >= st.targetH
            println("[$(_now())] p=$p target reached inside Amax=$Amax_new; stopping stage early")
            break
        end

        if !isassigned(ckeys, i)
            st.key_time += @elapsed (ckeys[i] = f1_invariant_key(red[i][1], st.Tmax, p))
        end
        key_i = ckeys[i]
        bucket_before = get(st.f1_key_buckets, key_i, Int[])
        isempty(bucket_before) ? (st.key_bucket_misses += 1) : (st.key_bucket_hits += 1)

        cid, isnewF1, fs = _find_f1_class_keyed(red, st.f1_class_reps_idx, st.f1_keys, st.f1_key_buckets, i, key_i;
                                                depth=depth, bacher_depth=bacher_depth, verify=verify)
        st.f1_checked += fs.checked; st.f1_merged += fs.merged
        st.f1_setup_time += fs.setup; st.f1_iso_time += fs.iso
        keep = true

        if isnewF1
            push!(st.f1_class_reps_idx, i)
            push!(st.f1_bucket_reps_idx, Int[])
            push!(st.f1_keys, key_i)
            new_cid = length(st.f1_class_reps_idx)
            push!(get!(st.f1_key_buckets, key_i, Int[]), new_cid)
            st.f1_new_classes += 1
        else
            for rep_i in _ordered_reps(st.f1_bucket_reps_idx[cid], st.new_bounds, i, rep_order)
                st.packet_checked += 1
                r = hecke_packet_isometric(red[i], red[rep_i]; depth=depth, bacher_depth=bacher_depth, verify=verify)
                st.packet_setup_time += r.setup_time; st.packet_iso_time += r.iso_time
                if r.b
                    st.packet_merged += 1
                    keep = false
                    break
                elseif r.total_time >= slow_threshold
                    st.slow_count += 1
                    println("  SLOW p=$p i=$i rep=$rep_i total=$(round(r.total_time; digits=3))s")
                    flush(stdout)
                end
            end
        end

        new_rep_found = false
        if keep
            push!(st.reps, new_cand[i])
            push!(st.reps_idx, i)
            push!(st.f1_bucket_reps_idx[cid], i)
            new_rep_found = true
            println("  NEW p=$p rep #$(length(st.reps)) at i=$i Amax=$Amax_new F1class=$cid newF1=$isnewF1 H=$(new_cand[i])")
            flush(stdout)
        end

        # Append each new rep as it is found, so the results file tracks
        # progress during a long unattended run.
        if new_rep_found && outfile !== nothing
            _append_polarization_row!(outfile, length(st.reps), st.reps[end], p)
        end

        if new_rep_found && stop_when_target && length(st.reps) >= st.targetH
            println("[$(_now())] p=$p target reached by new rep at Amax=$Amax_new")
            break
        end

        if count == 1 || count % print_every == 0 || count == length(new_indices)
            println("[$(_now())] p=$p Amax=$Amax_new processed_new=$count/$(length(new_indices)) " *
                    "reps=$(length(st.reps))/$(st.targetH) F1classes=$(length(st.f1_class_reps_idx)) " *
                    "packet_checked_since_call=$(st.packet_checked - base_packet_checked) elapsed=$(round(time()-tmerge; digits=1))s")
            flush(stdout)
        end
    end

    st.merge_time += time() - tmerge

    # Stage complete: commit Amax and rewrite the results file.
    st.Amax = Amax_new
    outfile !== nothing && write_polarizations_file(outfile, st.reps, p)

    println("[$(_now())] p=$p DONE Amax=$Amax_new reps=$(length(st.reps))/$(st.targetH) " *
            "F1classes=$(length(st.f1_class_reps_idx)) " *
            "packet_checked_since_call=$(st.packet_checked - base_packet_checked) " *
            "key_hits_since_call=$(st.key_bucket_hits - base_key_hits) " *
            "key_misses_since_call=$(st.key_bucket_misses - base_key_misses) (misses skip F1-isometry entirely)")
    flush(stdout)
    return st
end

function _state_stats(st::ClassificationState, total_time::Float64; Astart, process_new_order, rep_order, depth, bacher_depth)
    sb_old = sort(st.old_bounds); sb_new = sort(st.new_bounds)
    return Dict{Symbol,Any}(
        :Amax => st.Amax, :Astart => Astart, :Astep => "dynamic", :Tmax => st.Tmax,
        :depth => depth, :bacher_depth => bacher_depth,
        :process_new_order => process_new_order, :rep_order => rep_order,
        :targetH => st.targetH, :classes => length(st.reps), :candidates => length(st.candidates),
        :total_time => total_time, :enum_time => st.enum_time, :packet_time => st.packet_time,
        :reduce_time => st.reduce_time, :merge_time => st.merge_time, :key_time => st.key_time,
        :packet_checked => st.packet_checked, :packet_merged => st.packet_merged,
        :f1_checked => st.f1_checked, :f1_merged => st.f1_merged,
        :f1_classes => length(st.f1_class_reps_idx), :f1_new_classes => st.f1_new_classes,
        :key_bucket_hits => st.key_bucket_hits, :key_bucket_misses => st.key_bucket_misses,
        :packet_setup_time => st.packet_setup_time, :packet_iso_time => st.packet_iso_time,
        :f1_setup_time => st.f1_setup_time, :f1_iso_time => st.f1_iso_time,
        :slow_count => st.slow_count,
        :old_bound_min => isempty(sb_old) ? 0 : first(sb_old),
        :old_bound_median => isempty(sb_old) ? 0 : sb_old[cld(length(sb_old),2)],
        :old_bound_max => isempty(sb_old) ? 0 : last(sb_old),
        :new_bound_min => isempty(sb_new) ? 0 : first(sb_new),
        :new_bound_median => isempty(sb_new) ? 0 : sb_new[cld(length(sb_new),2)],
        :new_bound_max => isempty(sb_new) ? 0 : last(sb_new),
    )
end

function classify_prime_lll_until_target(p::Int;
                                         Astart::Int=1, Amax_cap::Union{Nothing,Int}=nothing,
                                         Tmax::Int=6,
                                         small_until::Int=12, medium_until::Int=40,
                                         small_step::Int=1, medium_step::Int=2, large_step::Int=4,
                                         rep_order::Symbol=:small_bound,
                                         process_new_order::Symbol=:small_bound,
                                         stop_when_target::Bool=true,
                                         depth::Int=0, bacher_depth::Int=0,
                                         verify::Bool=true, print_every::Int=100,
                                         slow_threshold::Float64=2.0,
                                         max_steps::Union{Nothing,Int}=nothing,
                                         max_wall_seconds::Union{Nothing,Float64}=nothing,
                                         outfile::Union{Nothing,AbstractString}=nothing)
    4 <= Tmax <= 10 || @warn "Tmax=$Tmax is outside the recommended 4:10 range for ThetaSeriesInitials cutoff"
    check_p_11mod12(p)

    st = _empty_state(p, targetH(p); Tmax=Tmax)

    # Lay down the results-file header that `_append_polarization_row!` appends to.
    outfile !== nothing && write_polarizations_file(outfile, st.reps, p)

    t0 = time()
    A = Astart
    steps = 0
    while length(st.reps) < st.targetH
        if Amax_cap !== nothing && A > Amax_cap
            println("[$(_now())] p=$p stopping because next Amax=$A exceeds Amax_cap=$Amax_cap")
            break
        end
        if max_steps !== nothing && steps >= max_steps
            println("[$(_now())] p=$p stopping because max_steps=$max_steps was reached")
            break
        end
        if max_wall_seconds !== nothing && time() - t0 >= max_wall_seconds
            println("[$(_now())] p=$p stopping because max_wall_seconds=$max_wall_seconds was reached")
            break
        end
        steps += 1
        _extend_to_Amax!(st, A;
                         rep_order=rep_order, process_new_order=process_new_order,
                         stop_when_target=stop_when_target, depth=depth,
                         bacher_depth=bacher_depth, verify=verify,
                         print_every=print_every, slow_threshold=slow_threshold,
                         outfile=outfile)

        length(st.reps) >= st.targetH && break
        A = _next_Amax(A; small_until=small_until, medium_until=medium_until,
                       small_step=small_step, medium_step=medium_step, large_step=large_step)
    end
    return st, _state_stats(st, time() - t0; Astart=Astart, process_new_order=process_new_order,
                            rep_order=rep_order, depth=depth, bacher_depth=bacher_depth)
end

"""
    _completed_status(path) -> Union{Nothing,String}

`status=...` from `path`'s stats footer if present, else `nothing`.
Only `_write_stats_footer` appends a footer, once per finished run, so
a present footer means that run reached a terminal state.
"""
function _completed_status(path::AbstractString)
    isfile(path) || return nothing
    for line in reverse(readlines(path))
        m = match(r"^# status=(\S+)", line)
        m !== nothing && return m.captures[1]
    end
    return nothing
end

"""
    _write_stats_footer(path, st, stats, status)

Append summary stats as `# key=value` lines to `path`. Kept in the
per-prime results file (not a shared summary file) so concurrent
SLURM tasks never contend over one file.
"""
function _write_stats_footer(path::AbstractString, st::ClassificationState, stats, status::String)
    open(path, "a") do io
        println(io, "# status=$status")
        println(io, "# targetH=$(st.targetH) classes=$(length(st.reps)) Amax=$(st.Amax) Tmax=$(stats[:Tmax]) candidates=$(stats[:candidates])")
        println(io, "# checks packet_checked=$(st.packet_checked) packet_merged=$(st.packet_merged) f1_checked=$(st.f1_checked) f1_classes=$(length(st.f1_class_reps_idx))")
        println(io, "# key_buckets hits=$(stats[:key_bucket_hits]) misses=$(stats[:key_bucket_misses]) slow_count=$(stats[:slow_count])")
        println(io, "# bounds old_bound_max=$(stats[:old_bound_max]) new_bound_max=$(stats[:new_bound_max])")
        println(io, "# order process_new_order=$(stats[:process_new_order]) rep_order=$(stats[:rep_order])")
        println(io, "# timing_s total=$(round(stats[:total_time]; digits=3)) enum=$(round(stats[:enum_time]; digits=3)) packet=$(round(stats[:packet_time]; digits=3)) reduce=$(round(stats[:reduce_time]; digits=3)) merge=$(round(stats[:merge_time]; digits=3)) key=$(round(stats[:key_time]; digits=3))")
        println(io, "# timing_s packet_setup=$(round(stats[:packet_setup_time]; digits=3)) packet_iso=$(round(stats[:packet_iso_time]; digits=3)) f1_setup=$(round(stats[:f1_setup_time]; digits=3)) f1_iso=$(round(stats[:f1_iso_time]; digits=3))")
    end
    return path
end

"""
    run_lll_until_target(p; outdir=".", kwargs...)

Classify `p`, writing one file under `outdir` (see `_outfile_path`):
`polarizations_p<p>.txt`, holding the representatives plus a stats
footer. No shared batch-level file, so concurrent SLURM-array jobs
never collide.

A prime whose results file already carries a terminal `# status=`
footer is reported and skipped; delete or move that file to
reclassify it.

Calls `clear_caches!()` before returning so memory doesn't linger past
this prime.
"""
function run_lll_until_target(p::Int; outdir::AbstractString=".",
                              Astart::Int=1, Amax_cap::Union{Nothing,Int}=nothing,
                              Tmax::Int=6,
                              small_until::Int=12, medium_until::Int=40,
                              small_step::Int=1, medium_step::Int=2, large_step::Int=4,
                              rep_order::Symbol=:small_bound,
                              process_new_order::Symbol=:small_bound,
                              stop_when_target::Bool=true,
                              depth::Int=0, bacher_depth::Int=0,
                              verify::Bool=true, print_every::Int=100,
                              slow_threshold::Float64=2.0,
                              max_steps::Union{Nothing,Int}=nothing,
                              max_wall_seconds::Union{Nothing,Float64}=nothing)
    isdir(outdir) || mkpath(outdir)
    outfile = _outfile_path(outdir, p)

    # A prime already finished (terminal status footer, see
    # _completed_status) needs no re-classifying; just report it.
    prior_status = _completed_status(outfile)
    if prior_status !== nothing
        println("\n============================================================")
        println("LLL/F1-gate p=$p targetH=$(targetH(p)) -- already $prior_status, skipping ($outfile)")
        println("============================================================")
        clear_caches!()
        return (state=nothing, stats=nothing, outfile=outfile, status=prior_status)
    end

    println("\n============================================================")
    println("LLL/F1-gate p=$p targetH=$(targetH(p))")
    println("============================================================")
    st, stats = classify_prime_lll_until_target(p; Astart=Astart, Amax_cap=Amax_cap, Tmax=Tmax,
        small_until=small_until, medium_until=medium_until, small_step=small_step,
        medium_step=medium_step, large_step=large_step, rep_order=rep_order,
        process_new_order=process_new_order, stop_when_target=stop_when_target,
        depth=depth, bacher_depth=bacher_depth, verify=verify, print_every=print_every,
        slow_threshold=slow_threshold, max_steps=max_steps, max_wall_seconds=max_wall_seconds,
        outfile=outfile)
    status = length(st.reps) == st.targetH ? "TARGET_PASSED" : (length(st.reps) > st.targetH ? "OVER_TARGET" : "INCOMPLETE")
    _write_stats_footer(outfile, st, stats, status)
    println("$status p=$p classes=$(length(st.reps))/$(st.targetH), wrote $outfile")

    clear_caches!()
    return (state=st, stats=stats, outfile=outfile, status=status)
end

# Script entry point: `julia polz_all.jl <prime> [outdir]`.
# outdir defaults to this script's directory. A prime whose results file
# is already complete is skipped; delete it to reclassify. Progress goes
# to stdout/stderr, which SLURM already captures per-job.
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia polz_all.jl <prime> [outdir]")
        exit(1)
    end

    p = try
        parse(Int, ARGS[1])
    catch e
        println("Failed to parse prime from ARGS[1]=", ARGS[1])
        rethrow(e)
    end
    try
        check_p_11mod12(p)
    catch e
        println("Invalid prime p=$p: ", sprint(showerror, e))
        exit(1)
    end

    outdir = length(ARGS) >= 2 ? ARGS[2] : (@__DIR__)
    mkpath(outdir)

    println("Running polz for p=", p, " outdir=", outdir)
    run_lll_until_target(p; outdir=outdir)
end