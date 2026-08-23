using Oscar

# Legendre symbol (a/p) for odd prime p (Kronecker = Legendre for prime p)
legendre(a::Integer, p::Integer)::Int = Int(kronecker_symbol(a, p))
ls_minus(d::Integer, p::Integer)::Int = legendre(-d, p)   # (-d/p)

const AUT_ROWS = [
    (0, "C2",        :C1),
    (1, "C2 × C2",   :C2),
    (2, "D6",        :S3),
    (3, "D4",        :V4),
    (4, "C3 × D4",   :D12),
    (5, "GL2(F3)",   :S4),
    (6, "C10",       :C5),
]

function auto_count_IKO(p::Integer, gamma::Symbol)::Int
    p ≥ 7 || error("Use p ≥ 7 (IKO Theorem 3.3 (I)).")

    q   = QQ(p)
    lm1 = ls_minus(1, p)
    lm2 = ls_minus(2, p)
    lm3 = ls_minus(3, p)

    N = QQ(0)

    if gamma == :C1
        N = (q - 1) * (q^2 - 35q + 346) / 2880
        N -= (QQ(1) - lm1) / 32
        N -= (QQ(1) - lm2) / 8
        N -= (QQ(1) - lm3) / 9
        if mod(p, 5) == 4
            N -= QQ(1) / 5
        end

    elseif gamma == :C2
        N = (q - 1) * (q - 17) / 48
        N += (QQ(1) - lm1) / 8
        N += (QQ(1) - lm2) / 2
        N += (QQ(1) - lm3) / 2

    elseif gamma == :S3
        N = (q - 1) / 6
        N -= (QQ(1) - lm2) / 2
        N -= (QQ(1) - lm3) / 3

    elseif gamma == :V4
        N = (q - 1) / 8
        N -= (QQ(1) - lm1) / 8
        N -= (QQ(1) - lm2) / 4
        N -= (QQ(1) - lm3) / 2

    elseif gamma == :D12
        N = (QQ(1) - lm3) / 2

    elseif gamma == :S4
        N = (QQ(1) - lm2) / 2

    elseif gamma == :C5
        N = mod(p, 5) == 4 ? QQ(1) : QQ(0)

    else
        error("Unknown gamma. Use :C1 :C2 :S3 :V4 :D12 :S4 :C5.")
    end

    denominator(N) == 1 || error("Non-integral result: $N")
    return Int(numerator(N))
end

function aut_counts_IKO(p::Integer)::Dict{String,Int}
    d = Dict{String,Int}()
    for (_, autname, gamma) in AUT_ROWS
        d[autname] = auto_count_IKO(p, gamma)
    end
    return d
end


"""
    fileReader(filename)

Reads polarization files of the format "polarizations_p<prime>.txt" where each line contains rhi_param values.

"""
function fileReader(filename::String)
    prime = nothing
    polarizations = Vector{Vector{BigInt}}()

    # Accept either "# p = 11" or "p = 11" headers.
    prime_re = r"^\s*#?\s*p\s*=\s*(\d+)"
    data_re = r"^\s*\[\d+\]\s+(.+)$"

    for line in eachline(filename)
        raw = strip(line)
        if isempty(raw)
            continue
        end

        m = match(prime_re, raw)
        if m !== nothing
            prime = parse(BigInt, m.captures[1])
            continue
        end

        if startswith(raw, "#")
            continue
        end

        data = match(data_re, raw)
        if data !== nothing
            raw = data.captures[1]
        end

        nums = split(raw)
        if length(nums) >= 6
            try
                values = [parse(BigInt, nums[i]) for i in 1:6]
                push!(polarizations, values)
            catch
                continue
            end
        end
    end

    return prime, polarizations
end

function classNumbers(p::Int)
    # Handle small p quickly H - (h(h+1))/2
    if p == 2 || p == 3
        return [1, 1, 0]
    elseif p == 5
        return [2, 1, 1]
    end

    # Compute H
    # Precompute Jacobi values and remainders
    a = (1 - jacobi_symbol(-1, p))
    b = (1 - jacobi_symbol(-2, p))
    c = (1 - jacobi_symbol(-3, p))
    r5 = p % 5
    d = (r5 == 4) ? (4//5) : 0
    H =
        ((p - 1) * (p + 12) * (p + 23))//2880 +
        (a * (2p + 13))//96 +
        (c * (p + 11))//36 +
        (b//8) +
        ((a * c)//12) +
        d

    # Compute h
    # Use a small lookup table for offset in h
    # Only 1, 5, 7, 11 matter; default 0 otherwise
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

    # Return result
    return Int(h), Int((h * (h + 1)) ÷ 2), Int(H)
end


"""
    RHI2(p, param)

Given Bp = (-1, -p | Q) and polarization param = [u0, v0, w0, x0, y0, z0] computes the coefficient matrix of the 5-ary refined Humbert invariant which does not represent 1. 

"""
function RHI2(p::Integer, param::AbstractVector{<:Integer})
    u0, v0, w0, x0, y0, z0 = param

    A = zero_matrix(QQ, 6, 6)

    # Fill diagonal
    A[1, 1] = 2 * v0^2
    A[2, 2] = 2 * u0^2
    A[3, 3] = 8 * w0^2 + 8 * w0 * z0 + 2 * z0^2 + 8
    A[4, 4] = 8 * x0^2 + 8 * x0 * y0 + 2 * y0^2 + 8
    A[5, 5] = 2 * x0^2 + 2 * x0 * y0 * p + 2 * x0 * y0 + QQ(1, 2) * y0^2 * p^2 + y0^2 * p + QQ(1, 2) * y0^2 + 2 * p + 2
    A[6, 6] = 2 * w0^2 - 2 * w0 * z0 * p + 2 * w0 * z0 + QQ(1, 2) * z0^2 * p^2 - z0^2 * p + QQ(1, 2) * z0^2 + 2 * p + 2

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

    A[5, 6] = 2 * w0 * x0 + w0 * y0 * p + w0 * y0 - x0 * z0 * p + x0 * z0 - QQ(1, 2) * y0 * z0 * p^2 + QQ(1, 2) * y0 * z0

    # Mirror the upper triangular part to the lower triangular part
    for i = 2:6
        for j = 1:(i-1)
            A[i, j] = A[j, i]
        end
    end

    # Check A is positive semidefinite
    V = quadratic_space(QQ, A)
    D = diagonal(V)
    if !all(>=(0), D)
        println("Not semi-pd")
        return 0
    end

    Azz = map_entries(x -> ZZ(x//2), A)
    AA = lll_gram(Azz)

    # Check rank, symmetry, and last row
    if rank(AA) != 5 || !is_symmetric(AA) || any(!iszero, AA[6, :])
        println("non sym")
        return 0
    end

    # Work with top-left 5×5 submatrix
    B = @view AA[1:5, 1:5]
    
    L = integer_lattice(; gram = B)
    if is_positive_definite(L) && minimum(L) > 1
        C = B .* 2
        return C
    end
    return 0
end


"""
    polyForm(M)

Given the coefficient matrix M of a quadratic form, this function computes the polynomial form using f = 1/2 * (X^t * M * X).

"""
function polyForm(M::ZZMatrix)
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
    allRHI2(p)

Given a prime p congruent to 11 mod 12

"""
function allRHI2(p::Integer)
    start_time = time()
    println("working with prime ", p)

    # Open file once for writing.
    filename = "./RHI2_$(p).txt"

    file = open(filename, "w")
    try
        println(file, "p = ", p, "\n")

        h, h2, H = classNumbers(p)
        println(file, "H = ", H)
        println(file, "H-h(h+1)/2 = ", H-h2, "\n")

        idx = 0   # Counting unique forms.
        total = 0 # Total RHIs computed.

        unique_forms = Vector{ZZMatrix}()
        pol_count = Dict{Integer,Integer}()
        rep_count = Dict{Integer,Integer}()

        prime, params = fileReader("polarizations_p$(p).txt")
        count = length(params)

        if prime == p
            for param in params
                cmA = RHI2(p, param)  # ZZMatrix coefficient matrix.
                if cmA != 0
                    total += 1
                    is_unique = true
                    for (k, cmB) in enumerate(unique_forms)
                        if cmA == cmB
                            is_unique = false
                            pol_count[k] = get(pol_count, k, 1) + 1
                            break
                        end
                    end
                    if is_unique
                        LA = integer_lattice(gram = cmA .÷ 2)
                        for (k, cmB) in enumerate(unique_forms)
                            LB = integer_lattice(gram = cmB .÷ 2)
                            if is_isometric(LA, LB)
                                is_unique = false
                                pol_count[k] = get(pol_count, k, 1) + 1
                                break
                            end
                        end
                    end

                    if is_unique
                        idx += 1
                        println(file, "Type ", idx)
                        s4 = param[4] >= 0 ? "+" : "-"
                        s5 = param[5] >= 0 ? "+" : "-"
                        s6 = param[6] >= 0 ? "+" : "-"
                        a4 = abs(param[4])
                        a5 = abs(param[5])
                        a6 = abs(param[6])

                        s4b = (-param[4]) >= 0 ? "+" : "-"
                        s5b = (-param[5]) >= 0 ? "+" : "-"
                        s6b = (-param[6]) >= 0 ? "+" : "-"
                        a4b = abs(-param[4])
                        a5b = abs(-param[5])
                        a6b = abs(-param[6])

                        println(
                            file,
                            "θ = [",
                            param[1],
                            "  ",
                            param[3],
                            s4,
                            a4,
                            "β₁",
                            s5,
                            a5,
                            "β₂",
                            s6,
                            a6,
                            "β₃]",
                        )
                        println(
                            file,
                            "    [",
                            param[3],
                            s4b,
                            a4b,
                            "β₁",
                            s5b,
                            a5b,
                            "β₂",
                            s6b,
                            a6b,
                            "β₃  ",
                            param[2],
                            "]",
                        )

                        push!(unique_forms, cmA)

                        q = polyForm(cmA)
                        println(file, "q(A,θ) = ", q)

                        L = integer_lattice(gram = cmA .÷ 2)
                        ways = 2*length(short_vectors(L, 4, 4))
                        println(file, "ways to represent 4 = ", ways, "\n")
                        rep_count[ways] = get(rep_count, ways, 0) + 1
                    end
                end
            end
        end                

        println(file, "total polarizations checked: ", count)
        println(file, "total RHI's computed: ", total, "\n")
        println(file, "polarization leading to same type: ", pol_count, "\n")
        println(file, "representation of 4 distribution: ", rep_count, "\n")

        println(file, "p = $p (IKO Thm 3.3 (I), grouped by Aut(C))")
        for (caseno, autname, gamma) in AUT_ROWS
            println(file, "($caseno) Aut(C) ≅ $autname : ", auto_count_IKO(p, gamma))
        end
        println(file, "\n")

        end_time = time()
        elapsed_time = end_time - start_time
        hours = floor(elapsed_time / 3600)
        minutes = floor((elapsed_time % 3600) / 60)
        seconds = round(elapsed_time % 60)
        println(file, "Total run time: ", hours, " hrs ", minutes, " min ", seconds, " sec")
    finally
        close(file)
    end
    println("saved data for prime ", p)
    return nothing
end


# # uncomment to run
# for p in 11:12:260
#     if is_probable_prime(p)
#         allRHI2(p)
#     end
# end