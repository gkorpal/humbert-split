using Oscar

"""
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
Representatives of 5-ary genus rep 1
"""
function Genus5(p::Int)
    A = ZZ[2 0 0 0 0 ; 0 8 0 0 4; 0 0 8 4 0; 0 0 4 2*(p+1) 0; 0 4 0 0 2*(p+1)]
    LA = integer_lattice(; gram = A .÷2)
    Llist = genus_representatives(LA)
    count = 0
    output_filename = "Gen5_$(p).txt"
    open(output_filename, "w") do io
        println(io, "p = ", p, "\n")
        for L in Llist
            if minimum(L) == 1
                M = gram_matrix(L)
                q = polyForm(ZZ.(M .*2))
                println(io, "q = ", q, "\n")
                count += 1
            end
        end
        println(io, "Total Gen5 forms: ", length(Llist))
        println(io, "Total Gen5 forms rep 1: ", count)
    end
    return nothing
end

# Uncomment to run:
# for p in 11:12:660
#     if is_probable_prime(p)
#         Genus5(p)
#     end
# end
