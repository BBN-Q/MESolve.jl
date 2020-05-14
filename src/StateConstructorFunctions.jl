"""
This function creates the operators for a Harmonic oscillator of the specified dimension.
Input arguments:
    dim: Hilbert space dimesnion of the oscillator
Output:
    (lower, raise, N): tuple of the lowering, raising, and photon number operator for the HO
"""
function create_HO(dim::Int)

    lower = zeros(dim,dim)
    for ii = 1:(dim-1)
        lower[ii,ii+1] = sqrt(ii)
    end

    raise = lower'

    N = raise*lower

    return lower, raise, N
end

"""
This function creates the Hamiltonian for a Harmonic oscillator network.
Input arguments:
    dim: Hilbert space dimesnion of the oscillators
    freqs: frequencies of the oscillators
    couple: matrix of tunnel couplings (beamsplitter interaction) between the HOs. Only the upper triangle is used, and the diagonal should be zero. For row index j and column index k, couple[j,k] is the rate for the operator raise_j ⊗ lower_k
Output:
    H: full Hamiltonian of the network
"""
function create_HO_network(dim::Int,freqs::Vector{Float64},couple::Array{T,2}) where {T <: Number}

    num = length(freqs)
    (lower, raise, N) = create_HO(dim)

    H = zeros(dim^num,dim^num)

    # Tunnel coupling (note the normal ordering)
    for jj = 1:1:num
        for kk = (jj+1):1:num
            if abs(couple[jj,kk]) > 1e-10
                raise_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                lower_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(lower,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))

                H = H + couple[jj,kk]*raise_jj*lower_kk
            end
        end
    end

    H = H + H'

    # Self Hamiltonians
    for ii = 1:1:num
        H = H + freqs[ii]*kron(Matrix{ComplexF64}(I,dim^(ii-1),dim^(ii-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-ii),dim^(num-ii))))
    end

    return H

end

# In place version of the above
function create_HO_network(H::Array{T1,2},dim::Int,freqs::Vector{Float64},couple::Array{T2,2}) where {T1 <: Number, T2 <: Number}

    num = length(freqs)
    (lower, raise, N) = create_HO(dim)

    # Tunnel coupling (note the normal ordering)
    for jj = 1:1:num
        for kk = (jj+1):1:num
            if abs(couple[jj,kk]) > 1e-10
                raise_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                lower_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(lower,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))

                H[:,:] = H[:,:] + couple[jj,kk]*raise_jj*lower_kk
            end
        end
    end

    H[:,:] = H[:,:] + H[:,:]'

    # Self Hamiltonians
    for ii = 1:1:num
        H[:,:] = H[:,:] + freqs[ii]*kron(Matrix{ComplexF64}(I,dim^(ii-1),dim^(ii-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-ii),dim^(num-ii))))
    end

    nothing

end

"""
This function creates the Hamiltonian for a Kerr parametric oscillator network.
Input arguments:
    dim: Hilbert space dimesnion of the oscillators
    freqs: frequencies of the oscillators
    couple: matrix of tunnel couplings (beamsplitter interaction) between the HOs. Only the upper triangle is used, and the diagonal should be zero. For row index j and column index k, couple[j,k] is the rate for the operator raise_j ⊗ lower_k
    sqz: matrix of squeezing interaction rates. Only the upper triangle is used, with the diagonal describing single-mode, and the off-diagonal two-mode squeezing. For row index j and column index k, couple[j,k] is the rate for the operator raise_j ⊗ raise_k
    kerr: matrix of the self and cross Kerr interaction rates. Only the upper triangle is used with the diagonal describing self, and the off-diagonal cross Kerr. For row index j and column index k, couple[j,k] is the rate for the operator N_j ⊗ N_k. Note that this is NOT NORMAL ORDERED for j = k
Output:
    H: full Hamiltonian of the network
"""
function create_KPO_network(dim::Int,freqs::Vector{Float64},couple::Array{T1,2},sqz::Array{T2,2},kerr::Array{Float64,2}) where {T1 <: Number, T2 <: Number}

    num = length(freqs)
    (lower, raise, N) = create_HO(dim)

    H = zeros(dim^num,dim^num)

    # Squeezing and Kerr (factor of 1/2 in Kerr term is to account for doing H + H' later)
    for jj = 1:1:num
        for kk = jj:1:num
            if abs(sqz[jj,kk]) > 1e-10
                raise_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                raise_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))

                H = H + sqz[jj,kk]*raise_jj*raise_kk
            end

            if abs(kerr[jj,kk]) > 1e-10

                N_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                N_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))

                H = H + kerr[jj,kk]*N_jj*N_kk/2  # This expression is NOT NORMAL ORDERED for jj = kk
            end
        end
    end

    H = H + H' + create_HO_network(dim,freqs,couple)

    return H

end

# In place version of the above
function create_KPO_network(H::Array{T1,2},dim::Int,freqs::Vector{Float64},couple::Array{T2,2},sqz::Array{T3,2},kerr::Array{Float64,2}) where {T1 <: Number, T2 <: Number, T3 <: Number}

    num = length(freqs)
    (lower, raise, N) = create_HO(dim)

    # Squeezing and Kerr (factor of 1/2 in Kerr term is to account for doing H + H' later)
    for jj = 1:1:num
        for kk = jj:1:num
            if abs(sqz[jj,kk]) > 1e-10
                raise_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                raise_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))

                N_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                N_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))

                H[:,:] = H[:,:] + sqz[jj,kk]*raise_jj*raise_kk
            end

            if abs(kerr[jj,kk]) > 1e-10

                N_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                N_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(N,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))

                H[:,:] = H[:,:] + kerr[jj,kk]*N_jj*N_kk/2  # This expression is NOT NORMAL ORDERED for jj = kk
            end
        end
    end

    H[:,:] = H[:,:] + H[:,:]' + create_HO_network(dim,freqs,couple)

    nothing

end

"""
This function creates the operators for a single qubit. Note the convention used that (1,0)^T is the excited state.
Output:
    X, Y, Z: Pauli matrices
    lower: lowering operator
    raise: raising operator
"""
function create_Qubit()

    Z = [1. 0.;0. -1.]
    X = [0. 1.;1. 0.]
    Y = -1im*Z*X
    lower = (X - 1im*Y)/2
    raise = lower'

    return X, Y, Z, lower, raise

end

"""
This function creates the operators for a single qubit. Note the convention used that (1,0)^T is the excited state.
Input arguments:
    conv: +1 or -1 indicating the convention that either (1,0)^T or (0,1)^T is the excited state. +1 and -1 imply a qubit self Hamiltonian of +Z or -Z respectively.
Output:
    X, Y, Z: Pauli matrices
    lower: lowering operator
    raise: raising operator
"""
function create_Qubit(conv::Int)

    Z = [1. 0.;0. -1.]
    X = [0. 1.;1. 0.]
    Y = -1im*Z*X
    lower = (X - conv*1im*Y)/2
    raise = lower'

    return X, Y, Z, lower, raise

end

"""
This function creates a network of qubits with the coupling type specified and returns its Hamiltonian.
Input arguments:
REQUIRED
    freqs: vector of frequencies of the qubits
    couple: matrix of coupling rates between the qubits. Only the upper triangle is used, and the diagonal should be zero. For row index j and column index k, couple[j,k] is the rate for the operator raise_j ⊗ lower_k for the (default) RWA flipflop interaction
KEYWORD OPTIONAL
    type: one of {"XX","YY","ZZ","flipflop"} describing the type of the interaction. The first three possibilites describe Pauli matrix coupling between the qubits. If type = "flipflop" or is unset, then the interaction is the RWA, excitation number conserving flip-flop interaction raise_j ⊗ lower_k + h.c.. Note that the type is consisnent across the qubit network.
    conv: +1 or -1 indicating the convention that either (1,0)^T or (0,1)^T is the excited state. +1 and -1 imply a qubit self Hamiltonian of +Z or -Z respectively.
Output:
    H: full Hamiltonian of the network
"""
function create_Qubit_network(freqs::Vector{Float64},couple::Array{T,2};Ctype::String = "flipflop",conv::Int = 1) where {T <: Number}

    num = length(freqs)
    X, Y, Z, lower, raise = create_Qubit(conv)

    H = zeros(2^num,2^num)

    # Coupling
    if Ctype == "XX"
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10 && imag(couple[jj,kk]) < 1e-10
                    # X_jj = kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(X,Matrix{ComplexF64}(I,2^(num-jj),2^(num-jj))))
                    # X_kk = kron(Matrix{ComplexF64}(I,2^(kk-1),2^(kk-1)),kron(X,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))
                    # H = H + couple[jj,kk]*X_jj*X_kk
                    H = H + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(X,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(X,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end

    elseif Ctype == "YY"
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10 && imag(couple[jj,kk]) < 1e-10
                    # Y_jj = kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(Y,Matrix{ComplexF64}(I,2^(num-jj),2^(num-jj))))
                    # Y_kk = kron(Matrix{ComplexF64}(I,2^(kk-1),2^(kk-1)),kron(Y,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))
                    # H = H + couple[jj,kk]*Y_jj*Y_kk
                    H = H + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(Y,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(Y,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end

    elseif Ctype == "ZZ"
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10 && imag(couple[jj,kk]) < 1e-10
                    # Y_jj = kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(Y,Matrix{ComplexF64}(I,2^(num-jj),2^(num-jj))))
                    # Y_kk = kron(Matrix{ComplexF64}(I,2^(kk-1),2^(kk-1)),kron(Y,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))
                    # H = H + couple[jj,kk]*Y_jj*Y_kk
                    H = H + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(Z,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(Z,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end

    else
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10
                    # raise_jj = kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(raise,Matrix{ComplexF64}(I,2^(num-jj),2^(num-jj))))
                    # lower_kk = kron(Matrix{ComplexF64}(I,2^(kk-1),2^(kk-1)),kron(lower,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))
                    # H = H + couple[jj,kk]*raise_jj*lower_kk
                    H = H + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(raise,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(lower,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end
        H = H + H'
    end

    # Self Hamiltonians
    for ii = 1:1:num
        H = H + conv*freqs[ii]*kron(Matrix{ComplexF64}(I,2^(ii-1),2^(ii-1)),kron(Z,Matrix{ComplexF64}(I,2^(num-ii),2^(num-ii))))
    end

    return H
end

# In place version of above
function create_Qubit_network(H::Array{T1,2},freqs::Vector{Float64},couple::Array{T2,2};Ctype::String = "flipflop",conv::Int = 1) where {T1 <: Number, T2 <: Number}

    num = length(freqs)
    X, Y, Z, lower, raise = create_Qubit(conv)

    # Coupling
    if Ctype == "XX"
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10 && imag(couple[jj,kk]) < 1e-10
                    H[:,:] = H[:,:] + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(X,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(X,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end

    elseif Ctype == "YY"
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10 && imag(couple[jj,kk]) < 1e-10
                    H[:,:] = H[:,:] + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(Y,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(Y,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end

    elseif Ctype == "ZZ"
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10 && imag(couple[jj,kk]) < 1e-10
                    H[:,:] = H[:,:] + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(Z,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(Z,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end

    else
        for jj = 1:1:num
            for kk = (jj+1):1:num
                if abs(couple[jj,kk]) > 1e-10
                    H[:,:] = H[:,:] + couple[jj,kk]*kron(Matrix{ComplexF64}(I,2^(jj-1),2^(jj-1)),kron(raise,kron(Matrix{ComplexF64}(I,2^(kk-jj-1),2^(kk-jj-1)),kron(lower,Matrix{ComplexF64}(I,2^(num-kk),2^(num-kk))))))
                end
            end
        end
        H[:,:] = H[:,:] + H[:,:]'
    end

    # Self Hamiltonians
    for ii = 1:1:num
        H[:,:] = H[:,:] + conv*freqs[ii]*kron(Matrix{ComplexF64}(I,2^(ii-1),2^(ii-1)),kron(Z,Matrix{ComplexF64}(I,2^(num-ii),2^(num-ii))))
    end

    nothing
end
