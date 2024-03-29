"""
    create_HO(dim::Int)

This function creates the operators for a Harmonic oscillator of the specified dimension.

## args
* dim: Hilbert space dimesnion of the oscillator

## returns
(lower, raise, N): tuple of the lowering, raising, and photon number operator for the HO
"""
function create_HO(dim::Int)

    lower = zeros(dim,dim)
    for ii = 1:(dim-1)
        @inbounds lower[ii,ii+1] = sqrt(ii)
    end

    raise = lower'

    N = raise*lower

    return lower, raise, N
end

"""
    create_HO_network(dim::Array{Int},freqs::Vector{Float64},couple::Array{T,2}) where {T <: Number}

This function creates the Hamiltonian for a Harmonic oscillator network.

## args
* dim:      array of the Hilbert space dimensions of the oscillators
* freqs:    frequencies of the oscillators
* couple:   matrix of tunnel couplings (beamsplitter interaction) between the HOs. Only the upper triangle is used, and the diagonal should be zero. For row index j and column index k, couple[j,k] is the rate for the operator raise_j ⊗ lower_k

## returns
* H:        full Hamiltonian of the network
"""
function create_HO_network(dim::Array{Int},freqs::Vector{Float64},couple::Array{T,2}) where {T <: Number}

    num = length(freqs)

    H = zeros(prod(dim),prod(dim))

    # Tunnel coupling (note the normal ordering)
    for jj = 1:1:num
        ~, raise1, ~ = create_HO(dim[jj])
        for kk = (jj+1):1:num
            if abs(couple[jj,kk]) > 1e-10
                lower2, ~, ~ = create_HO(dim[kk])
                # raise_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                raise_jj = kron(Matrix{ComplexF64}(I,prod(dim[1:jj-1]),prod(dim[1:jj-1])),kron(raise1,Matrix{ComplexF64}(I,prod(dim[(jj+1):end]),prod(dim[(jj+1):end]))))
                # lower_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(lower,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))
                lower_kk = kron(Matrix{ComplexF64}(I,prod(dim[1:kk-1]),prod(dim[1:kk-1])),kron(lower2,Matrix{ComplexF64}(I,prod(dim[(kk+1):end]),prod(dim[(kk+1):end]))))

                @inbounds H = H + couple[jj,kk]*raise_jj*lower_kk
            end
        end
    end

    H = H + H'

    # Self Hamiltonians
    for ii = 1:1:num
        ~, ~, N1 = create_HO(dim[ii])
        @inbounds H = H + freqs[ii]*kron(Matrix{ComplexF64}(I,prod(dim[1:ii-1]),prod(dim[1:ii-1])),kron(N1,Matrix{ComplexF64}(I,prod(dim[(ii+1):end]),prod(dim[(ii+1):end]))))
    end

    return H

end

# To be updated
# In place version of the above
"""
    create_HO_network!(H::Array{T1,2},dim::Int,freqs::Vector{Float64},couple::Array{T2,2}) where {T1 <: Number, T2 <: Number}

In place version of `create_HO_network` where the Hamiltonian H is modified

## args
* H:      full Hamiltonian of the network
* dim:    array of the Hilbert space dimensions of the oscillators
* freqs:  frequencies of the oscillators
* couple: matrix of tunnel couplings (beamsplitter interaction) between the 
            HOs. Only the upper triangle is used, and the diagonal should be zero. 
            For row index j and column index k, couple[j,k] is the rate for the 
            operator raise_j ⊗ lower_k
## returns
nothing 
"""
function create_HO_network!(H::Array{T1,2},dim::Int,freqs::Vector{Float64},couple::Array{T2,2}) where {T1 <: Number, T2 <: Number}

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
    create_KPO_network(dim::Array{Int},freqs::Vector{Float64},couple::Array{T1,2},sqz::Array{T2,2},kerr::Array{Float64,2}) where {T1 <: Number, T2 <: Number}

This function creates the Hamiltonian for a Kerr parametric oscillator network.

## args
* dim:    array of the Hilbert space dimensions of the oscillators
* freqs:  frequencies of the oscillators
* couple: matrix of tunnel couplings (beamsplitter interaction) between the 
            HOs. Only the upper triangle is used, and the diagonal should be zero. 
            For row index j and column index k, couple[j,k] is the rate for the 
            operator raise_j ⊗ lower_k
* sqz:    matrix of squeezing interaction rates. Only the upper triangle is 
            used, with the diagonal describing single-mode, and the off-diagonal 
            two-mode squeezing. For row index j and column index k, sqz[j,k] is 
            the rate for the operator raise_j ⊗ raise_k
* kerr:   matrix of the self and cross Kerr interaction rates. Only the upper 
            triangle is used with the diagonal describing self, and the 
            off-diagonal cross Kerr. For row index j and column index k, kerr[j,k] 
            is the rate for the operator N_j ⊗ N_k. Note that this is NOT 
            NORMAL ORDERED for j = k

## returns
* H:      full Hamiltonian of the network
"""
function create_KPO_network(dim::Array{Int},freqs::Vector{Float64},couple::Array{T1,2},sqz::Array{T2,2},kerr::Array{Float64,2}) where {T1 <: Number, T2 <: Number}

    num = length(freqs)
    # (lower, raise, N) = create_HO(dim)

    H = zeros(prod(dim),prod(dim))

    # Squeezing and Kerr (factor of 1/2 in Kerr term is to account for doing H + H' later)
    for jj = 1:1:num
        ~, raise1, N1 = create_HO(dim[jj])
        for kk = jj:1:num
            if abs(sqz[jj,kk]) > 1e-10
                ~, raise2, ~ = create_HO(dim[kk])
                # raise_jj = kron(Matrix{ComplexF64}(I,dim^(jj-1),dim^(jj-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-jj),dim^(num-jj))))
                raise_jj = kron(Matrix{ComplexF64}(I,prod(dim[1:jj-1]),prod(dim[1:jj-1])),kron(raise1,Matrix{ComplexF64}(I,prod(dim[(jj+1):end]),prod(dim[(jj+1):end]))))
                # raise_kk = kron(Matrix{ComplexF64}(I,dim^(kk-1),dim^(kk-1)),kron(raise,Matrix{ComplexF64}(I,dim^(num-kk),dim^(num-kk))))
                raise_kk = kron(Matrix{ComplexF64}(I,prod(dim[1:kk-1]),prod(dim[1:kk-1])),kron(raise2,Matrix{ComplexF64}(I,prod(dim[(kk+1):end]),prod(dim[(kk+1):end]))))

                H = H + sqz[jj,kk]*raise_jj*raise_kk
            end

            if abs(kerr[jj,kk]) > 1e-10

                ~, ~, N2 = create_HO(dim[kk])
                N_jj = kron(Matrix{ComplexF64}(I,prod(dim[1:jj-1]),prod(dim[1:jj-1])),kron(N1,Matrix{ComplexF64}(I,prod(dim[(jj+1):end]),prod(dim[(jj+1):end]))))
                N_kk = kron(Matrix{ComplexF64}(I,prod(dim[1:kk-1]),prod(dim[1:kk-1])),kron(N2,Matrix{ComplexF64}(I,prod(dim[(kk+1):end]),prod(dim[(kk+1):end]))))

                H = H + kerr[jj,kk]*N_jj*N_kk/2  # This expression is NOT NORMAL ORDERED for jj = kk
            end
        end
    end

    H = H + H' + create_HO_network(dim,freqs,couple)

    return H

end

# To be updated
# In place version of the above
"""
    create_KPO_network!(H::Array{T1,2},dim::Int,freqs::Vector{Float64},couple::Array{T2,2},sqz::Array{T3,2},kerr::Array{Float64,2}) where {T1 <: Number, T2 <: Number, T3 <: Number}

In place version of `create_KPO_network` where the Hamiltonian H is update.

This function creates the Hamiltonian for a Kerr parametric oscillator network.

## args
* dim:    array of the Hilbert space dimensions of the oscillators
* freqs:  frequencies of the oscillators
* couple: matrix of tunnel couplings (beamsplitter interaction) between the 
            HOs. Only the upper triangle is used, and the diagonal should be zero. 
            For row index j and column index k, couple[j,k] is the rate for the 
            operator raise_j ⊗ lower_k
* sqz:    matrix of squeezing interaction rates. Only the upper triangle is 
            used, with the diagonal describing single-mode, and the off-diagonal 
            two-mode squeezing. For row index j and column index k, sqz[j,k] is 
            the rate for the operator raise_j ⊗ raise_k
* kerr:   matrix of the self and cross Kerr interaction rates. Only the upper 
            triangle is used with the diagonal describing self, and the 
            off-diagonal cross Kerr. For row index j and column index k, kerr[j,k] 
            is the rate for the operator N_j ⊗ N_k. Note that this is NOT 
            NORMAL ORDERED for j = k

## returns 
nothing 
"""
function create_KPO_network!(H::Array{T1,2},dim::Int,freqs::Vector{Float64},couple::Array{T2,2},sqz::Array{T3,2},kerr::Array{Float64,2}) where {T1 <: Number, T2 <: Number, T3 <: Number}

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
    create_Qubit()

This function creates the operators for a single qubit. Note the convention 
used that (1,0)^T is the excited state.

## returns
* X, Y, Z:  Pauli matrices
* lower:    lowering operator
* raise:    raising operator

as a tuple : (X, Y, Z, lower, raise)
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
    create_Qubit(conv::Int)

This function creates the operators for a single qubit. Note the convention 
used that (1,0)^T is the excited state.

## args
* conv:     +1 or -1 indicating the convention that either (1,0)^T or (0,1)^T is 
                the excited state. +1 and -1 imply a qubit self Hamiltonian of 
                +Z or -Z respectively.

## returns:
* X, Y, Z:  Pauli matrices
* lower:    lowering operator
* raise:    raising operator

as a tuple : (X, Y, Z, lower, raise)
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
    create_qubit_network(freqs::Vector{Float64},couple::Array{T,2};Ctype::String = "flipflop",conv::Int = 1) where {T <: Number}

This function creates a network of qubits with the coupling type specified and returns its Hamiltonian.
Input arguments:

## args
* freqs:    vector of frequencies of the qubits
* couple:   matrix of coupling rates between the qubits. Only the upper triangle 
                is used, and the diagonal should be zero. For row index j and 
                column index k, couple[j,k] is the rate for the operator 
                raise_j ⊗ lower_k for the (default) RWA flipflop interaction
* Ctype:    one of {"XX","YY","ZZ","flipflop"} describing the type of the 
                interaction. The first three possibilites describe Pauli 
                matrix coupling between the qubits. If type = "flipflop" or is 
                unset, then the interaction is the RWA, excitation number 
                conserving flip-flop interaction raise_j ⊗ lower_k + h.c.. 
                Note that the type is consisnent across the qubit network.
* conv:     +1 or -1 indicating the convention that either (1,0)^T or (0,1)^T 
                is the excited state. +1 and -1 imply a qubit self Hamiltonian 
                of +Z or -Z respectively.

## returns
* H:        full Hamiltonian of the network
"""
function create_qubit_network(freqs::Vector{Float64},couple::Array{T,2};Ctype::String = "flipflop",conv::Int = 1) where {T <: Number}

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
"""
    create_qubit_network!(H::Array{T1,2},freqs::Vector{Float64},couple::Array{T2,2};Ctype::String = "flipflop",conv::Int = 1) where {T1 <: Number, T2 <: Number}

In place version of `create_Qubit_network` where the Hamiltonian H is updated.

## args
* freqs:    vector of frequencies of the qubits
* couple:   matrix of coupling rates between the qubits. Only the upper 
                triangle is used, and the diagonal should be zero. For 
                row index j and column index k, couple[j,k] is the rate 
                for the operator raise_j ⊗ lower_k for the (default) RWA 
                flipflop interaction
* Ctype:    one of {"XX","YY","ZZ","flipflop"} describing the type of the 
                interaction. The first three possibilites describe Pauli 
                matrix coupling between the qubits. If type = "flipflop" 
                or is unset, then the interaction is the RWA, excitation 
                number conserving flip-flop interaction 
                raise_j ⊗ lower_k + h.c.. Note that the type is consisnent 
                across the qubit network.
* conv:     +1 or -1 indicating the convention that either (1,0)^T or 
                (0,1)^T is the excited state. +1 and -1 imply a qubit self 
                Hamiltonian of +Z or -Z respectively.

## returns
* nothing
"""
function create_qubit_network!(H::Array{T1,2},freqs::Vector{Float64},couple::Array{T2,2};Ctype::String = "flipflop",conv::Int = 1) where {T1 <: Number, T2 <: Number}

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
