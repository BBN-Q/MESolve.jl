"""
    husimi_Q(rho_in::Array{T1,2},xgrid::Array{Float64,1},
                                      ygrid::Array{Float64,1}) where {T1 <: Number}

Generates the Husimi Q distribution of a quantum state on a rectangular grid in phase space

## args
* rho_in:   d x d array, the density matrix of the state
* xgrid:    vector of floats, x range to calculate the Q function at
* ygrid:    vector of floats, y range to calculate the Q function at

## returns
* Q:        matrix of the calculated Q function
"""
function husimi_Q(rho_in::Array{T1,2},xgrid::Array{Float64,1},
                                      ygrid::Array{Float64,1}) where {T1 <: Number}

    dim = Int(maximum([3*maximum(abs.(xgrid))^2,3*maximum(abs.(ygrid))^2,size(rho_in,1)]))

    # Disp = zeros(size(rho_in))
    Q = zeros(size(xgrid,1),size(ygrid,1))

    lower, raise, N = create_HO(dim)

    rho_temp = zeros(dim,dim)
    rho_temp = convert(Array{ComplexF64,2},rho_temp)
    rho_temp[1:size(rho_in,1),1:size(rho_in,1)] = rho_in

    state = zeros(dim,1)
    state[1] = 1

    for ii = 1:1:size(xgrid,1)
        for jj = 1:1:size(ygrid,1)
            @inbounds temp_a = xgrid[ii]+1im*ygrid[jj]
            Disp = exp(temp_a*raise - conj(temp_a)*lower)
            temp_v = Disp*state
            @inbounds Q[ii,jj] = real(temp_v'*rho_temp*temp_v./pi)[1]
            #Q[ii:ii,jj:jj] = real(temp_v'*rho_temp*temp_v./pi) also works
        end
    end

    return Q
end

"""
    husimi_Q_pw(rho_in::Array{T1,2},
                xgrid::Array{Float64,1},
                ygrid::Array{Float64,1}) where {T1 <: Number}

Calculates the value of the Husimi Q distribution of a quantum state at the specified points

## args
* rho_in:   d x d array, the density matrix of the state
* xgrid:    vector of floats, x coordinates to calculate the Q function at
* ygrid:    vector of floats, y coordinates to calculate the Q function at

## returns
* Q:        vector of the calculated Q function values
"""
function husimi_Q_pw(rho_in::Array{T1,2},
                     xgrid::Array{Float64,1},
                     ygrid::Array{Float64,1}) where {T1 <: Number}

    size(xgrid,1) == size(ygrid,1) || throw(DimensionMismatch("size of xgrid is not equal to size of ygrid"))

    dim = Int(maximum([3*maximum(abs.(xgrid))^2,3*maximum(abs.(ygrid))^2,size(rho_in,1)]))

    # Disp = zeros(size(rho_in))
    Q = zeros(size(xgrid,1))

    lower, raise, N = create_HO(dim)

    rho_temp = zeros(dim,dim)
    rho_temp = convert(Array{ComplexF64,2},rho_temp)
    rho_temp[1:size(rho_in,1),1:size(rho_in,1)] = rho_in

    state = zeros(dim,1)
    state[1] = 1

    for ii = 1:1:size(xgrid,1)
        @inbounds temp_a = xgrid[ii]+1im*ygrid[ii]
        Disp = exp(temp_a*raise - conj(temp_a)*lower)
        temp_v = Disp*state
        @inbounds Q[ii] = real(temp_v'*rho_temp*temp_v./pi)[1]
    end

    return Q
end

# """
# This function creates the coherent part of a Hamiltonian term in the covariance matrix differential equation.
# Input Arguments:
#     freq: Frequency of the oscillator.
#     sqz: Two-photon (parametric) drive complex amplitude (can be time-dependent)
# """
# function create_h_C(freq::Float64,sqz::Function,t::AbstractFloat)
#     return [imag(sqz(t)) (freq - real(sqz(t)));-(freq + real(sqz(t))) -imag(sqz(t))]
# end
