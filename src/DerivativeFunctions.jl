# Time independent
# function dRho(rho::Array{ComplexF64,2},H::Array{ComplexF64,2},Gamma::Array{ComplexF64,3},rates::Array{Float64,1},dRho_L::Array{ComplexF64,2})
#
#     fill!(dRho_L,0.)
#     for ii = 1:size(Gamma,3)
#         GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
#         dRho_L = dRho_L + rates[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
#     end
#
#     out = -1im*(H*rho-rho*H) + dRho_L
# end

"""
In place time independent
"""
function dRho(out::Array{T,2},rho::Array{ComplexF64,2},H::Array{ComplexF64,2},Gamma::Array{ComplexF64,3},rates::Array{Float64,1},dRho_L::Array{ComplexF64,2}) where {T <: Number}

    fill!(dRho_L,0.) #zeros(Complex{Float64},size(rho,1),size(rho,2))
    for ii = 1:size(Gamma,3)
        GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
        dRho_L = dRho_L + rates[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
    end

    out[:,:] =  -1im*(H*rho-rho*H) + dRho_L
    nothing
end

# Time dependent Hamiltonian
# function dRho(rho::Array{ComplexF64,2},H::Function,Gamma::Array{ComplexF64,3},rates::Array{Float64,1},dRho_L::Array{ComplexF64,2},t::AbstractFloat)
#
#     fill!(dRho_L,0.)
#     for ii = 1:size(Gamma,3)
#         GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
#         dRho_L = dRho_L + rates[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
#     end
#
#     out = -1im*(H(t)*rho-rho*H(t)) + dRho_L
# end

"""
# In place time dependent Hamiltonian with matrix function input
"""
function dRho(out::Array{T,2},rho::Array{ComplexF64,2},H::Function,H_temp::Array{ComplexF64,2},Gamma::Array{ComplexF64,3},rates::Array{Float64,1},dRho_L::Array{ComplexF64,2},t::AbstractFloat) where {T <: Number}

    fill!(dRho_L,0.)
    fill!(H_temp,0.)
    for ii = 1:size(Gamma,3)
        # GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
        dRho_L[:,:] = dRho_L[:,:] + rates[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (Gamma[:,:,ii]'*Gamma[:,:,ii]*rho + rho*Gamma[:,:,ii]'*Gamma[:,:,ii])./2)
    end

    H(H_temp,t)
    out[:,:] = -1im*(H_temp*rho-rho*H_temp) + dRho_L
    nothing
end

"""
# In place time dependent Hamiltonian with matrix basis and vector function inputs
"""

function dRho(out::Array{T,2},rho::Array{ComplexF64,2},Hops::Array{ComplexF64,3},Hfuncs::Function,H_temp::Array{ComplexF64,2},Hf_temp::Array{ComplexF64,1},Gamma::Array{ComplexF64,3},rates::Array{Float64,1},dRho_L::Array{ComplexF64,2},t::AbstractFloat) where {T <: Number}

    fill!(dRho_L,0.)
    fill!(H_temp,0.)
    Hfuncs(Hf_temp,t)
    for jj = 1:size(Hops,3)
        H_temp = H_temp + Hf_temp[jj]*Hops[:,:,jj]
    end

    for ii = 1:size(Gamma,3)
        # GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
        dRho_L[:,:] = dRho_L[:,:] + rates[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (Gamma[:,:,ii]'*Gamma[:,:,ii]*rho + rho*Gamma[:,:,ii]'*Gamma[:,:,ii])./2)
    end

    out[:,:] = -1im*(H_temp*rho-rho*H_temp) + dRho_L
    nothing
end

# # Time dependent dissipator
# function dRho(rho::Array{ComplexF64,2},H::Array{ComplexF64,2},Gamma::Array{ComplexF64,3},rates::Function,dRho_L::Array{ComplexF64,2},rates_t::Array{Float64,1},t::AbstractFloat)
#
#     rates_t = rates(t)
#
#     fill!(dRho_L,0.)
#     for ii = 1:size(Gamma,3)
#         GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
#         dRho_L = dRho_L + rates_t[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
#     end
#
#     out = -1im*(H*rho-rho*H) + dRho_L
# end

"""
# In place time dependent dissipator
"""
function dRho(out::Array{T,2},rho::Array{ComplexF64,2},H::Array{ComplexF64,2},Gamma::Array{ComplexF64,3},rates::Function,dRho_L::Array{ComplexF64,2},rates_t::Array{Float64,1},t::AbstractFloat) where {T <: Number}

    rates_t = rates(t)

    fill!(dRho_L,0.)
    for ii = 1:size(Gamma,3)
        GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
        dRho_L = dRho_L + rates_t[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
    end

    out[:,:] = -1im*(H*rho-rho*H) + dRho_L
    nothing
end

# # Time dependent Hamiltonian and dissipator
# function dRho(rho::Array{ComplexF64,2},H::Function,Gamma::Array{ComplexF64,3},rates::Function,dRho_L::Array{ComplexF64,2},rates_t::Array{Float64,1},t::AbstractFloat)
#
#     rates_t = rates(t)
#
#     fill!(dRho_L,0.)
#     for ii = 1:size(Gamma,3)
#         GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
#         dRho_L = dRho_L + rates_t[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
#     end
#
#     out = -1im*(H(t)*rho-rho*H(t)) + dRho_L
# end

"""
# In place time dependent Hamiltonian and dissipator
"""
function dRho(out::Array{T,2},rho::Array{ComplexF64,2},H::Function,Gamma::Array{ComplexF64,3},rates::Function,dRho_L::Array{ComplexF64,2},rates_t::Array{Float64,1},t::AbstractFloat) where {T <: Number}

    rates_t = rates(t)

    fill!(dRho_L,0.)
    for ii = 1:size(Gamma,3)
        GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
        dRho_L = dRho_L + rates_t[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
    end

    out[:,:] = -1im*(H(t)*rho-rho*H(t)) + dRho_L
    nothing
end

# # For vectorized input
# function dRho_vec(rho_v::Array{ComplexF64,1},H::Array{ComplexF64,2},Gamma::Array{ComplexF64,3},rates::Array{Float64,1},dRho_L::Array{ComplexF64,2},rho::Array{ComplexF64,2})
#
#     fill!(rho,0.)
#     rho[:] = rho_v
#
#     fill!(dRho_L,0.)
#     for ii = 1:size(Gamma,3)
#         GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
#         dRho_L = dRho_L + rates[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
#     end
#
#     out = -1im*(H*rho-rho*H) + dRho_L
#     out_vec = out[:]
# end

"""
# In place for vectorized input
"""
function dRho_vec(out_vec::Array{T,1},rho_v::Array{ComplexF64,1},H::Array{ComplexF64,2},Gamma::Array{ComplexF64,3},rates::Array{Float64,1},dRho_L::Array{ComplexF64,2},rho::Array{ComplexF64,2},out::Array{ComplexF64,2}) where {T <: Number}

    fill!(rho,0.)
    rho[:] = rho_v

    fill!(dRho_L,0.)
    for ii = 1:size(Gamma,3)
        GammaT = Gamma[:,:,ii]'*Gamma[:,:,ii]
        dRho_L = dRho_L + rates[ii]*(Gamma[:,:,ii]*rho*Gamma[:,:,ii]' - (GammaT*rho + rho*GammaT)./2)
    end

    out[:,:] = -1im*(H*rho-rho*H) + dRho_L
    out_vec[:] = out[:]
    nothing
end

"""
# In place time independent covariance matrix
"""
function dCV(out::Array{T,2},C::Array{ComplexF64,2},M::Array{Float64,2},Z::Array{ComplexF64,2},t::AbstractFloat) where {T <: Number}
    out[:,:] = M*C + C*transpose(M) + Z
    nothing
end

"""
# In place time independent average vector
"""
function d_av(out::Vector{Float64},av_vec::Vector{Float64},M::Array{Float64,2},drv::Array{Float64,1},t::AbstractFloat)
    out[:] = M*av_vec + drv
    nothing
end


"""
# In place time dependent covariance matrix
"""
# function dCV(out::Array{T,2},C::Array{ComplexF64,2},h::Function,t::AbstractFloat,h_tempL::Array{Float64,2}) where {T <: Number}
#     h_tempL[:,:] = h(t)
#     out[:,:] = h_tempL*C + C*transpose(h_tempL[:,:])
#     nothing
# end

# New version
function dCV(out::Array{T,2},C::Array{ComplexF64,2},M::Function,Z::Array{ComplexF64,2},t::AbstractFloat,M_temp::Array{Float64,2}) where {T <: Number}
    M(M_temp,t)
    out[:,:] = M_temp*C + C*transpose(M_temp[:,:]) + Z
    nothing
end

# function d_av(out::Vector{Float64},av_vec::Vector{Float64},M::Function,t::AbstractFloat,drv::Function,M_temp::Array{Float64,2},drv_temp::Array{Float64,1})
#     M_temp[:,:] = M(t)
#     drv_temp[:] = drv(t)
#     out[:] = M_temp*av_vec - drv_temp
#     nothing
# end

"""
# In place time dependent average vector
"""
function d_av(out::Vector{Float64},av_vec::Vector{Float64},M::Array{Float64,2},drv::Function,t::AbstractFloat,drv_temp::Array{Float64,1})
    drv(drv_temp,t)
    out[:] = M*av_vec + drv_temp
    nothing
end

function d_av(out::Vector{Float64},av_vec::Vector{Float64},M::Function,drv::Function,t::AbstractFloat,M_temp::Array{Float64,2},drv_temp::Array{Float64,1})
    M(M_temp,t)
    drv(drv_temp,t)
    out[:] = M_temp*av_vec + drv_temp
    nothing
end
