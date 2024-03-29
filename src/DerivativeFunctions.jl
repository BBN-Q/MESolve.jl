function dissipator!(dρ_L::Array{ComplexF64,2},
              ρ::Array{ComplexF64,2},
              γs::Vector{Array{ComplexF64,2}},
              γTs::Vector{Array{ComplexF64,2}},
              γSqs::Vector{Array{ComplexF64,2}},
              A::Array{ComplexF64, 2},
              B::Array{ComplexF64, 2},
              α::Vector{Float64}) where {T <: Number}

    for ii = 1:length(γs)
        mul!(B, γSqs[ii], ρ )            # B = γSq ρ
        mul!(B, ρ, γSqs[ii], 1, 1)       # B = ρ γSq + γSq ρ
        mul!(A, ρ, γTs[ii])              # A = ρ γ'
        mul!(B, γs[ii], A, α[ii], -α[ii])        # B = γ (ρ γ') - (ρ γSq + γSq ρ)
        dρ_L .+= B
    end
    nothing
end

function dissipator!(dρ_L::Array{ComplexF64,2},
              ρ::Array{ComplexF64,2},
              γs::Vector{Array{ComplexF64,2}},
              γTs::Vector{Array{ComplexF64,2}},
              γSqs::Vector{Array{ComplexF64,2}},
              A::Array{ComplexF64, 2},
              B::Array{ComplexF64, 2}) where {T <: Number}

    for ii = 1:length(γs)
        mul!(B, γSqs[ii], ρ )            # B = γSq ρ
        mul!(B, ρ, γSqs[ii], 1, 1)       # B = ρ γSq + γSq ρ
        mul!(A, ρ, γTs[ii])              # A = ρ γ'
        mul!(B, γs[ii], A, 1, -1)        # B = γ (ρ γ') - (ρ γSq + γSq ρ)
        dρ_L .+= B
    end
    nothing
end


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
#     out = 1im*(rho*H-H*rho) + dRho_L
# end

"""
In place time independent
"""
function dRho!(dρ::Array{T,2},
              ρ::Array{ComplexF64,2},
              H::Array{ComplexF64,2},
              γs::Vector{Array{ComplexF64,2}},
              γTs::Vector{Array{ComplexF64,2}},
              γSqs::Vector{Array{ComplexF64,2}},
              dρ_L::Array{ComplexF64,2},
              A::Array{ComplexF64, 2},
              B::Array{ComplexF64, 2}) where {T <: Number}

    # Dissipation. Rates and numerical pre-factors have been absorbed into γ terms
    fill!(dρ_L,0.0im)
    dissipator!(dρ_L, ρ, γs, γTs, γSqs, A, B)

    # # Rates and numerical pre-factors have been absorbed into γ terms
    # for ii = 1:length(γs)
    #     mul!(B, γSqs[ii], ρ )            # B = γSq ρ
    #     mul!(B, ρ, γSqs[ii], 1, 1)       # B = ρ γSq + γSq ρ
    #     mul!(A, ρ, γTs[ii])              # A = ρ γ'
    #     mul!(B, γs[ii], A, 1, -1)        # B = γ (ρ γ') - (ρ γSq + γSq ρ)
    #     dρ_L .+= B
    # end

    # Hamiltonian evolution
    mul!(A, H, ρ)                        # A = H ρ
    mul!(A, ρ, H, 1, -1)                 # A = ρ H - H ρ

    dρ .= 1.0im*A .+ dρ_L                # dρ = -i( H ρ - ρ H ) + dρ_L
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
function dRho!(dρ::Array{T,2},
              ρ::Array{ComplexF64,2},
              H::Function,
              H_temp::Array{ComplexF64,2},
              γs::Vector{Array{ComplexF64,2}},
              γTs::Vector{Array{ComplexF64,2}},
              γSqs::Vector{Array{ComplexF64,2}},
              dρ_L::Array{ComplexF64,2},
              A::Array{ComplexF64, 2},
              B::Array{ComplexF64, 2},
              t::AbstractFloat) where {T <: Number}

    # Dissipation. Rates and numerical pre-factors have been absorbed into γ terms
    fill!(dρ_L,0.0im)
    dissipator!(dρ_L, ρ, γs, γTs, γSqs, A, B)

    # Hamiltonian evolution
    fill!(H_temp,0.0im)
    H(H_temp,t)
    mul!(A, H_temp, ρ)                        # A = H ρ
    mul!(A, ρ, H_temp, 1, -1)                 # A = ρ H - H ρ

    dρ .= 1.0im*A .+ dρ_L                # dρ = -i( H ρ - ρ H ) + dρ_L
    nothing
end

"""
# In place time dependent Hamiltonian with matrix basis and vector function inputs
"""

function dRho!(dρ::Array{T,2},
              ρ::Array{ComplexF64,2},
              Hops::Array{ComplexF64,3},
              Hfuncs::Function,
              H_temp::Array{ComplexF64,2},
              Hf_temp::Array{ComplexF64,1},
              γs::Vector{Array{ComplexF64,2}},
              γTs::Vector{Array{ComplexF64,2}},
              γSqs::Vector{Array{ComplexF64,2}},
              dρ_L::Array{ComplexF64,2},
              A::Array{ComplexF64, 2},
              B::Array{ComplexF64, 2},
              t::AbstractFloat) where {T <: Number}

    # Dissipation. Rates and numerical pre-factors have been absorbed into γ terms
    fill!(dρ_L,0.0im)
    dissipator!(dρ_L, ρ, γs, γTs, γSqs, A, B)

    # Hamiltonian evolution
    fill!(H_temp,0.0im)
    Hfuncs(Hf_temp,t)

    for jj = 1:size(Hops,3)
        mul!(H_temp,Hf_temp[jj],Hops[:,:,jj],1,1)
        # H_temp .+= Hf_temp[jj] * Hops[:,:,jj]
    end
    mul!(A, H_temp, ρ)                        # A = H ρ
    mul!(A, ρ, H_temp, 1, -1)                 # A = ρ H - H ρ

    dρ .= 1.0im*A .+ dρ_L                # dρ = -i( H ρ - ρ H ) + dρ_L
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
function dRho!(dρ::Array{T,2},
              ρ::Array{ComplexF64,2},
              H::Array{ComplexF64,2},
              rates::Function,
              rates_t::Array{Float64,1},
              γs::Vector{Array{ComplexF64,2}},
              γTs::Vector{Array{ComplexF64,2}},
              γSqs::Vector{Array{ComplexF64,2}},
              dρ_L::Array{ComplexF64,2},
              A::Array{ComplexF64, 2},
              B::Array{ComplexF64, 2},
              t::AbstractFloat) where {T <: Number}

    # Dissipation. Rates and numerical pre-factors have NOT been absorbed into γ terms
    rates_t = rates(t)
    fill!(dρ_L,0.0im)
    dissipator!(dρ_L, ρ, γs, γTs, γSqs, A, B, rates_t)

    # Hamiltonian evolution
    mul!(A, H, ρ)                        # A = H ρ
    mul!(A, ρ, H, 1, -1)                 # A = ρ H - H ρ

    dρ .= 1.0im*A .+ dρ_L                # dρ = -i( H ρ - ρ H ) + dρ_L
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
function dRho!(dρ::Array{T,2},
              ρ::Array{ComplexF64,2},
              H::Function,
              H_temp::Array{ComplexF64,2},
              rates::Function,
              rates_t::Array{Float64,1},
              γs::Vector{Array{ComplexF64,2}},
              γTs::Vector{Array{ComplexF64,2}},
              γSqs::Vector{Array{ComplexF64,2}},
              dρ_L::Array{ComplexF64,2},
              A::Array{ComplexF64, 2},
              B::Array{ComplexF64, 2},
              t::AbstractFloat) where {T <: Number}

    # Dissipation. Rates and numerical pre-factors have NOT been absorbed into γ terms
    rates_t = rates(t)
    fill!(dρ_L,0.0im)
    dissipator!(dρ_L, ρ, γs, γTs, γSqs, A, B, rates_t)

    # Hamiltonian evolution
    fill!(H_temp,0.0im)
    H(H_temp,t)
    mul!(A, H_temp, ρ)                        # A = H ρ
    mul!(A, ρ, H_temp, 1, -1)                 # A = ρ H - H ρ

    dρ .= 1.0im*A .+ dρ_L                # dρ = -i( H ρ - ρ H ) + dρ_L
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

# LUKE: I think we should remove this functionality.
# """
# # In place for vectorized input
# """
# function dRho_vec(out_vec::Array{T,1},
#                   ρ_v::Array{ComplexF64,1},
#                   H::Array{ComplexF64,2},
#                   γ::Array{ComplexF64,3},
#                   rates::Array{Float64,1},
#                   dρ_L::Array{ComplexF64,2},
#                   ρ::Array{ComplexF64,2},
#                   ut::Array{ComplexF64,2}) where {T <: Number}
#
#     fill!(ρ,0.0im)
#     ρ .= ρ_v
#
#     fill!(dρ_L,0.0im)
#     for ii = 1:size(γ,3)
#         #γT = γ[:,:,ii]'*γ[:,:,ii]
#         dρ_L = dρ_L + rates[ii]*(γ[:,:,ii]*ρ*γ[:,:,ii]' - (γT*ρ + ρ*γT)./2)
#     end
#
#     out_vec .= -1.0im .* (H .* ρ .- ρ .* H) .+ dρ_L
#     nothing
# end

"""
# In place time independent covariance matrix
"""
function dCV!(out::Array{T,2},
             C::Array{ComplexF64,2},
             M::Array{Float64,2},
             Z::Array{ComplexF64,2},
             t::AbstractFloat) where {T <: Number}
    out .= M * C + C * transpose(M) .+ Z
    nothing
end

"""
# In place time independent average vector
"""
function d_av!(out::Vector{Float64},
              av_vec::Vector{Float64},
              M::Array{Float64,2},
              drv::Array{Float64,1},
              t::AbstractFloat)

    out .= M * av_vec .+ drv
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
function dCV!(out::Array{T,2},
             C::Array{ComplexF64,2},
             M::Function,
             Z::Array{ComplexF64,2},
             t::AbstractFloat,
             M_temp::Array{Float64,2}) where {T <: Number}

    M(M_temp,t)
    out .= M_temp * C .+ C * transpose(M_temp[:,:]) .+ Z
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
function d_av!(out::Vector{Float64},
              av_vec::Vector{Float64},
              M::Array{Float64,2},
              drv::Function,
              t::AbstractFloat,
              drv_temp::Array{Float64,1})

    drv(drv_temp,t)
    out .= M * av_vec .+ drv_temp
    nothing
end

function d_av!(out::Vector{Float64},
              av_vec::Vector{Float64},
              M::Function,
              drv::Function,
              t::AbstractFloat,
              M_temp::Array{Float64,2},
              drv_temp::Array{Float64,1})

    M(M_temp,t)
    drv(drv_temp,t)
    out .= M_temp * av_vec .+ drv_temp
    nothing
end
