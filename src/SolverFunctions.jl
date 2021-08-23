"""
    me_solve_time_independent(rho_in::Array{T1,2},
                                   H::Array{T2,2},
                                   Gamma::Array{T3,3},
                                   rates::Array{T4,1},
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5()) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat}

Time independent master equation solver using a non-vectorized algorithm.

## ags
REQUIRED
* rho_in:   d x d array, the density matrix of the initial state
* H:        d x d array, system Hamiltonian
* Gamma:    d x d x K array, all K Lindblad operators
* rates:    K x 1 array, dissipative rate for each Lindblad operator
* t0:       float, start time
* tf:       float, end time
KEYWORD OPTIONAL
* tstep:    float, time steps at which output data should be saved
* tols:     2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
* alg:      function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
* iter_num: number of iterations for an adpative solver

## returns
* tvec:     vector of time points where the density matrix has been simulated
* rho_out:  vector of simulated density matricies
"""
function me_solve_time_independent(rho_in::Array{T1,2},
                                   H::Array{T2,2},
                                   Gamma::Array{T3,3},
                                   rates::Array{T4,1},
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5(),
                                   iter_num=1e5) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat}

    rho_in       = convert(Array{ComplexF64,2},rho_in)
    H            = convert(Array{ComplexF64,2},H)
    Gamma        = convert(Array{ComplexF64,3},Gamma)
    rates        = convert(Array{Float64,1},rates)
    dRho_L       = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))

    scratchA     = similar(rho_in);
    scratchB     = similar(rho_in);

    Gammas       = [Gamma[:,:,ii].*sqrt(rates[ii]) for ii=1:size(Gamma,3)]
    GammaTs      = [collect(G') for G in Gammas]
    GammaSqs     = [0.5.*G'G for G in Gammas]

    dif_f(du,u,p,t) = dRho!(du,u,H,Gammas,GammaTs,GammaSqs,dRho_L,scratchA,scratchB) # In place
    tspan = (t0,tf)

    prob = ODEProblem{true}(dif_f,rho_in,tspan)

    if tstep < 1e-10
        tstep = (tf-t0)./100
    end

    sol = solve(prob, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], maxiters=iter_num)

    tvec = sol.t
    rho_out = sol.u

    return tvec, rho_out
end
"""
    me_solve_H_time_dependent(rho_in::Array{T1,2},
                                   H::Function,
                                   Gamma::Array{T3,3},
                                   rates::Array{T4,1},
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5(),
                                   iter_num=1e5,
                                   stop_points=[]
                                   adapt::Bool = true,
                                   δt::Float64 = tols[1],
                                   override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat}

Time dependent (Hamiltonian) master equation solver using a non-vectorized algorithm.

## args
REQUIRED
* rho_in:       d x d array, the density matrix of the initial state
* H:            function, system Hamiltonian, takes input of time and returns the
                    Hamiltonian at the given time in matrix form
* Gamma:        d x d x K array, all K Lindblad operators
* rates:        K x 1 array, dissipative rate for each Lindblad operator
* t0:           float, start time
* tf:           float, end time
KEYWORD OPTIONAL
* tstep:        float, time steps at which output data should be saved
* tols:         2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
* alg:          function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
* iter_num:     number of iterations for an adpative solver
* stop_points:  vector of time-points the solver must step through (for adaptive solvers)
* adapt:        Boolean. If false, the solver assumes it is using a fixed
                timestep integration method.
* δt:           Timestep for fixed timestep integration method.
* override:     Booelan. If true, overrides the error checking that the data
                save timestep (tstep) is not less than the fixed integration
                timestep (δt).

## returns
* tvec:         vector of time points where the density matrix has been simulated
* rho_out:      vector of simulated density matricies
"""
function me_solve_H_time_dependent(rho_in::Array{T1,2},
                                   H::Function,
                                   Gamma::Array{T3,3},
                                   rates::Array{T4,1},
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5(),
                                   iter_num=1e5,
                                   stop_points=[],
                                   adapt::Bool = true,
                                   δt::Float64 = tols[1],
                                   override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat}

    rho_in = convert(Array{ComplexF64,2},rho_in)
    Gamma = convert(Array{ComplexF64,3},Gamma)
    rates = convert(Array{Float64,1},rates)

    dRho_L = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
    H_temp = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))

    scratchA     = similar(rho_in);
    scratchB     = similar(rho_in);

    Gammas       = [Gamma[:,:,ii].*sqrt(rates[ii]) for ii=1:size(Gamma,3)]
    GammaTs      = [collect(G') for G in Gammas]
    GammaSqs     = [0.5.*G'G for G in Gammas]

    dif_f(du,u,p,t) = dRho!(du,u,H,H_temp,Gammas,GammaTs,GammaSqs,dRho_L,scratchA,scratchB,t) # In place
    tspan = (t0,tf)

    prob = ODEProblem{true}(dif_f,rho_in,tspan)

    if tstep < 1e-10
        tstep = (tf-t0)./100
    end

    if adapt
        sol = solve(prob, alg, saveat = tstep, tstops = stop_points, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], maxiters=iter_num)
    else
        if tstep < δt && override == false
            error("You are using a fixed-timestep solver and the save timestep is smaller than solver timestep. This is probably a bad idea. If you insist, set override = true in the input arguments to override this error checking.")
        end
        sol = solve(prob, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], adaptive = adapt, dt = δt)
    end

    tvec = sol.t
    rho_out = sol.u

    return tvec, rho_out
end

"""
    me_solve_H_time_dependent(rho_in::Array{T1,2},
                                   Hops::Array{T5,3},
                                   Hfuncs::Function,
                                   Gamma::Array{T3,3},
                                   rates::Array{T4,1},
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5(),iter_num=1e5,
                                   stop_points=[],
                                   adapt::Bool = true,
                                   δt::Float64 = tols[1],
                                   override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat, T5 <: Number}

Time dependent (Hamiltonian) master equation solver using a non-vectorized algorithm. Instead of inputing the Hamiltonian as a matrix function, it is input as a basis of matrices, and a function that describes the time evolution of the scalar prefactor of each basis element.

This solver also allows for fixed timestep integration.

Input arguments (different from above):
## args
REQUIRED
* Hops:     Array of matrices that describe the Hamiltonian. In the most
                general case is a full basis for operator space.
* Hfuncs:   Function that takes as input a Float (the time) and returns
                a Vector of Floats. Each element of the output vector is
                the value of the scalar prefactor for the corresponding
                basis element in the decomposition of the Hamiltonian.

## returns
* tvec:         vector of time points where the density matrix has been simulated
* rho_out:      vector of simulated density matricies
"""
function me_solve_H_time_dependent(rho_in::Array{T1,2},
                                   Hops::Array{T5,3},
                                   Hfuncs::Function,
                                   Gamma::Array{T3,3},
                                   rates::Array{T4,1},
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5(),
                                   iter_num=1e5,
                                   stop_points=[],
                                   adapt::Bool = true,
                                   δt::Float64 = tols[1],
                                   override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat, T5 <: Number}

    rho_in = convert(Array{ComplexF64,2},rho_in)
    Gamma = convert(Array{ComplexF64,3},Gamma)
    rates = convert(Array{Float64,1},rates)
    Hops = convert(Array{ComplexF64,3},Hops)

    dRho_L = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
    H_temp = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
    Hf_temp = zeros(ComplexF64,size(Hops,3))

    scratchA     = similar(rho_in);
    scratchB     = similar(rho_in);

    Gammas       = [Gamma[:,:,ii].*sqrt(rates[ii]) for ii=1:size(Gamma,3)]
    GammaTs      = [collect(G') for G in Gammas]
    GammaSqs     = [0.5.*G'G for G in Gammas]

    dif_f(du,u,p,t) = dRho!(du,u,Hops,Hfuncs,H_temp,Hf_temp,Gammas,GammaTs,GammaSqs,dRho_L,scratchA,scratchB,t) # In place
    tspan = (t0,tf)

    prob = ODEProblem{true}(dif_f,rho_in,tspan)

    if tstep < 1e-10
        tstep = (tf-t0)./100
    end

    if adapt
        sol = solve(prob, alg, saveat = tstep, tstops = stop_points, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], maxiters=iter_num)
    else
        if tstep < δt && override == false
            error("You are using a fixed-timestep solver and the save timestep is smaller than solver timestep. This is probably a bad idea. If you insist, set override = true in the input arguments to override this error checking.")
        end
        sol = solve(prob, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], adaptive = adapt, dt = δt)
    end

    tvec = sol.t
    rho_out = sol.u

    return tvec, rho_out
end

"""
    me_solve_L_time_dependent(rho_in::Array{T1,2},
                                   H::Array{T2,2},
                                   Gamma::Array{T3,3},
                                   rates::Function,
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5(),
                                   iter_num=1e5,
                                   stop_points=[],
                                   adapt::Bool = true,
                                   δt::Float64 = tols[1],
                                   override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number}

Time dependent (dissipative rates) master equation solver using a non-vectorized algorithm.

## args
* rho_in:       d x d array, the density matrix of the initial state
* H:            d x d array, system Hamiltonian
* Gamma:        d x d x K array, all K Lindblad operators
* rates:        K x 1 array, dissipative rate for each Lindblad operator, takes input time and returns a vector of the dissipative rates at the given time
* t0:           float, start time
* tf:           float, end time
KEYWORD OPTIONAL
* tstep:        float, time steps at which output data should be saved
* tols:         2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
* alg:          function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
* iter_num:     number of iterations for an adpative solver
* stop_points:  vector of time-points the solver must step through (for adaptive solvers)
* adapt:        Boolean. If false, the solver assumes it is using a fixed
                timestep integration method.
* δt:           Timestep for fixed timestep integration method.
* override:     Booelan. If true, overrides the error checking that the data
                save timestep (tstep) is not less than the fixed integration
                timestep (δt).

## returns
* tvec:         vector of time points where the density matrix has been simulated
* rho_out:      vector of simulated density matricies
"""
function me_solve_L_time_dependent(rho_in::Array{T1,2},
                                   H::Array{T2,2},
                                   Gamma::Array{T3,3},
                                   rates::Function,
                                   t0::AbstractFloat,
                                   tf::AbstractFloat;
                                   tstep::AbstractFloat=0.,
                                   tols::Vector{Float64}=[1e-6,1e-3],
                                   alg = Tsit5(),
                                   iter_num=1e5,
                                   stop_points=[],
                                   adapt::Bool = true,
                                   δt::Float64 = tols[1],
                                   override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number}

    rho_in = convert(Array{ComplexF64,2},rho_in)
    H = convert(Array{ComplexF64,2},H)
    Gamma = convert(Array{ComplexF64,3},Gamma)

    scratchA     = similar(rho_in);
    scratchB     = similar(rho_in);

    Gammas       = [Gamma[:,:,ii] for ii=1:size(Gamma,3)]
    GammaTs      = [collect(G') for G in Gammas]
    GammaSqs     = [0.5.*G'G for G in Gammas]

    dRho_L = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
    rates_temp = Array{Float64}(undef,1)

    dif_f(du,u,p,t) = dRho!(du,u,H,rates,rates_temp,Gammas,GammaTs,GammaSqs,dRho_L,scratchA,scratchB,t) # In place
    tspan = (t0,tf)

    prob = ODEProblem{true}(dif_f,rho_in,tspan)

    if tstep < 1e-10
        tstep = (tf-t0)./100
    end

    if adapt
        sol = solve(prob, alg, saveat = tstep, tstops = stop_points, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], maxiters=iter_num)
    else
        if tstep < δt && override == false
            error("You are using a fixed-timestep solver and the save timestep is smaller than solver timestep. This is probably a bad idea. If you insist, set override = true in the input arguments to override this error checking.")
        end
        sol = solve(prob, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], adaptive = adapt, dt = δt)
    end

    tvec = sol.t
    rho_out = sol.u

    return tvec, rho_out
end

"""
    me_solve_full_time_dependent(rho_in::Array{T1,2},
                                      H::Function,
                                      Gamma::Array{T3,3},
                                      rates::Function,
                                      t0::AbstractFloat,
                                      tf::AbstractFloat;
                                      tstep::AbstractFloat=0.,
                                      tols::Vector{Float64}=[1e-6,1e-3],
                                      alg = Tsit5(),
                                      stop_points=[],
                                      adapt::Bool = true,
                                      δt::Float64 = tols[1],
                                      override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number}

Time dependent (Hamiltonian and dissipative rates) master equation solver using a non-vectorized algorithm.

## args
REQUIRED
* rho_in:       d x d array, the density matrix of the initial state
* H:            function, system Hamiltonian, takes input of time and returns
                    the Hamiltonian at the given time in matrix form
* Gamma:        d x d x K array, all K Lindblad operators
* rates:        K x 1 array, dissipative rate for each Lindblad operator,
                    takes input time and returns a vector of the dissipative
                    rates at the given time
* t0:           float, start time
* tf:           float, end time
KEYWORD OPTIONAL
* tstep:        float, time steps at which output data should be saved
* tols:         2 x 1 array, vector of solver tolernaces in the
                    order [abstol, reltol]
* alg:          function, algorithm from DifferentialEquations for the solver
                    to use, default is Tsit5
* iter_num:     number of iterations for an adpative solver
* stop_points:  vector of time-points the solver must step through
                    (for adaptive solvers)
* adapt:        Boolean. If false, the solver assumes it is using a fixed
                timestep integration method.
* δt:           Timestep for fixed timestep integration method.
* override:     Booelan. If true, overrides the error checking that the data
                save timestep (tstep) is not less than the fixed integration
                timestep (δt).

## returns
* tvec:         vector of time points where the density matrix has been simulated
* rho_out:      vector of simulated density matricies
"""
function me_solve_full_time_dependent(rho_in::Array{T1,2},
                                      H::Function,
                                      Gamma::Array{T3,3},
                                      rates::Function,
                                      t0::AbstractFloat,
                                      tf::AbstractFloat;
                                      tstep::AbstractFloat=0.,
                                      tols::Vector{Float64}=[1e-6,1e-3],
                                      alg = Tsit5(),
                                      iter_num=1e5,
                                      stop_points=[],
                                      adapt::Bool = true,
                                      δt::Float64 = tols[1],
                                      override::Bool = false) where {T1 <: Number, T2 <: Number, T3 <: Number}

    rho_in = convert(Array{ComplexF64,2},rho_in)
    Gamma = convert(Array{ComplexF64,3},Gamma)

    dRho_L = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
    H_temp = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
    rates_temp = Array{Float64}(undef,1)

    scratchA     = similar(rho_in);
    scratchB     = similar(rho_in);

    Gammas       = [Gamma[:,:,ii] for ii=1:size(Gamma,3)]
    GammaTs      = [collect(G') for G in Gammas]
    GammaSqs     = [0.5.*G'G for G in Gammas]

    dif_f(du,u,p,t) = dRho!(du,u,H,H_temp,rates,rates_temp,Gammas,GammaTs,GammaSqs,dRho_L,scratchA,scratchB,t) # In place
    tspan = (t0,tf)

    prob = ODEProblem{true}(dif_f,rho_in,tspan)

    if tstep < 1e-10
        tstep = (tf-t0)./100
    end

    if adapt
        sol = solve(prob, alg, saveat = tstep, tstops = stop_points, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], maxiters=iter_num)
    else
        if tstep < δt && override == false
            error("You are using a fixed-timestep solver and the save timestep is smaller than solver timestep. This is probably a bad idea. If you insist, set override = true in the input arguments to override this error checking.")
        end
        sol = solve(prob, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2], adaptive = adapt, dt = δt)
    end

    tvec = sol.t
    rho_out = sol.u

    return tvec, rho_out
end

"""
    me_solve_time_independent_vec(rho_in::Array{T1,2},
                                       H::Array{T2,2},
                                       Gamma::Array{T3,3},
                                       rates::Array{T4,1},
                                       t0::AbstractFloat,
                                       tf::AbstractFloat;
                                       tstep::AbstractFloat=0.,
                                       tols::Vector{Float64}=[1e-6,1e-3],
                                       alg = Tsit5()) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat}

Time independent master equation solver using a non-vectorized algorithm but with vector ODE solvers

## args
REQUIRED
* rho_in:       d x d array, the density matrix of the initial state
* H:            d x d array, system Hamiltonian
* Gamma:        d x d x K array, all K Lindblad operators
* rates:        K x 1 array, dissipative rate for each Lindblad operator
* t0:           float, start time
* tf:           float, end time
KEYWORD OPTIONAL
* tstep:        float, time steps at which output data should be saved
* tols:         2 x 1 array, vector of solver tolernaces in the
                    order [abstol, reltol]
* alg:          function, algorithm from DifferentialEquations for the
                    solver to use, default is Tsit5

## returns
* tvec:         vector of time points where the density matrix has been simulated
* rho_out:      vector of simulated density matricies
"""
# LUKE: I think we can remove this functionality.
# function me_solve_time_independent_vec(rho_in::Array{T1,2},
#                                        H::Array{T2,2},
#                                        Gamma::Array{T3,3},
#                                        rates::Array{T4,1},
#                                        t0::AbstractFloat,
#                                        tf::AbstractFloat;
#                                        tstep::AbstractFloat=0.,
#                                        tols::Vector{Float64}=[1e-6,1e-3],
#                                        alg = Tsit5()) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: AbstractFloat}
#
#     rho_in = convert(Array{ComplexF64,2},rho_in)
#     H = convert(Array{ComplexF64,2},H)
#     Gamma = convert(Array{ComplexF64,3},Gamma)
#     rates = convert(Array{Float64,1},rates)
#
#     rho_in_vec = rho_in[:]
#
#     dRho_L = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
#     rho_temp = Array{ComplexF64}(undef,size(rho_in,1),size(rho_in,2))
#
#     dif_f(du,u,p,t) = dRho_vec(du,u,H,Gamma,rates,dRho_L,rho_temp) # In place
#     tspan = (t0,tf)
#
#     prob = ODEProblem{true}(dif_f,rho_in_vec,tspan)
#
#     if tstep < 1e-10
#         tstep = (tf-t0)./100
#     end
#
#     sol = solve(prob, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2])
#
#     tvec = sol.t
#     rho_out_vec = sol.u
#
#     rho_out = Array{Array{ComplexF64,2}}(undef,length(tvec))
#     for jj = 1:1:length(tvec)
#         rho_out[jj] = reshape(rho_out_vec[jj][:],size(H))
#     end
#
#     return tvec, rho_out
# end


# """
# Time dependent solver for Gaussian continuous variable systems using the covariance matrix approach. Defined in terms of the canonical X and P quadratures, the covariance matrix elements are defined by
#   C_ab = <O_aO_b> - <O_a><O_b> where vec(O) = [X,P] with
#   X = (a + a^†)/√2, P = -i(a - a^†)/√2 with a the lowering operator of a harmonic oscilaltor.
# Input Parameters:
# REQUIRED
#   C_in: Array 2 x 2, initial condition for the covariance matrix
#   a_in: vector 2 x 1, initial condition for the average value vector [<X>,<P>]
#   h: Function that returnas as output a matrix 2 x 2, Hamiltonian evolution term for the covariance matrix. For H = ω a^†a + (λa^2 + conj(λ)a^†^2)/2, h = [Im(λ), (ω - Re(λ);-(ω + Re(λ)), -Im(λ)].
#       Can use h = create_h_C(ω(t),λ(t),t) to generate the correct h.
#   drv: Function that returns as output a vector 2 x 1, linear drive term. For H = αa + conj(α)a^†, drv = √2[Im(α), Re(α)]
#   t0: float, start time
#   tf: float, end time
# KEYWORD OPTIONAL
#   tstep: float, time steps at which output data should be saved
#   tols: 2 x 1 array, vector of solver tolernaces in the order [abstol, reltol]
#   alg: function, algorithm from DifferentialEquations for the solver to use, default is Tsit5
# """
#
# function CV_solve_full_time_dependent(C_in::Array{T,2},a_in::Vector{Float64},h::Function,drv::Function,t0::AbstractFloat,tf::AbstractFloat; tstep::AbstractFloat=0.,tols::Vector{Float64}=[1e-6,1e-3],alg = Tsit5()) where {T <: Number}
#
#   C_in = convert(Array{ComplexF64,2}, C_in)
#   if tstep < 1e-10
#         tstep = (tf-t0)./100
#     end
#   tspan = (t0,tf)
#
#   # Covariance matrix
#   h_tempL = Array{Float64}(undef,size(C_in,1),size(C_in,2))
#   # h_tempR = Array{Float64}(undef,size(C_in,1),size(C_in,2))
#     dif_C(du,u,p,t) = dCV(du,u,h,t,h_tempL) # In place
#
#     prob_C = ODEProblem{true}(dif_C,C_in,tspan)
#
#     sol_C = solve(prob_C, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2])
#
#   # Average
#   h_temp = Array{Float64}(undef,size(C_in,1),size(C_in,2))
#   drv_temp = Array{Float64}(undef,size(a_in,1))
#   dif_a(du,u,p,t) = d_av(du,u,h,t,drv,h_temp,drv_temp) # In place
#
#     prob_a = ODEProblem{true}(dif_a,a_in,tspan)
#
#     sol_a = solve(prob_a, alg, saveat = tstep, dense=false, save_everystep=false, abstol = tols[1], reltol = tols[2])
#
# return sol_C.t, sol_C.u, sol_a.t, sol_a.u
# end
