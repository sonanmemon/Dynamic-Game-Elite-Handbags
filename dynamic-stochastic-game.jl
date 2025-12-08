##############################################################
# PACKAGES
##############################################################
using Random, LinearAlgebra, Distributions, Optim, Plots

##############################################################
# STRUCTURES
##############################################################
struct State
    B1::Float64
    B2::Float64
    alpha::Float64
    upsilon::Float64
    tau::Float64
    Q::Float64
end

struct Action
    q1::Float64
    I1::Float64
    q2::Float64
    I2::Float64
end

##############################################################
# MODEL FUNCTIONS
##############################################################

psi(tau,q1,Q) = tau * log(1 + q1/Q) / log(2)

rho(I; ϵ=1e-2) = (log(I + ϵ) - log(ϵ)) / (log(1 + ϵ) - log(ϵ))

function brand_transition(B,I,i)
    # firm-specific persistence and investment effectiveness
    persist = (i==1 ? 0.08 : 0.12)
    inv_eff = (i==1 ? 0.35 : 0.22)
    return persist * rho(I) * B + inv_eff * I
end

function prices(S::State, a::Action)
    ψ = psi(S.tau, a.q1, S.Q)

    # asymmetric competitive sensitivities
    p1 = S.alpha - S.upsilon*a.q1 + ψ * (1.1*S.B1 - 0.9*S.B2)
    p2 = S.alpha - S.upsilon*a.q2 + (1-ψ) * (1.05*S.B2 - 0.95*S.B1)

    return p1,p2
end

function profit_i(i::Int, S::State, a::Action)
    p1,p2 = prices(S,a)
    ψ = psi(S.tau, a.q1, S.Q)

    # asymmetric costs
    cq = (i==1 ? 1.0 : 0.85)
    cI = (i==1 ? 1.1 : 0.9)

    if i==1
        return p1*a.q1 - cq*a.q1^2 - cI*(a.I1^2)/ψ
    else
        return p2*a.q2 - cq*a.q2^2 - cI*(a.I2^2)/(1-ψ)
    end
end


##############################################################
# SIMPLE NASH
##############################################################
function best_response(i::Int, S::State, a_op::Action)
    obj(x) = begin
        q,I = clamp.(x,0.0,1.0)
        a = i==1 ? Action(q,I,a_op.q2,a_op.I2) : Action(a_op.q1,a_op.I1,q,I)
        -profit_i(i,S,a)
    end
    res = optimize(obj,[0.5,0.5],[0.0,0.0],[1.0,1.0],NelderMead())
    return clamp.(Optim.minimizer(res),0.0,1.0)
end

function nash_equilibrium(S::State)
    a = Action(0.5,0.5,0.5,0.5)
    for _ in 1:7
        q2,I2 = best_response(2,S,a)
        a2 = Action(a.q1,a.I1,q2,I2)
        q1,I1 = best_response(1,S,a2)
        a_new = Action(q1,I1,a2.q2,a2.I2)
        if norm([a_new.q1-a.q1, a_new.I1-a.I1, a_new.q2-a.q2, a_new.I2-a.I2]) < 1e-4
            return a_new
        end
        a = a_new
    end
    return a
end


##############################################################
# GRID
##############################################################
create_grid(N=15) = collect(range(0.0,1.0,length=N))

function build_policy_grids(Sdummy)
    N = length(create_grid())
    Bgrid = create_grid(N)
    q1grid = zeros(N,N)
    q2grid = zeros(N,N)
    I1grid = zeros(N,N)
    I2grid = zeros(N,N)

    for i in 1:N, j in 1:N
        S = State(Bgrid[i],Bgrid[j], Sdummy.alpha, Sdummy.upsilon, Sdummy.tau, Sdummy.Q)
        a = nash_equilibrium(S)
        q1grid[i,j] = a.q1
        q2grid[i,j] = a.q2
        I1grid[i,j] = a.I1
        I2grid[i,j] = a.I2
    end
    return Bgrid,q1grid,q2grid,I1grid,I2grid
end

find_closest(Bgrid,B) = argmin(abs.(Bgrid .- B))


##############################################################
# SHOCKS
##############################################################
alphas = [12.0,13.0,15.0]
upsilons = [3.0,2.0,1.0]
taus = [1.0,1,1.0]
probs = [0.6,0.25,0.15]

function draw_shock()
    s = rand(Categorical(probs))
    return alphas[s], upsilons[s], taus[s]
end


##############################################################
# SIMULATION
##############################################################
function simulate(T, B1_init, B2_init, Bgrid,q1grid,q2grid,I1grid,I2grid)
    B1 = zeros(T); B2 = zeros(T)
    q1 = zeros(T); q2 = zeros(T)
    I1 = zeros(T); I2 = zeros(T)
    p1 = zeros(T); p2 = zeros(T)

    B1[1] = B1_init
    B2[1] = B2_init

    for t in 1:T
        α, υ, τ = draw_shock()
        Sshock = State(B1[t],B2[t],α,υ,τ,2.0)

        idx1 = find_closest(Bgrid,B1[t])
        idx2 = find_closest(Bgrid,B2[t])

        q1[t] = q1grid[idx1,idx2]
        q2[t] = q2grid[idx1,idx2]
        I1[t] = I1grid[idx1,idx2]
        I2[t] = I2grid[idx1,idx2]

        p1[t], p2[t] = prices(Sshock,Action(q1[t],I1[t],q2[t],I2[t]))

        if t < T
            B1[t+1] = clamp(brand_transition(B1[t],I1[t],1),0,1)
            B2[t+1] = clamp(brand_transition(B2[t],I2[t],2),0,1)
        end
    end

    return B1,B2,q1,q2,I1,I2,p1,p2
end


##############################################################
# PLOT
##############################################################
##############################################################
# PLOT: BRAND DYNAMICS ONLY
##############################################################
function plot_brand_dynamics(T, B1, B2)
    t = 1:T
    plt = plot(t, B1, lw=3, color=:blue, label="Brand 1",
        xlabel="Time", ylabel="Brand Level",
        title="Brand Dynamics Over Time",
        legend=:topright)
    plot!(plt, t, B2, lw=3, color=:red, label="Brand 2")

    display(plt)
end


##############################################################
# MAIN
##############################################################
Sdummy = State(0.5,0.5,12.0,3.0,1.0,2.0)
Bgrid,q1grid,q2grid,I1grid,I2grid = build_policy_grids(Sdummy)

T = 80
B1,B2,q1,q2,I1,I2,p1,p2 = simulate(T,0.75,0.25,Bgrid,q1grid,q2grid,I1grid,I2grid)

plot_paths(T,B1,B2,q1,q2,I1,I2,p1,p2)
