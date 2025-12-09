##############################################################
# PACKAGES
##############################################################
using Random, LinearAlgebra, Distributions, Optim, Plots

cd("D:/Fall-2025-Courses/IO/Research-Proposal")



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
psi(tau,q1,Q) = tau * log(1 + (q1/Q)) / log(2)
rho(I; ϵ=1e-3) = (log(I + ϵ) - log(ϵ)) / (log(1 + ϵ) - log(ϵ))

function brand_transition(B,I,i)
    persist = (i==1 ? 0.3 : 0.1)
    inv_eff = (i==1 ? 0.35 : 0.1)
    return persist * rho(I) * B + inv_eff * I
end

function prices(S::State, a::Action)
    ψ = psi(S.tau, a.q1, S.Q)
    p1 = S.alpha - S.upsilon*a.q1 + ψ*(2*S.B1 - 0.1*S.B2)
    p2 = S.alpha - S.upsilon*a.q2 + (1-ψ)*(1.0*S.B2 - 1.0*S.B1)
    return p1,p2
end

function profit_i(i::Int, S::State, a::Action)
    p1,p2 = prices(S,a)
    ψ = psi(S.tau, a.q1, S.Q)
    if i==1
        return p1*a.q1 - a.q1^2 - 0.9*(a.I1^2)/(ψ+1e-6)
    else
        return p2*a.q2 - a.q2^2 - 1*(a.I2^2)/((1-ψ)+1e-6)
    end
end

##############################################################
# BEST RESPONSE & NASH
##############################################################
function best_response(i::Int, S::State, a_op::Action)
    obj(x) = begin
        q,I = clamp.(x,0.0,1.0)
        a = i==1 ? Action(q,I,a_op.q2,a_op.I2) : Action(a_op.q1,a_op.I1,q,I)
        -profit_i(i,S,a)
    end
    res = optimize(obj,[0.4,0.4],[0.0,0.0],[1.0,1.0],NelderMead())
    return clamp.(Optim.minimizer(res),0,1)
end

function nash_equilibrium(S::State)
    a = Action(0.5,0.5,0.5,0.5)
    for _ in 1:8
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
# SHOCKS & MARKOV MATRIX
##############################################################
alphas = [100, 110, 90]
upsilons = [1.0, 1.0, 1.0]
taus     = [0.5, 1.0, 0.1]

P = [
    0.7 0.2 0.1;
    0.10 0.60 0.3;
    0.25 0 0.75
]

function draw_next_shock(current_s)
    w = P[current_s, :]
    return rand(Categorical(w))
end

##############################################################
# GRID & POLICY
##############################################################
create_grid(N=15) = collect(range(0.0,1.0,length=N))

function build_policy_for_shock(α, υ, τ)
    N = 15
    Bgrid = create_grid(N)
    q1grid = zeros(N,N)
    q2grid = zeros(N,N)
    I1grid = zeros(N,N)
    I2grid = zeros(N,N)
    for i in 1:N, j in 1:N
        S = State(Bgrid[i], Bgrid[j], α, υ, τ, 2.0)
        a = nash_equilibrium(S)
        q1grid[i,j] = a.q1
        q2grid[i,j] = a.q2
        I1grid[i,j] = a.I1
        I2grid[i,j] = a.I2
    end
    return (Bgrid,q1grid,q2grid,I1grid,I2grid)
end

shock_policies = [build_policy_for_shock(alphas[s], upsilons[s], taus[s]) for s in 1:3]

##############################################################
# SIMULATION WITH INITIAL ASYMMETRIC Q
##############################################################
function simulate(T, B1_init, B2_init, q1_init, q2_init, shock_policies)
    B1 = zeros(T); B2 = zeros(T)
    q1 = zeros(T); q2 = zeros(T)
    I1 = zeros(T); I2 = zeros(T)
    p1 = zeros(T); p2 = zeros(T)
    shock = zeros(Int,T)

    B1[1] = B1_init; B2[1] = B2_init
    q1[1] = q1_init; q2[1] = q2_init
    shock[1] = 1

    for t in 1:T
        s = shock[t]
        α = alphas[s]; υ = upsilons[s]; τ = taus[s]
        Sshock = State(B1[t], B2[t], α, υ, τ, 2.0)
        Bgrid,q1grid,q2grid,I1grid,I2grid = shock_policies[s]

        if t > 1
            idx1 = argmin(abs.(Bgrid .- B1[t]))
            idx2 = argmin(abs.(Bgrid .- B2[t]))
            q1[t] = q1grid[idx1, idx2]
            q2[t] = q2grid[idx1, idx2]
        end

        idx1 = argmin(abs.(Bgrid .- B1[t]))
        idx2 = argmin(abs.(Bgrid .- B2[t]))
        I1[t] = I1grid[idx1, idx2]
        I2[t] = I2grid[idx1, idx2]

        p1[t], p2[t] = prices(Sshock, Action(q1[t],I1[t],q2[t],I2[t]))

        if t < T
            B1[t+1] = clamp(brand_transition(B1[t], I1[t], 1), 0, 1)
            B2[t+1] = clamp(brand_transition(B2[t], I2[t], 2), 0, 1)
            shock[t+1] = draw_next_shock(s)
        end
    end

    return B1,B2,q1,q2,I1,I2,p1,p2,shock
end

##############################################################
# PLOT: BRAND, PRICES, MARKET SHARE
##############################################################
function plot_BP(T, B1, B2, p1, p2; filename="brands_prices.pdf")
    t = 1:T
    plt = plot(layout=(1,2), size=(1200,500))  # 1 row, 2 columns

    # --- Brands subplot ---
    plot!(plt[1], t, B1, lw=3, label="Brand1", color=:blue)
    plot!(plt[1], t, B2, lw=3, label="Brand2", color=:red)
    title!(plt[1], "Brand Images")
    xlabel!(plt[1], "Time")
    ylabel!(plt[1], "Brand")

    # --- Prices subplot ---
    plot!(plt[2], t, p1, lw=3, label="Price1", color=:green)
    plot!(plt[2], t, p2, lw=3, label="Price2", color=:orange)
    title!(plt[2], "Prices")
    xlabel!(plt[2], "Time")
    ylabel!(plt[2], "Price")

    # Display and save as PDF
    display(plt)
    savefig(plt, filename)
end

# Call it
T = 20
B1,B2,q1,q2,I1,I2,p1,p2,shock =
    simulate(T, 0.75, 0.25, 1.0, 0.3, shock_policies)

plot_BP(T, B1, B2, p1, p2; filename="brands_prices.pdf")



##############################################################
# MAIN
##############################################################







function plot_BR(T, B1, B2; filename="brands-diff.pdf")
    t = 1:T
    plt = plot(layout=(1,1), size=(1200,500))  # 1 row, 2 columns

    # --- Brands subplot ---
    plot!(plt[1], t, B1-B2, lw=3, label="", color=:blue)
    title!(plt[1], "Relative Brand Power: LVMH")
    xlabel!(plt[1], "Time")
    ylabel!(plt[1], "Relative Brand Power: LVMH")

    

    # Display and save as PDF
    display(plt)
    savefig(plt, filename)
end



T = 20
B1,B2,q1,q2,I1,I2,p1,p2,shock =
    simulate(T, 0.75, 0.25, 1.0, 0.3, shock_policies)



plot_BR(T, B1, B2; filename="brands-diff.pdf")

