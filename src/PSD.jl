__precompile__()
module PSD

import Base: mean

using SymPy
# using DataFrames

# %% General distribution
"""
Abstract type for all distributions
"""
abstract Distribution

abstract UnboundDistribution <: Distribution
abstract BoundDistribution <: Distribution

"""
Probability density function
"""
function pdf(d :: Distribution, x)
	z = symbols("z")
    subs(diff(cdf(d, z), z), z => x)
end

"""
Integrate `f(x) p(x)` expression
"""
function integrate_distribution(f, d :: Distribution; numerical = false, xmin = -oo, xmax = oo, assumption = false)
	if numerical
		quadgk(z -> f(z)*pdf(d, z), xmin, xmax) |> first
	else
		x = assumption ? symbols("x"; (assumption, true)) : symbols("x")
		integrate(f(x)*pdf(d,x), (x, xmin, xmax))
	end
end

"""
Mathematical expectation, mean value, of the distribution

Keyword arguements:
- `numerical` (`false`) use numerical integration?
- `xmin = -oo` - minimum value of x
- `xmax = oo` - maximum value of x
- `assumption = false` - add any assumptions about x (e.g., `:positive`)
"""
function mean(d :: Distribution; kv...)
	integrate_distribution(x->x, d; kv...)
end

"""
Expectation of ξ². Sometimes it is more efficient to calculate just this
without calculating the full variance (that requires also mean of ξ)
"""
function mean_of_square(d :: Distribution; kv...)
	integrate_distribution(x->x^2, d; kv...)
end

"""
Variance of the distribution
"""
function variance(d :: Distribution; kv...)
	x2 = mean_of_square(d; kv...)
    me = mean(d; kv...)
    x2 - me^2
end

function relvar(d :: Distribution; kv...)
	x2 = mean_of_square(d; kv...)
	me = mean(d; kv...)
	x2 / me^2 - (me/me)
end

"""
Returns coefficient of determination R² defined as

R² = 1 - SS(res)/SS(tot)

where

SS(tot) = ∑ (y(i)-E[y])²
SS(res) = ∑(f(i)-y(i))²
"""
function r2{T}(data :: Vector{T}, approximation :: Vector{T})
    data_mean = mean(data)
    ss_tot = sum((y - data_mean)^2 for y ∈ data)
    ss_res = sum((data[i]-approximation[i])^2 for i=1:length(data))
    one(T) - ss_res/ss_tot
end

function r2{T}(y :: Vector{T}, x :: Vector{T}, f :: Function)
    y_mean = mean(y)
    ss_tot = sum((η - y_mean)^2 for η ∈ y)
    ss_res = sum((y[i]-f(x[i]))^2 for i=1:length(y))
    one(T) - ss_res/ss_tot
end


# %% GGS

"""
Gates-Gaudin-Schuhmann distribution

P(x) = (x/xmax)^α
"""
type GGS <: BoundDistribution
    α
    xmax
end

# All properties can be expressed directly:
cdf(d :: GGS, x) = (x / d.xmax)^d.α
pdf(d :: GGS, x) = d.α / d.xmax * (x / d.xmax)^(d.α-1)
mean(d :: GGS) = d.α / (d.α + 1) * d.xmax
variance(d :: GGS) = d.α/((d.α+1)^2 * (d.α+2))*d.xmax^2
relvar(d :: GGS) = 1/(d.α*(d.α+2))

"""
Fits data representing a CFD to a particular
distribution. Returns a pair (D, R2) where
D is a distribution object and R2 is R² coefficient
of determinition of the fit.
"""
function fit_cdf(::Type{Val{GGS}}, x, y)
    xmax = x[end]
    xw = log.(x / xmax)
    yw = log.(y)

    a, λ = linreg(xw,yw)
    f = z -> λ*z
    (GGS(λ,xmax), r2(yw,xw,f))
end

# %%
# %% Harris
"""
Harris distribution
"""
type Harris <: PSD.BoundDistribution
    n
    s
    xmax
end

cdf(d :: Harris, x) = 1-(1-(x/d.xmax)^d.s)^d.n

function pdf(d :: Harris, x)
	ξ = x / d.xmax
	ξs = ξ^(d.s-1)
	d.s*d.n/d.xmax*ξs*(1-ξs*ξ)^(d.n-1)
end

mean(d ::Harris) = d.n * d.xmax / (1+d.n*d.s)*beta(1/d.s, d.n)

function variance(d :: Harris)
	me = mean(d)
	(2*d.n*d.xmax^2)/(2+d.n*d.s)*beta(2/d.s,d.n)-me^2
end

function relvar(d :: Harris)
	(2*(1+d.n*d.s)^2)/(d.n*(2+d.n*d.s)) * beta(2/d.s,d.n)/(beta(1/d.s,d.n))^2-1
end

function fit_cdf(::Type{Val{Harris}}, x, y)
    xmax = x[end]
    ξ = x / xmax
    # Stage 1: small ξ ⇒ get s
    ξ_small = filter(u -> u < 0.1, ξ)
    lnξ_small = log.(ξ_small)
    yw_small = [log(log(1.0/(1.0-y[i]))) for i=1:length(ξ_small)]
    (lnn, s) = linreg(lnξ_small, yw_small)
    r2s = r2(yw_small, lnξ_small, z -> s*z + lnn)
    # Stage 2: get n from all the data points
    # except for last where y=1
    arg = log.(1.0 .- ξ[1:end-1] .^ s)
    val = log.(1.0 .- y[1:end-1])
    (_,n) = linreg(arg,val)
    r2n = r2(val, arg, z -> n*z)
    (Harris(n,s,xmax), (r2n,r2s))
end

# %% Log-normal
"""
Log-normal distribution
"""
type LogNormal <: UnboundDistribution
	μ
	σ
end

cdf(d :: LogNormal, x) = 0.5*(1+erf((log(x)-d.μ)/(√2*d.σ)))
pdf(d :: LogNormal, x) = 1/(x * d.σ*√(2π))*exp(-0.5*((log(x)-d.μ)/d.σ)^2)
mean(d :: LogNormal) = exp(d.μ+(d.σ)^2/2)
variance(d :: LogNormal) = (exp((d.σ)^2)-1)*(mean(d))^2
relvar(d :: LogNormal) = exp((d.σ)^2)-1

function fit_cdf(::Type{Val{LogNormal}}, x, y)
    h = erfinv.(2y-1)
    lnx = log.(x)
    (l,k) = linreg(h,lnx)
    σ = k / sqrt(2.0)
    μ = l
    (LogNormal(μ,σ), r2(lnx, h, z -> k*z+l))
end

# %% Rosin-Rammler

"""
Rosin-Rammler distribution
"""
type RosinRammler <: UnboundDistribution
    α
    x632
end

cdf(d :: RosinRammler, x) = 1 - exp(-(x/d.x632)^d.α)
pdf(d :: RosinRammler, x) = d.α/x * (x/d.x632)^d.α * exp(-(x/d.x632)^d.α)
mean(d :: RosinRammler) = d.x632 * gamma(1+1/d.α)
variance(d :: RosinRammler) = d.x632^2*(gamma(1+2/d.α)-(gamma(1+1/d.α))^2)
relvar(d :: RosinRammler) = gamma(1+2/d.α)/(gamma(1+1/d.α))^2-1

function fit_cdf(::Type{Val{RosinRammler}}, x, y)
    xw = log.(x)
    yw = log.(log.(1.0 ./ (1.0 .- y)))
    (l,k) = linreg(yw, xw)
    α = 1.0/k
    x632 = exp(l)
    (RosinRammler(α,x632), r2(xw, yw, z -> k*z+l))
end

# %% Truncation
"""
Truncate: forward transformation: from 0 < x < ∞ to 0 < y < ymax
"""
truncf(ymax, x) = ymax*x/(ymax+x)
"""
Truncate: backward transformation: from 0 < y < ymax to 0 < x < ∞
"""
truncb(ymax, y) = ymax*y /(ymax - y)

"""
Derivative of truncate backward
"""
dtruncb(ymax, y) = (ymax/(y-ymax))^2

abstract AbstractTruncatedDistribution <: BoundDistribution

type TruncatedDistribution{T <: UnboundDistribution} <: AbstractTruncatedDistribution
    d :: T
    xmax
end

cdf{T <: UnboundDistribution}(d::TruncatedDistribution{T}, x) =
	cdf(d.d, truncb(d.xmax, x))
pdf{T <: UnboundDistribution}(d :: TruncatedDistribution{T}, x) =
	pdf(d.d, truncb(d.xmax, x))*dtruncb(d.xmax, x)

function integrate_distribution(f, d :: TruncatedDistribution; numerical = true, xmin = 0.0, xmax = d.xmax, assumotions = false)
	integrate_distribution(f, d.d, xmin = xmin, xmax = xmax, numerical = numerical, assumptions = assumptions)
end

function mean(d :: TruncatedDistribution; kv...)
	integrate_distribution(x->x, d; kv...)
end

function mean_of_square(d :: TruncatedDistribution; kv...)
	integrate_distribution(x->x^2, d; kv...)
end

function variance(d :: TruncatedDistribution; kv...)
	E = mean(d; kv...)
	E2 = mean_of_square(d; kv...)
	E2 - E^2
end

function relvar(d :: TruncatedDistribution; kv...)
	E = mean(d; kv...)
	E2 = mean_of_square(d; kv...)
	E2/E^2 - 1.0
end

function fit_cdf{T <: UnboundDistribution}(::Type{Val{TruncatedDistribution{T}}}, x, y)
    xmax = x[end]
    xw = truncb.(xmax, x[1:end-1])
    d, r2d = fit_cdf(Val{T}, xw, y[1:end-1])
    (TruncatedDistribution(d, xmax), r2d)
end

# %% Truncated Rosin-Rammler

# Need to create the type to specify p() and P() functions explicitly
"""
Truncated Rosin-Rammler
"""
type TRR <: BoundDistribution
    d :: TruncatedDistribution{RosinRammler}
    TRR(α, x632, xmax) =
		new(TruncatedDistribution(RosinRammler(α, truncb(xmax, x632)), xmax))
end

cdf(d :: TRR, x) = 1 - exp(-(x*d.d.xmax / (d.d.d.x632*(d.d.xmax - x)))^d.d.d.α)

function pdf(d :: TRR, x)
    td = d.d # truncated distribution
    rr = td.d
    α = rr.α
    x632 = rr.x632
    xmax = td.xmax
    tmp = (x*xmax/(x632*(xmax-x)))^α
    α*xmax*tmp / (x*(xmax-x)) * exp(-tmp)
end

# These only work numerically
mean(d :: TRR) = quadgk(x -> x * pdf(d, x), 0.0, d.d.xmax) |> first
variance(d :: TRR) = (quadgk(x->x^2 * pdf(d, x), 0.0, d.d.xmax) |> first) - (mean(d))^2

function fit_cdf(::Type{Val{TRR}}, x, y)
    d, r2d = fit_cdf(Val{TruncatedDistribution{RosinRammler}}, x, y)
    (TRR(d.d.α, truncf(d.xmax, d.d.x632), d.xmax), r2d)
end

# %% Jumped (discontinuous) distribution
type JumpedDistribution{T <: Distribution} <: Distribution
	d :: T
	xmin
end

cdf(d :: JumpedDistribution, x) = cdf(d.d,x)*Heaviside(x-d.xmin)

function pdf(d :: JumpedDistribution, x)
	# No δ(x)-part as it cannot be hanled numerically
	pdf(d.d,x)*Heaviside(x-d.xmin)+cdf(d.d,x)*DiracDelta(x-d.xmin)
end

function integrate_distribution(f, d :: JumpedDistribution; xmin = -oo, kv...)
	if xmin > d.xmin
		integrate_distribution(f, d.d; xmin = xmin, kv...)
	else
		integrate_distribution(f, d.d; xmin = d.xmin, kv...) + f(d.xmin)*cdf(d.d, d.xmin)
	end
end

# %% Characteristics of the distribution

function particles_number(φ, d :: Distribution; numerical = false, kv...)
	(1-φ) * 6 / (numerical? pi : PI) *
	integrate_distribution(x->x^(-3), d; numerical = numerical, kv...)
end

function particles_area(φ, d :: Distribution; kv...)
	6(1-φ) * integrate_distribution(x -> x^(-1), d; kv...)
end

# %%

end # module
