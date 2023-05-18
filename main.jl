using LinearAlgebra
using Statistics
using DataFramesMeta
using StatsPlots
using CSV
using Turing
using Bijectors
using FillArrays


df = DataFrame(CSV.File("scores.csv"))

let
    plt = plot(
        title="Beer scores",
        xlabel="Score",
        ylabel="Beer",
        xlims=(-0.1, 5)        
    )
    for (key, group) in pairs(groupby(df, :judge))
        @df group scatter!(plt, :score .+ randn(size(group,1))*0.05, :beer .+ randn(size(group,1))*0.05, label="Judge $(key.judge)")
    end
    plt
end

@df df boxplot(:judge, :score, title="Judge scores", xlabel="Judge", ylabel="Score", label=nothing)

mean_score_df = combine(groupby(df, :beer), :score .=> [mean, std]) 

@model function item_response_normal_old(beer, judge, score, J, B)
    α ~ filldist(Normal(0, 1), J)
    β ~ filldist(Normal(0, 1), B)
    μ = α[judge] .+ β[beer]
    σ ~ Exponential(1)
    score .~ Normal.(μ, σ)
end

@model function item_response_normal_harshness_discrimination(b, j, score, J, B)
    α ~ filldist(Normal(0, 1), J)
    β ~ filldist(Normal(0, 1), B)
    γ ~ filldist(Normal(0, 1), J)
    μ = γ[j].*(β[b] .- α[j])
    σ ~ Exponential(1)
    score .~ Normal.(μ, σ)
end

@model function item_response_normal(b, j, score, J, B)
    αμ ~ Normal(0, 2)
    ασ ~ truncated(Normal(0, 1), lower=0)
    
    γμ ~ truncated(Normal(0, 1), lower=0)
    γσ ~ truncated(Normal(0, 1), lower=0)
        
    α ~ filldist(Normal(αμ, ασ), J)
    β ~ filldist(Normal(0, 1), B)    
    γ ~ filldist(Normal(γμ, γσ), J)

    μ = γ[j].*(β[b] .- α[j])
    σ ~ Exponential(1)
    score .~ Normal.(μ, σ)
end

judge_map = Dict(j => i for (i, j) in enumerate(unique(df.judge)))
df.judge_index .= getindex.(Ref(judge_map), df.judge)

j = df.judge_index
J = length(unique(j))
b = df.beer
B = length(unique(b))
μs, σs = mean(df.score), std(df.score)
s = (df.score .- μs) ./ σs

model = item_response_normal(b, j, s, J, B)
prior_samples = sample(model, Prior(), 50)
post_samples = sample(model, NUTS(), MCMCThreads(), 1000, 4; progress=false)
plot(post_samples)

αs = Array(group(post_samples, :α)) .* σs
vec(mean(αs, dims=1))
βs = Array(group(post_samples, :β)) .* σs .+ μs
mean_score_df.score_normal .= vec(mean(βs, dims=1))
mean_score_df

group(post_samples, :β)

let 
    plt = plot()
    density!(plt, βs, label=reshape(["Beer $i" for i in b], 1, :))
    scatter!(
        plt,
        vec(mean(βs, dims=1)),
        zeros(B),
        label=reshape(vcat("Posterior mean", fill(nothing, B-1)), 1, :),
        color=1:B,
    )
    
    @df mean_score_df scatter!(
        plt,
        :score_mean,
        zeros(B),
        group=:beer,
        label=reshape(vcat("Empirical mean", fill(nothing, B-1)), 1, :),
        color=1:B,
        marker=:utriangle,
    )
end

@model function item_response_ordered_categories(beer, judge, score, J, B, C)
    α ~ filldist(Normal(0, 1), J)
    β ~ filldist(Normal(0, 1), B)
    cutpoints ~ Bijectors.ordered(filldist(Normal(0.5, 1), C-1))
    μ = α[judge] .+ β[beer]
    score .~ OrderedLogistic.(μ, Ref(cutpoints))
end

s_map = sort(Dict(s => i for (i, s) in enumerate(unique(sort(df.score)))))
s_cat = getindex.(Ref(s_map), df.score)
C = length(unique(s_cat))
ordcat_model = item_response_ordered_categories(b, j, s_cat, J, B, C)
ordcat_post_samples = sample(ordcat_model, NUTS(), MCMCThreads(), 1000, 4; progress=false)

logodds2prob(x) = exp(x) / (1 + exp(x))
@chain quantile(ordcat_post_samples) begin
    DataFrame
    select(_, :parameters, names(_, r"%") .=> ByRow(logodds2prob); renamecols=false)
end

cutpoints = logodds2prob.(Array(group(ordcat_post_samples, :cutpoints))) #  .* maximum(keys(s_map))
density(cutpoints, labels=reshape(collect(keys(s_map)), 1, :))

let 
    plt = plot(
        title="Posterior cut points",
        ylabel="Cut point index",
        xlabel="Probability",
        yticks=0:1:C, 
        xticks=0:0.1:1
    )
    for (i, (cp, x)) in enumerate(zip((keys(s_map)), eachcol(cutpoints)))
        qs = quantile(x, [0.025, 0.985])
        m = mean(x)
        scatter!(plt, [m], [i], label=cp, xerror=[(m - qs[1], qs[2] - m)])
    end
    plt
end

cat_βs = logodds2prob.(Array(group(ordcat_post_samples, :β))) .* maximum(keys(s_map))
mean_score_df.score_ordcat .= vec(mean(cat_βs, dims=1))
sort(mean_score_df, :score_ordcat) == sort(mean_score_df, :score_mean)

