using LinearAlgebra
using Statistics
using DataFramesMeta
using StatsPlots
using CSV
using Turing
using Bijectors
using FillArrays
using Random

rng = Xoshiro(123)
isdir("plots") || mkdir("plots")

df = DataFrame(CSV.File("data/scores.csv"))

judge_map = Dict(j => i for (i, j) in enumerate(unique(df.judge)))
df.judge_index .= getindex.(Ref(judge_map), df.judge)

j = df.judge_index
J = length(unique(j))
b = df.beer
B = length(unique(b))
μs, σs = mean(df.score), std(df.score)
s = (df.score .- μs) ./ σs
judge_colors = reshape(1:J, 1, :)
judges = reshape(sort(unique(df.judge)), 1, :)
judge_offsets = [0.1, 0.05, 0, -0.05]

let
    plt = plot(
        title="Beer scores",
        xlabel="Score",
        ylabel="Beer",
        xlims=(-0.1, 5)        
    )

    for (key, group) in pairs(groupby(df, :beer))
        scatter!(
            plt,
            [mean(group.score)],
            [key.beer],
            color=:black,
            markerstyle=:cross,
            label=key.beer == 1 ? "Empirical mean" : nothing,
            legend=:right
        )
    end

    for (key, group) in pairs(groupby(df, :judge))
        @df group scatter!(
            plt,
            :score .+ randn(rng, size(group,1))*0.05,
            :beer .+ randn(rng, size(group,1))*0.05,
            label="Judge $(key.judge)",
            color=judge_map[key.judge]
        )
    end

    savefig(plt, joinpath("plots", "scores_scatterplot.png"))
    plt
end

@df df boxplot(:judge, :score, title="Judge scores", xlabel="Judge", ylabel="Score", label=nothing)

score_df = combine(groupby(df, :beer), :score => mean => :score_empirical) 

#= 
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
=#

@model function item_response_normal(b, j, y, J, B)
    αμ ~ Normal(0, 3)
    ασ ~ truncated(Normal(0, 3), lower=0)
    
    γμ ~ truncated(Normal(0, 3), lower=0)
    γσ ~ truncated(Normal(0, 3), lower=0)
    
    αz ~ filldist(Normal(0, 1), J)
    α = αμ .+ αz .* ασ # filldist(Normal(0, 1) αμ, ασ), J)

    γz ~ filldist(truncated(Normal(0, 1), lower=0), J)
    γ = γμ .+ γz .* γσ # filldist(truncated(Normal(γμ, γσ), lower=0), J)

    β ~ filldist(Normal(0, 1), B)
    μ = γ[j].*(β[b] .- α[j])
    σ ~ Exponential(0.5)
    y .~ Normal.(μ, σ)
    return (;α, β, γ)
end

model = item_response_normal(b, j, s, J, B)
prior_samples = sample(model, Prior(), 50)
post_samples = sample(Xoshiro(123), model, NUTS(), MCMCThreads(), 2000, 8; progress=false)
plot(post_samples)

gq = generated_quantities(model, Turing.MCMCChains.get_sections(post_samples, :parameters))
αs = Base.stack(x -> x.α, gq, dims=1) .* σs
γs = Base.stack(x -> x.γ, gq, dims=1) .* σs
βs = Array(group(post_samples, :β)) .* σs .+ μs
score_df.score_posterior .= vec(mean(βs, dims=1))

let
    plt = plot(
        title="Posterior beer quality",
        ylabel="Beer",
        xlabel="Quality",
        yticks=0:1:B,
    )

    for (i, x) in enumerate(eachcol(βs))
        qs = quantile(x, [0.025, 0.985])
        m = mean(x)
        scatter!(
            plt, [m], [i],
            label=i == 1 ? "Posterior mean" : nothing,
            xerror=[(m - qs[1], qs[2] - m)],
            color=:grey,
            markersize=7,
            markercolor=:white,
            markerstrokealpha=1,
            linewidth=2,
        )
        
        beer_df = @rsubset(df, :beer == i)
        
        @df combine(groupby(beer_df, :beer), :score => mean) scatter!(
            plt,
            :score_mean,
            [i], 
            color=:black,
            markerstyle=:cross,
            label=i == 1 ? "Empirical mean" : nothing,
        )

        @df beer_df scatter!(
            plt,
            :score,
            fill(i, J) .+ judge_offsets,
            group=:judge,
            color=judge_colors,
            label=i == 1 ? reshape("Judge " .* judges, 1, :) : nothing,
            legendposition=(0.8, 0.75)
        )
    end
    savefig(plt, joinpath("plots", "posterior_beer_quality.png"))
    plt
end

# We see beer 1, 3, and 7 get even higher posterior means! Why is this?

# explained by the judges A and C giving (somewhat favourable) scores.
# since their harshness is high and discrimination low, a high score from them means more than from e.g. judge B 
# who has a low harshness and high discrimination.
# It also means more than an average weight which they get in the empirical mean.

function plot_judge_posteriors(harshness, discrimination, judges, colors)
    J = size(discrimination, 2)
    plt_harshness = density(
        harshness,
        title="Posterior harshness",
        xlabel="Harshness",
        label="Judge " .* judges,
        linewidth=3
    )
    scatter!(
        plt_harshness,
        mean(harshness, dims=1),
        zeros(J),
        markershape=:utriangle,
        color=colors,
        label=hcat("Mean", fill(nothing, 1, J-1))
    )

    plt_discrimination = density(
        discrimination,
        title="Posterior discrimination",
        xlabel="discrimination",
        label=nothing,
        linewidth=3
    )    
    scatter!(
        plt_discrimination,
        mean(discrimination, dims=1),
        zeros(J),
        markershape=:utriangle,
        color=colors,
        label=nothing
    )

    plot(
        plt_harshness,
        plt_discrimination,
        layout=(2, 1),
        ylabel="Density",
        size=(800, 600)
    )
end

let 
    plt = plot_judge_posteriors(αs, γs, judges, judge_colors)
    savefig("plots/judge_posteriors.png")
    plt
end

#= let 
    plt = plot()
    density!(plt, βs, label=reshape(["Beer $i" for i in b], 1, :))
    scatter!(
        plt,
        vec(mean(βs, dims=1)),
        zeros(B),
        label=reshape(vcat("Posterior mean", fill(nothing, B-1)), 1, :),
        color=1:B,
    )
    
    @df score_df scatter!(
        plt,
        :score_mean,
        zeros(B),
        group=:beer,
        label=reshape(vcat("Empirical mean", fill(nothing, B-1)), 1, :),
        color=1:B,
        marker=:utriangle,
    )
end
=#

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
score_df.score_ordcat .= vec(mean(cat_βs, dims=1))
sort(score_df, :score_ordcat) == sort(score_df, :score_mean)

