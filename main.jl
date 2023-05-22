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

@model function judge_model(b, j, s, J, B)
    αμ ~ Normal(0, 3)
    ασ ~ truncated(Normal(0, 3), lower=0)
    
    γμ ~ truncated(Normal(0, 3), lower=0)
    γσ ~ truncated(Normal(0, 3), lower=0)
    
    αz ~ filldist(Normal(0, 1), J)
    α = αμ .+ αz .* ασ

    γz ~ filldist(truncated(Normal(0, 1), lower=0), J)
    γ = γμ .+ γz .* γσ

    β ~ filldist(Normal(0, 1), B)
    μ = γ[j].*(β[b] .- α[j])
    σ ~ Exponential(0.5)
    s .~ Normal.(μ, σ)
    return (;α, β, γ)
end

model = judge_model(b, j, s, J, B)
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