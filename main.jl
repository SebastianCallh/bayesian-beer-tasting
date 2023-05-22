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
figsize = (800, 600)
df = DataFrame(CSV.File("data/scores.csv"))

judge_map = Dict(j => i for (i, j) in enumerate(unique(df.judge)))
df.judge_index .= getindex.(Ref(judge_map), df.judge)

j = df.judge_index
N = length(unique(j))
i = df.beer
M = length(unique(i))
μs, σs = mean(df.score), std(df.score)
S = (df.score .- μs) ./ σs
judge_colors = reshape(1:N, 1, :)
judges = reshape(sort(unique(df.judge)), 1, :)
judge_offsets = [0.1, 0.05, 0, -0.05]

let
    plt = plot(
        title="Beer scores",
        xlabel="Score",
        ylabel="Beer",
        xlims=(-0.1, 5),
        size=figsize
    )

    for (key, group) in pairs(groupby(df, :beer))
        scatter!(
            plt,
            [mean(group.score)],
            [key.beer],
            color=:black,
            markerstyle=:cross,
            label=key.beer == 1 ? "Empirical mean" : nothing,
            legend=:right,
            markersize=8
        )
    end

    for (key, group) in pairs(groupby(df, :judge))
        @df group scatter!(
            plt,
            :score,
            :beer .+ randn(rng, size(group,1))*0.1,
            label="Judge $(key.judge)",
            color=judge_map[key.judge],
            markersize=6
        )
    end

    savefig(plt, joinpath("plots", "scores_scatterplot.png"))
    plt
end

@df df boxplot(:judge, :score, title="Judge scores", xlabel="Judge", ylabel="Score", label=nothing)

score_df = combine(groupby(df, :beer), :score => mean => :score_empirical)

@model function judge_model(i, j, S, M, N)
    Hμ ~ Normal(0, 3)
    Hσ ~ truncated(Normal(0, 3), lower=0)
    
    Dμ ~ truncated(Normal(0, 3), lower=0)
    Dσ ~ truncated(Normal(0, 3), lower=0)
    
    Hz ~ filldist(Normal(0, 1), M)
    H = Hμ .+ Hz .* Hσ

    Dz ~ filldist(truncated(Normal(0, 1), lower=0), M)
    D = Dμ .+ Dz .* Dσ

    Q ~ filldist(Normal(0, 1), N)
    σ ~ Exponential(0.5)

    μ = D[j].*(Q[i] .- H[j])    
    S .~ Normal.(μ, σ)
    return (;H, D)
end

model = judge_model(i, j, S, N, M)
prior_samples = sample(model, Prior(), 50)
post_samples = sample(Xoshiro(123), model, NUTS(), MCMCThreads(), 2000, 8; progress=false)
plot(post_samples)

gq = generated_quantities(model, Turing.MCMCChains.get_sections(post_samples, :parameters))
Hs = Base.stack(x -> x.H, gq, dims=1) .* σs .+ μs
Ds = Base.stack(x -> x.D, gq, dims=1) .* σs
Qs = Array(group(post_samples, :Q)) .* σs .+ μs
score_df.score_posterior .= vec(mean(Qs, dims=1))

let
    plt = plot(
        title="Posterior beer quality",
        ylabel="Beer",
        xlabel="Quality",
        yticks=0:1:M,
        size=figsize
    )

    for (i, x) in enumerate(eachcol(Qs))
        qs = quantile(x, [0.025, 0.985])
        m = mean(x)
        scatter!(
            plt, [m], [i],
            label=i == 1 ? "Posterior mean" : nothing,
            xerror=[(m - qs[1], qs[2] - m)],
            color=:grey,
            markersize=10,
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
            markersize=8,
            label=i == 1 ? "Empirical mean" : nothing,
        )

        @df beer_df scatter!(
            plt,
            :score,
            fill(i, N) .+ judge_offsets,
            group=:judge,
            color=judge_colors,
            label=i == 1 ? reshape("Judge " .* judges, 1, :) : nothing,
            legendposition=(0.8, 0.75),
            markersize=6,
        )
    end
    savefig(plt, joinpath("plots", "posterior_beer_quality.png"))
    plt
end

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
        label=hcat("Mean", fill(nothing, 1, J-1)),
        markersize=6
    )

    plt_discrimination = density(
        discrimination,
        title="Posterior discrimination",
        xlabel="Discrimination",
        label=nothing,
        linewidth=3
    )    
    scatter!(
        plt_discrimination,
        mean(discrimination, dims=1),
        zeros(J),
        markershape=:utriangle,
        color=colors,
        label=nothing,
        markersize=6
    )

    plot(
        plt_harshness,
        plt_discrimination,
        layout=(2, 1),
        ylabel="Density",
        size=figsize
    )
end

let 
    plt = plot_judge_posteriors(Hs, Ds, judges, judge_colors)
    savefig("plots/judge_posteriors.png")
    plt
end