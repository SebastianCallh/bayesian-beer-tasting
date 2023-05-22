# A Bayesian Beer Tasting
Companion repo to my blog article [A Bayesian Beer Tasting](https://sebastiancallh.github.io/post/bayesian-beer-tasting) Contains code for fitting the judge model and plotting posterior beer quality and posterior judge characteristics.

# Reproducing the experiments
Make sure you have Julia installed (tested with Julia 1.9). `cd` to the project root folder and run `julia --project` to enter the Julia REPL in the project environment. `Run using Pkg; Pkg.instantiate()`, to pull down the dependencies, followed by `include("main.jl")` produce the model and the plots.
