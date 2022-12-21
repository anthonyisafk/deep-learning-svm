using DataFrames, CSV
using Plots

skipped = 6;

filename = "logs/smoking-t2.csv";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);
df = last(df, skipped);
nrows = skipped;

p = Plots.scatter(
    title = "g = 0.04",
    xlabel = "C",
    ylabel = "Accuracy[%]",
    df[!, :c], df[!, :acc],
    markersize = 4,
    markercolor = :black,
    legend = false
);
savefig(p, "image/smoking-t2/smoking-t2-largec.png");