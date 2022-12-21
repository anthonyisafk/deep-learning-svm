# Graphs for `-t 3` (Sigmoid kernel).
# The `r` parameter ranges in `r_values`.
# For every constant value of `r`, output a plot.
# x-axis is the values of `C` (ranging in `c_values`).

using DataFrames, CSV
using Plots

skip = 1; # drop last `skip` rows

filename = "logs/smoking-t3.log";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);
df = first(df, nrows - skip);
nrows = nrows - skip;

r_values = [-0.2, -0.1, 0.0, 0.1, 0.2];
c_values = [1, 5, 10, 20, 50];
nr, nc = length(r_values), length(c_values);
rdict = Dict{Float64, Int64}(r_values[i]=>i for i in eachindex(r_values));
cdict = Dict{Int64, Int64}(c_values[i]=>i for i in eachindex(c_values));

tests_per_r = Vector{Vector{Tuple{Int64,Float64}}}(undef, nr);
for i in eachindex(tests_per_r)
    tests_per_r[i] = Vector{Tuple{Int64,Float64}}(undef, 0);
end

for i = 1:nrows
    local row = df[i, :];
    push!(tests_per_r[rdict[row[:r]]], (row[:c], row[:acc]));
end

for i in eachindex(tests_per_r)
    local tests = tests_per_r[i];
    local curr_r = r_values[i];
    local n = length(tests);
    local p = plot(
        title = "r: " * string(curr_r),
        xlabel = "C",
        ylabel = "Accuracy[%]",
        legend = false
    );

    for j = 1:n
        t = tests[j];
        scatter!(
            p,
            [t[1]], [t[2]],
            markercolor = :black,
            markersize = 4,
        );
    end
    savefig(p, "image/smoking-t3/smoking-t3-r" * string(curr_r) * ".png");
end