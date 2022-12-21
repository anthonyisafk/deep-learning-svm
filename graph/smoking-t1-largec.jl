# Graphs for `-t 1` (Polynomial kernel), anď large C values.
# For every `r` in `r_values`, display the accuracy
# as a function of C.

using DataFrames, CSV
using Plots

skip = 4; # drop last `skip` rows

filename = "logs/smoking-t1-large_c.csv";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);
df = last(df, nrows - skip);
nrows = nrows - skip;

r_values = [0.1 * i for i = -5:1];
r_values[3] = -0.3; # fix machine precision inaccuracy.
c_values = [10 ^ i for i = 1:5];
nr, nc = length(r_values), length(c_values);
rdict = Dict{Float64, Int64}(r_values[i]=>i for i in eachindex(r_values));
cdict = Dict{Int64, Int64}(c_values[i]=>i for i in eachindex(c_values));

tests_per_r = Vector{Vector{Tuple{Int64,Float64}}}(undef, nr);
for i in eachindex(tests_per_r)
    tests_per_r[i] = Vector{Tuple{Int64,Float64}}(undef, 0);
end

for i = 1:nrows
    local row = df[i, :];
    if row[:h] == 1
        push!(tests_per_r[rdict[row[:r]]], (row[:c], row[:acc]));
    end
end

for i in eachindex(tests_per_r)
    local tests = tests_per_r[i];
    local curr_r = r_values[i];
    local n = length(tests);
    local p = plot(
        title = "r = " * string(curr_r),
        xlabel = "C",
        ylabel = "Accuracy[%]",
        legend = false,
        xaxis = :log
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
    savefig(p, "image/smoking-t1-largec/smoking-t1-largec-r(" * string(curr_r) * ").png");
end