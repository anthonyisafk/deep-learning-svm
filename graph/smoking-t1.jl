# Graphs for `-t 1` (Polynomial kernel).
# For every degree `d` in [2,7], output an image with 3 subplots.
# One subplot for each value of `r` in `r_values`.
# The x-axis is the range of the values of `C`.

using DataFrames, CSV
using Plots

skip = 4; # drop last `skip` rows

filename = "logs/smoking-t1.log";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);
df = first(df, nrows - skip);
nrows = nrows - skip;

r_values = [-0.1, 0.0, 0.1];
c_values = [1, 5, 10, 20, 50];
d_values = [i for i = 2:7];
nd, nr, nc = length(d_values), length(r_values), length(c_values);
ddict = Dict{Int64, Int64}(d_values[i]=>i for i in eachindex(d_values));
rdict = Dict{Float64, Int64}(r_values[i]=>i for i in eachindex(r_values));
cdict = Dict{Int64, Int64}(c_values[i]=>i for i in eachindex(c_values));

tests_per_d = Vector{Vector{Tuple{Float64,Int64,Float64}}}(undef, nd);
for i in eachindex(tests_per_d)
    tests_per_d[i] = Vector{Tuple{Float64,Int64,Float64}}(undef, 0);
end

for i = 1:nrows
    local row = df[i, :];
    push!(tests_per_d[ddict[row[:d]]], (row[:r], row[:c], row[:acc]));
end

for i in eachindex(tests_per_d)
    local tests = tests_per_d[i];
    local curr_d = d_values[i];
    local n = length(tests);
    local plots = Vector{Plots.Plot{Plots.GRBackend}}(undef, nr);
    for j = 1:nr
        plots[j] = plot(
            title = "d: " * string(curr_d) * " r: " * string(r_values[j]),
            xlabel = "C",
            ylabel = "Accuracy[%]",
            legend = false
        );
    end

    for j = 1:n
        t = tests[j];
        r = rdict[t[1]];
        scatter!(
            plots[r],
            [t[2]], [t[3]],
            markercolor = :black,
            markersize = 4,
        );
    end
    local p = plot(plots[1], plots[3], plots[2], layout=@layout[grid(1, 2); b]);
    savefig(p, "image/smoking-t1/smoking-t1-d" * string(curr_d) * ".png");
end