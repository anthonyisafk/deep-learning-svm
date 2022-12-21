using DataFrames, CSV
using Plots

skip = 6; # drop last `skip` rows

filename = "logs/smoking-t2.log";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);
df = first(df, nrows - skip);
nrows = nrows - skip;

g_values = [0.01, 0.03, 0.04, 0.05, 0.1];
w_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
ng, nw = length(g_values), length(w_values);
gdict = Dict{Float64, Int64}(g_values[i]=>i for i in eachindex(g_values));
wdict = Dict{Float64, Int64}(w_values[i]=>i for i in eachindex(w_values));

tests_per_g = Vector{Vector{Tuple{Float64,Float64,Float64}}}(undef, ng);
for i in eachindex(tests_per_g)
    tests_per_g[i] = Vector{Tuple{Float64,Float64,Float64}}(undef, 0);
end

for i = 1:nrows
    local row = df[i, :];
    push!(tests_per_g[gdict[row[:g]]], (row[:w0], row[:w1], row[:acc]));
end

for i in eachindex(tests_per_g)
    local tests = tests_per_g[i];
    local curr_g = g_values[i];
    local n = length(tests);
    local plots = Vector{Plots.Plot{Plots.GRBackend}}(undef, nw);
    for j = 1:nw
        plots[j] = plot(
            title = "g: " * string(curr_g) * " w0: " * string(w_values[j]),
            titlefontsize = 10,
            xlabel = "w1",
            ylabel = "Accuracy[%]",
            legend = false
        );
    end

    for j = 1:n
        t = tests[j];
        # print("w0 : $w0\n");
        w0 = wdict[t[1]];
        scatter!(
            plots[w0],
            [t[2]], [t[3]],
            markercolor = :black,
            markersize = 4,
        );
    end
    local p = plot(plots[1], plots[2], plots[3], plots[4], plots[5], layout=@layout[grid(1, 3); grid(1, 2)]);
    savefig(p, "image/smoking-t2/smoking-t2-g" * string(curr_g) * ".png");
end