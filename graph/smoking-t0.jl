using DataFrames, CSV
using Plots

filename = "logs/smoking-t0.log";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);

w_values = [1.0, 2.0, 5.0, 10.0, 20.0];
nvalues = length(w_values);

# Keep w values for C = 1.
w0 = Vector{Float64}(undef, 0);
w1 = Vector{Float64}(undef, 0);
acc = Vector{Float64}(undef, 0);
for i = 1:nrows
    local row = df[i, :];
    if row[:c] == 1
        push!(w0, row[:w0]);
        push!(w1, row[:w1]);
        push!(acc, row[:acc]);
    end
end

# Each sub-vector contains the w1 values paired with w0 = w_values[i]
# For example, w_pairs[2] contains the values of w0 that were paired with w0 = 2.0
w_pairs = Vector{Vector{Tuple}}(undef, nvalues);
for i = 1:nvalues
    w_pairs[i] = Vector{Tuple{Float64,Float64}}(undef, 0);
end
for i in eachindex(w0)
    w0_idx = findfirst(x -> x == w0[i], w_values);
    push!(w_pairs[w0_idx], (w1[i], acc[i]));
end

for i = 1:nvalues
    pair = w_pairs[i]
    local p = plot(
        title = "Linear kernel, w0 = " * string(w_values[i]),
        xlabel = "w1",
        ylabel = "Accuracy[%]",
        legend = false,
    );
    for j in eachindex(pair)
        scatter!(
            p, [pair[j][1]], [pair[j][2]],
            markercolor = :black,
            markersize = 5
            );
    end
    savefig(p, "image/smoking-t0/smoking-t0-w0_" * string(w_values[i]) * ".png");
end
