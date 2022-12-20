### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 21e9fe62-7e50-11ed-3f82-afa741839b3e
begin
	using Markdown
	using PlutoUI
end

# ╔═╡ ed431250-0869-430e-a724-8549804f05a7
md"""
# Deep Learning - Neural Networks
## Aristotle Universtity Thessaloniki - School of Informatics
### Assignment 2: Support Vector Machines
#### Antoniou, Antonios - 9482
#### aantonii@ece.auth.gr
[GitHub repository can be found here](https://github.com/anthonyisafk/deep-learning-svm)
"""

# ╔═╡ 94869478-5a9b-44e0-8818-97f6e11a8cbb
md"""
## Introduction

The purpose of the second assignment of the Deep Learning course is to build a Support Vector Machine **(SVM)**, and experiment with the different types of hyperparamters and pre-processing of the dataset the model will be trained upon. Due to the diversity of hyperparameters and how they impact the result of a predictor's training, accuracy and ability to classify a newly introduced data point, we first need to clear up the fundamentals of SVM's.

### How does an SVM work?

An SVM's function is based on the training datapoints that reside on the **separating hyperplane** between two classes, called the _**support vectors**_. For example, if we use [the libsvm official site](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) to build a basic two-class dataset, we can see how that notion affects the forming of the hyperplane that's used to differentiate between them.

$(LocalResource(
	"libsvm-example.png",
	:style => "text-align: center;
			   display: block;
			   margin-left: auto;
			   margin-right: auto;"
))
\
The support vectors here are the datapoints lying closest to the hyperplane, and are the only points actively participating in the training procedure and refining the classifier. To make that statement clearer:
"""

# ╔═╡ 4ed05c08-47ea-4fcb-8f3a-c62518fb3858
html"""
<figure>
	<img src="https://www.mathworks.com/discovery/support-vector-machine/_jcr_content/mainParsys/image.adapt.full.medium.jpg/1671257003156.jpg" style="width:60%; display:block; margin-right:auto; margin-left:auto;">
	<figcaption style="font-size: 14px; text-align:center;">Credit: Matlab. https://www.mathworks.com/discovery/support-vector-machine.html</figcaption> 
</figure>
"""

# ╔═╡ 2147348f-261e-42ef-a5b8-db03476c5b5b
md"""
Despite the multitude of both `+` and `-` instances, the only ones that matter to the SVM are the highlighted ones, that lie on either side of the separating area.

### What is the role of support vectors?

During both training and predicting, a datapoint (or *vector*, as has already been addressed as) is essentially compared to every support vector. That comparison, is done, not in the **dimension** (and space in general) that is "native" to the data. SVM's apply **transformations** to the data, in an attempt to augment the probability of the dataset being **linearly discriminant**. This is done through a _**Kernel function**_, and this is where the variety of options in order to solve the same problem starts to set in. A few examples, for two vectors $x_{0}$ and $x_{1}$:
1. Linear: $K(x_{0},x_{1})=x_{0}^{T}\cdot x_{1}$
2. Polynomial: $K(x_{0},x_{1})=(\alpha x_{0}^{T}\cdot x_{1}+r)^{d}$
3. Gaussian:
    * a. $K(x_{0},x_{1})=exp(-\frac{||x_{0}-x_{1}||^{2}}{2\sigma^{2}}), or$
    * b. $K(x_{0},x_{1})=exp(-\gamma||x_{0}-x_{1}||^{2})$

Needless to say, some libraries make a few more assumptions about the values of some coefficients. For example, *libsvm* supposes that $2\sigma^{2}=\#features$, $\gamma=\frac{1}{\#features}$. So, depending on the framework one uses to train a model, there is more or less flexibility depending on the type of SVM and kernel they want to use.

### Outline of other hyperparameters

The training of an SVM is translated into the solution of a **quadratic programming problem**:
\
\
4. $Q(a)=\sum_{i=1}^{N}\alpha_{i}-\frac{1}{2}\sum_{i=1}^{N}\sum_{j=1}^{N}\alpha_{i}\alpha_{j}d_{i}d_{j}x_{i}^{T}x_{j}$
\
\
The problem above is derived, or is the **[dual problem](https://en.wikipedia.org/wiki/Duality_(optimization))**, of a **primal** one. Briefly explained, the dual problem is a transformation of an optimization problem, that yields the same results, by optimizing a different set of variables:
\
\
5. $J(\underline{w},b,\alpha)=\frac{1}{2}\underline{w}^{T}\underline{w}-\sum_{i=1}^{N}\alpha_{i}[d_{i}(\underline{w}^{T}x_{i}+b)-1]$
\
\
The $\alpha$ coefficients in (4) are non-zero for the indices that correspond to the support vectors of either class. 
\
However, the formula above makes the assumption that, even on a higher dimension, all vectors are **distinguishable**. So, we introduce the $C$ quantity, and its $w_{j}$ counterparts. These parameters are a way to tell the SVM **how much we care about it making a couple of errors** (the falsely classified elements could be noisy after all, this is not the model's fault, but a pre-processing matter). So, they are used to ensure that the margin of the resulting hyperplane (see image above), is still optimally large, even though the two classes aren't perfectly divided.
\
So the primal problem in (5) will be transformed into finding the minimum of:
\
\
6. $\Phi(\underline{w}, \xi)=\frac{1}{2}\underline{w}^{T}\underline{w}+C\sum_{i=1}^{N}\xi_{i}$
\
\
This is where $C$ comes into play. Regarding the $w_{j}$ parameters, they are weights that are added to the $\xi_{i}$ ones, for class $j$.
\
After that overview, we now have an idea of all the variation we can provide to an SVM, in order to help the model reach a solution we deem trustworthy.

## Building an SVM with libsvm

The solution to the optimization problem that we researched above means **massive computation and memory workload**. Many methods have been suggested, implemented and used, such as **decomposition** of the solution matrix. Consequently, we will use **[libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)** for creating a problem to be optimized and training the respective SVM.
\
\
Suppose $n$ is the total number of datapoints in the training set, and $f$ is the number of useful features we will use after pre-processing. The library expects an array X of size $(n, f)$ and Y of size $(n, 1)$. With these two arrays we form an `svm_problem`. The second component of training with libsvm is the `svm_parameter` string, that dictates the values of the parameters in the libsvm manual.
\
\
Regarding those parameters, an argument parser was built in such a way that both the parameter string would be conveniently given to the library and a selected set of values would be logged after every run, depending on the type of kernel utilized. This means that a call to the script will look like this:
```bash
.\src\svm.py -s 0 -t 1 -d 3 -r 0 -c 5000 -w0 1 -w1 2
```
And a logging entry of the run above would be:
```csv
s,t,d,g,r,c,h,time,acc
0,1,3,0.04,0.0,5000,1,236.507,69.240
```
At this point, it's useful to point out that the parser (based on the [argparse](https://docs.python.org/3/library/argparse.html) python library) is set up with the same default values libsvm is, except for the `-h` shrinking parameter, that was set to 0. **The reason for that was that, in the particular examples that the predictor was tested on, it resulted in both smaller training times and the exact same rates in accuracy.**

## Training tests conducted

The larger part of the tests was done on the ["Body Signal of Smoking" dataset](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking). It was selected because of its natively binary nature (are there signs of smoking or not?), large number of samples (55692) and adequate number of features (26). The nummber of features was deliberately kept somewhat low, in order to keep the training time _relatively_ low. 

### Preprocessing 

Upon examining the dataset, we can observe three preprocessing tasks that need to take place:
* Some columns of data are in string form, and they need to be encoded. Those are `gender` and `tartar`,
* Some columns contain the same value on both classes. That happens to be `oral`.
* The features that remained now need to have their values normalized, to make sure we don't artificially make one feature more important than the others.
To serve that purpose, we introduce the **`split_features_and_classes`** function:
```python
def split_features_and_classes(
    df,
    class_col: str,
    encode: List[str] = None,
    drop: List[str] = None
) -> tuple[pandas.DataFrame, numpy.ndarray, numpy.ndarray]
```
It takes the column that corresponds to the y targets, the list of columns that need to be encoded into numeric values (with `sklearn.preprocessing.LabelEncoder`), and the values that will be dropped from the DataFrame. Here, apart from `oral`, we also drop `ID`.
\
\
Last, we need a function to get the whole X and Y arrays, and divide them into the training and testing set. Because we're dealing with support vectors, whose placements in the dataset is unknown, we only keep a little segment of the original data (more often 10%), so as not to disrupt class balance and minimize the probability of depriving the dataset of its support vectors. It is of great significance to note that there is a reason behind keeping some datapoints -even a few- for testing, with the risk of imbalance. We need to make sure that, after training, the model **wasn't overfitted**. The safest way to do that is by using vectors that had never been implanted into the dataset, up until the moment of testing. Think of the procedure as a very loosely put-together cross-validation.
\
\
Now that the dataset is normalized, sanitized, and split into training and testing samples, we can move on to the experimenting phase. We will see the results of using different kernels, and different hyperparameters given to them, on a **C-SVC**. For the mathematical notation of each of the kernels that will be examined, you can another look at (1), (2) and (3).

### Linear kernel

Here, we have room to experiment with the error weight parameters, $C$, $w_{0}$ and $w_{1}$. The majority of the samples correspond to non-smokers, so an initial thought would suggest that different $w_{0}$ and $w_{1}$ values could affect the resulting accuracy for the better. 

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6466e524967496866901a78fca3f2e9ea445a559"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─21e9fe62-7e50-11ed-3f82-afa741839b3e
# ╟─ed431250-0869-430e-a724-8549804f05a7
# ╟─94869478-5a9b-44e0-8818-97f6e11a8cbb
# ╟─4ed05c08-47ea-4fcb-8f3a-c62518fb3858
# ╠═2147348f-261e-42ef-a5b8-db03476c5b5b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
