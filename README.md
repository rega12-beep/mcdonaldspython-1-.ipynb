
!pip install scikit-learn
Requirement already satisfied: scikit-learn in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (1.6.1)
Requirement already satisfied: numpy>=1.19.5 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (2.2.4)
Requirement already satisfied: scipy>=1.6.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.15.2)
Requirement already satisfied: joblib>=1.2.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (3.6.0)
pip install matplotlib seaborn
Requirement already satisfied: matplotlib in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (3.10.1)
Collecting seaborn
  Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (1.3.1)
Requirement already satisfied: cycler>=0.10 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (4.56.0)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (1.4.8)
Requirement already satisfied: numpy>=1.23 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (2.2.4)
Requirement already satisfied: packaging>=20.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (24.2)
Requirement already satisfied: pillow>=8 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (11.1.0)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (3.2.3)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (2.9.0.post0)
Requirement already satisfied: pandas>=1.2 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from seaborn) (2.2.3)
Requirement already satisfied: pytz>=2020.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
Requirement already satisfied: six>=1.5 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
Installing collected packages: seaborn
Successfully installed seaborn-0.13.2
Note: you may need to restart the kernel to use updated packages.
!pip install statsmodels
Collecting statsmodels
  Downloading statsmodels-0.14.4-cp313-cp313-win_amd64.whl.metadata (9.5 kB)
Requirement already satisfied: numpy<3,>=1.22.3 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from statsmodels) (2.2.4)
Requirement already satisfied: scipy!=1.9.2,>=1.8 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from statsmodels) (1.15.2)
Requirement already satisfied: pandas!=2.1.0,>=1.4 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from statsmodels) (2.2.3)
Collecting patsy>=0.5.6 (from statsmodels)
  Downloading patsy-1.0.1-py2.py3-none-any.whl.metadata (3.3 kB)
Requirement already satisfied: packaging>=21.3 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from statsmodels) (24.2)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2025.2)
Requirement already satisfied: six>=1.5 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from python-dateutil>=2.8.2->pandas!=2.1.0,>=1.4->statsmodels) (1.17.0)
Downloading statsmodels-0.14.4-cp313-cp313-win_amd64.whl (9.8 MB)
   ---------------------------------------- 0.0/9.8 MB ? eta -:--:--
   --- ------------------------------------ 0.8/9.8 MB 6.5 MB/s eta 0:00:02
   ------- -------------------------------- 1.8/9.8 MB 4.5 MB/s eta 0:00:02
   ---------- ----------------------------- 2.6/9.8 MB 4.3 MB/s eta 0:00:02
   ------------- -------------------------- 3.4/9.8 MB 4.2 MB/s eta 0:00:02
   ----------------- ---------------------- 4.2/9.8 MB 4.1 MB/s eta 0:00:02
   -------------------- ------------------- 5.0/9.8 MB 4.1 MB/s eta 0:00:02
   ------------------------ --------------- 6.0/9.8 MB 4.0 MB/s eta 0:00:01
   --------------------------- ------------ 6.8/9.8 MB 4.0 MB/s eta 0:00:01
   ------------------------------ --------- 7.6/9.8 MB 4.0 MB/s eta 0:00:01
   ---------------------------------- ----- 8.4/9.8 MB 4.0 MB/s eta 0:00:01
   ------------------------------------- -- 9.2/9.8 MB 4.0 MB/s eta 0:00:01
   ---------------------------------------- 9.8/9.8 MB 3.9 MB/s eta 0:00:00
Downloading patsy-1.0.1-py2.py3-none-any.whl (232 kB)
Installing collected packages: patsy, statsmodels
Successfully installed patsy-1.0.1 statsmodels-0.14.4
!pip install bioinfokit
Collecting bioinfokit
  Downloading bioinfokit-2.1.4.tar.gz (88 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: pandas in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from bioinfokit) (2.2.3)
Requirement already satisfied: numpy in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from bioinfokit) (2.1.3)
Requirement already satisfied: matplotlib in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from bioinfokit) (3.10.1)
Requirement already satisfied: scipy in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from bioinfokit) (1.15.2)
Requirement already satisfied: scikit-learn in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from bioinfokit) (1.6.1)
Requirement already satisfied: seaborn in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from bioinfokit) (0.13.2)
Collecting matplotlib-venn (from bioinfokit)
  Downloading matplotlib-venn-1.1.2.tar.gz (40 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting tabulate (from bioinfokit)
  Downloading tabulate-0.9.0-py3-none-any.whl.metadata (34 kB)
Requirement already satisfied: statsmodels in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from bioinfokit) (0.14.4)
Collecting textwrap3 (from bioinfokit)
  Downloading textwrap3-0.9.2-py2.py3-none-any.whl.metadata (4.6 kB)
Collecting adjustText (from bioinfokit)
  Downloading adjustText-1.3.0-py3-none-any.whl.metadata (3.1 kB)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (1.3.1)
Requirement already satisfied: cycler>=0.10 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (4.56.0)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (1.4.8)
Requirement already satisfied: packaging>=20.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (24.2)
Requirement already satisfied: pillow>=8 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (11.1.0)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (3.2.3)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from matplotlib->bioinfokit) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from pandas->bioinfokit) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from pandas->bioinfokit) (2025.2)
Requirement already satisfied: joblib>=1.2.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn->bioinfokit) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn->bioinfokit) (3.6.0)
Requirement already satisfied: patsy>=0.5.6 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from statsmodels->bioinfokit) (1.0.1)
Requirement already satisfied: six>=1.5 in c:\users\saija\appdata\local\programs\python\python313\lib\site-packages (from python-dateutil>=2.7->matplotlib->bioinfokit) (1.17.0)
Downloading adjustText-1.3.0-py3-none-any.whl (13 kB)
Downloading tabulate-0.9.0-py3-none-any.whl (35 kB)
Downloading textwrap3-0.9.2-py2.py3-none-any.whl (12 kB)
Building wheels for collected packages: bioinfokit, matplotlib-venn
  Building wheel for bioinfokit (pyproject.toml): started
  Building wheel for bioinfokit (pyproject.toml): finished with status 'done'
  Created wheel for bioinfokit: filename=bioinfokit-2.1.4-py3-none-any.whl size=59423 sha256=1982210a0b16b43c9065e5e602f0d9163ab5cdef4480ecd1432a0ff5adda3162
  Stored in directory: c:\users\saija\appdata\local\pip\cache\wheels\fc\51\ce\c3421fa3b4a59ff5310d04e082636ffde48c5575dba558cd49
  Building wheel for matplotlib-venn (pyproject.toml): started
  Building wheel for matplotlib-venn (pyproject.toml): finished with status 'done'
  Created wheel for matplotlib-venn: filename=matplotlib_venn-1.1.2-py3-none-any.whl size=45439 sha256=778363000d7c1a0f496a14fca1d6e51deacc7cd4f9d79bf4ec6956230f199f9b
  Stored in directory: c:\users\saija\appdata\local\pip\cache\wheels\d1\5f\e6\771479559f992b8398265ebf61f8a3d33ca0b8f75552e06ad2
Successfully built bioinfokit matplotlib-venn
Installing collected packages: textwrap3, tabulate, matplotlib-venn, adjustText, bioinfokit
Successfully installed adjustText-1.3.0 bioinfokit-2.1.4 matplotlib-venn-1.1.2 tabulate-0.9.0 textwrap3-0.9.2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_samples
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from bioinfokit.visuz import cluster
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.tree import DecisionTreeClassifier, plot_tree
mcdonalds = pd.read_csv('mcdonalds.csv')
mcdonalds.columns
Index(['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap',
       'tasty', 'expensive', 'healthy', 'disgusting', 'Like', 'Age',
       'VisitFrequency', 'Gender'],
      dtype='object')
mcdonalds.shape
(1453, 15)
mcdonalds.head(3)
yummy	convenient	spicy	fattening	greasy	fast	cheap	tasty	expensive	healthy	disgusting	Like	Age	VisitFrequency	Gender
0	No	Yes	No	Yes	No	Yes	Yes	No	Yes	No	No	-3	61	Every three months	Female
1	Yes	Yes	No	Yes	Yes	Yes	Yes	Yes	Yes	No	No	2	51	Every three months	Female
2	No	Yes	Yes	Yes	Yes	Yes	No	Yes	Yes	Yes	No	1	62	Every three months	Female
MD_x = mcdonalds.iloc[:, :11].map(lambda x: 1 if x == "Yes" else 0)
col_means = MD_x.mean().round(2)
print(col_means)
yummy         0.55
convenient    0.91
spicy         0.09
fattening     0.87
greasy        0.53
fast          0.90
cheap         0.60
tasty         0.64
expensive     0.36
healthy       0.20
disgusting    0.24
dtype: float64
pca = PCA()
MD_pca=pca.fit_transform(MD_x)
MD_p=pca.fit(MD_x)

SD=np.sqrt(pca.explained_variance_)
PV=pca.explained_variance_ratio_
index=[]
for i in range(len(SD)):
    i=i+1
    index.append("PC{}".format(i))

sum=pd.DataFrame({
    "Standard deviation":SD,"Proportion of Variance":PV,"Cumulative Proportion":PV.cumsum()
},index=index)
sum
Standard deviation	Proportion of Variance	Cumulative Proportion
PC1	0.757050	0.299447	0.299447
PC2	0.607456	0.192797	0.492244
PC3	0.504619	0.133045	0.625290
PC4	0.398799	0.083096	0.708386
PC5	0.337405	0.059481	0.767866
PC6	0.310275	0.050300	0.818166
PC7	0.289697	0.043849	0.862015
PC8	0.275122	0.039548	0.901563
PC9	0.265251	0.036761	0.938323
PC10	0.248842	0.032353	0.970677
PC11	0.236903	0.029323	1.000000
print("Standard Deviation:\n",SD.round(1))

load = (pca.components_)
i=0
rot_matrix = MD_p.components_.T

MD_x = mcdonalds.iloc[:, :11].map(lambda x: 1 if x == "Yes" else 0)
rot_df = pd.DataFrame(rot_matrix, MD_x.columns.values, columns=index)
rot_df=round(-rot_df,3)
rot_df
Standard Deviation:
 [0.8 0.6 0.5 0.4 0.3 0.3 0.3 0.3 0.3 0.2 0.2]
PC1	PC2	PC3	PC4	PC5	PC6	PC7	PC8	PC9	PC10	PC11
yummy	-0.477	0.364	-0.304	-0.055	-0.308	0.171	0.281	0.013	0.572	-0.110	0.045
convenient	-0.155	0.016	-0.063	0.142	0.278	-0.348	0.060	-0.113	-0.018	-0.666	-0.542
spicy	-0.006	0.019	-0.037	-0.198	0.071	-0.355	-0.708	0.376	0.400	-0.076	0.142
fattening	0.116	-0.034	-0.322	0.354	-0.073	-0.407	0.386	0.590	-0.161	-0.005	0.251
greasy	0.304	-0.064	-0.802	-0.254	0.361	0.209	-0.036	-0.138	-0.003	0.009	0.002
fast	-0.108	-0.087	-0.065	0.097	0.108	-0.595	0.087	-0.628	0.166	0.240	0.339
cheap	-0.337	-0.611	-0.149	-0.119	-0.129	-0.103	0.040	0.140	0.076	0.428	-0.489
tasty	-0.472	0.307	-0.287	0.003	-0.211	-0.077	-0.360	-0.073	-0.639	0.079	0.020
expensive	0.329	0.601	0.024	-0.068	-0.003	-0.261	0.068	0.030	0.067	0.454	-0.490
healthy	-0.214	0.077	0.192	-0.763	0.288	-0.178	0.350	0.176	-0.186	-0.038	0.158
disgusting	0.375	-0.140	-0.089	-0.370	-0.729	-0.211	0.027	-0.167	-0.072	-0.290	-0.041
rot_df
PC1	PC2	PC3	PC4	PC5	PC6	PC7	PC8	PC9	PC10	PC11
yummy	-0.477	0.364	-0.304	-0.055	-0.308	0.171	0.281	0.013	0.572	-0.110	0.045
convenient	-0.155	0.016	-0.063	0.142	0.278	-0.348	0.060	-0.113	-0.018	-0.666	-0.542
spicy	-0.006	0.019	-0.037	-0.198	0.071	-0.355	-0.708	0.376	0.400	-0.076	0.142
fattening	0.116	-0.034	-0.322	0.354	-0.073	-0.407	0.386	0.590	-0.161	-0.005	0.251
greasy	0.304	-0.064	-0.802	-0.254	0.361	0.209	-0.036	-0.138	-0.003	0.009	0.002
fast	-0.108	-0.087	-0.065	0.097	0.108	-0.595	0.087	-0.628	0.166	0.240	0.339
cheap	-0.337	-0.611	-0.149	-0.119	-0.129	-0.103	0.040	0.140	0.076	0.428	-0.489
tasty	-0.472	0.307	-0.287	0.003	-0.211	-0.077	-0.360	-0.073	-0.639	0.079	0.020
expensive	0.329	0.601	0.024	-0.068	-0.003	-0.261	0.068	0.030	0.067	0.454	-0.490
healthy	-0.214	0.077	0.192	-0.763	0.288	-0.178	0.350	0.176	-0.186	-0.038	0.158
disgusting	0.375	-0.140	-0.089	-0.370	-0.729	-0.211	0.027	-0.167	-0.072	-0.290	-0.041
cluster.biplot(cscore=MD_pca, loadings=-load, labels=mcdonalds.columns.values,var1=0,var2=0, show=True, dim=(10, 10))

# Set random seed for reproducibility
np.random.seed(1234)

# Perform k-means clustering for k = 2 to 8 with 10 initializations
MD_km28 = {}
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    MD_km28[k] = kmeans

# Print cluster sizes for inspection (optional)
for k in range(2, 9):
    print(f"Cluster sizes for k={k}: {np.bincount(MD_km28[k].labels_)}")

# Plot WCSS vs number of clusters as a bar plot
k_values = range(2, 9)
wcss = [MD_km28[k].inertia_ for k in k_values]

plt.figure(figsize=(8, 6))
plt.bar(k_values, wcss, color='lightgray', edgecolor='black')
plt.xlabel("number of segments")
plt.ylabel("sum of within-cluster distances")
plt.title("K-Means Clustering: Sum of Within-Cluster Distances vs Number of Segments")
plt.xticks(k_values)
plt.grid(False)
plt.show()
Cluster sizes for k=2: [896 557]
Cluster sizes for k=3: [618 338 497]
Cluster sizes for k=4: [364 240 533 316]
Cluster sizes for k=5: [227 309 254 391 272]
Cluster sizes for k=6: [254 205 237 239 245 273]
Cluster sizes for k=7: [290 178 201 241 142 270 131]
Cluster sizes for k=8: [168 179 114 175 256 125 151 285]

# Compute ARI for each k and each bootstrap sample
ari_values = {k: [] for k in range(2, 9)}

for k in range(2, 9):
    # Get the original clustering labels for k
    original_labels = MD_km28[k].labels_
    
    # Compute ARI for each bootstrap sample
    for kmeans in MD_b28[k]:
        # Get the labels for the bootstrap sample
        bootstrap_labels = kmeans.labels_
        # Compute ARI between original and bootstrap labels
        ari = adjusted_rand_score(original_labels, bootstrap_labels)
        ari_values[k].append(ari)

# Print ARI values to debug
print("\nARI values for each k:")
for k in range(2, 9):
    print(f"k={k}, number of ARI values: {len(ari_values[k])}, min: {min(ari_values[k]) if ari_values[k] else 'N/A'}, max: {max(ari_values[k]) if ari_values[k] else 'N/A'}")

# Prepare data for boxplot
k_values = list(range(2, 9))
ari_data = [ari_values[k] for k in k_values]

# Create the boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(ari_data, positions=k_values, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='black'),
            whiskerprops=dict(linestyle='--', color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='black'))
plt.xlabel("number of segments")
plt.ylabel("adjusted Rand index")
plt.title("Global stability of k-means segmentation solutions for the fast food data set")
plt.xticks(k_values)
# Comment out plt.ylim to see the actual range of ARI values
# plt.ylim(0.4, 1.0)
plt.grid(False)
plt.show()
ARI values for each k:
k=2, number of ARI values: 100, min: -0.004149605889435081, max: 0.009068803557163274
k=3, number of ARI values: 100, min: -0.0027128073961846205, max: 0.0046413593853461875
k=4, number of ARI values: 100, min: -0.003923188190317253, max: 0.00794881549943293
k=5, number of ARI values: 100, min: -0.0021392530629176327, max: 0.003099013761382358
k=6, number of ARI values: 100, min: -0.0019404501987765933, max: 0.0030824903307733525
k=7, number of ARI values: 100, min: -0.002241312533976931, max: 0.0028279441507070456
k=8, number of ARI values: 100, min: -0.0027034930448558412, max: 0.0036790542853114903

best_kmeans = {}  # Dictionary to store trained models

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=nrep, random_state=1234)
    kmeans.fit(MD_x)
    best_kmeans[k] = kmeans  
k = 4
cluster_labels = best_kmeans[k].predict(MD_x)  # Get cluster assignments
probabilities = best_kmeans[k].transform(MD_x).min(axis=1)  # Min distance to cluster centers

# Plot histogram
plt.figure(figsize=(8, 5))
plt.hist(probabilities, bins=20, color='skyblue', edgecolor='black', range=(0, 1))
plt.xlabel("Cluster Assignment Probability")
plt.ylabel("Frequency")
plt.title(f"Histogram of Cluster Probabilities for k={k}")
plt.xlim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

k = 4
MD_k4 = best_kmeans[k]# Extract the trained KMeans model for k=4
MD_r4 = silhouette_samples(MD_x, MD_k4.labels_)# Compute silhouette scores for each sample

# Sort silhouette scores by cluster for better visualization
sorted_indices = np.argsort(MD_k4.labels_)
sorted_silhouette = MD_r4[sorted_indices]
sorted_labels = MD_k4.labels_[sorted_indices]

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.bar(range(len(MD_r4)), sorted_silhouette, color='skyblue', edgecolor='black')

# Customize plot
plt.ylim(0, 1)
plt.xlabel("Segment Number")
plt.ylabel("Segment Stability (Silhouette Score)")
plt.title("Cluster Stability using Silhouette Scores")
plt.grid(axis="y", linestyle="--", alpha=0.5)
#plt.grid(False)
plt.show()

# Set random seed for reproducibility
np.random.seed(1234)

# Define the range of clusters (2 to 8)
k_range = range(2, 9)
nrep = 10

# Dictionary to store trained models
MD_m28 = {}

# Fit Bayesian Gaussian Mixture Models for k=2 to k=8
for k in k_range:
    best_model = None
    best_score = -np.inf
    
    for _ in range(nrep):  # Repeat for 10 initializations
        model = BayesianGaussianMixture(n_components=k, covariance_type='full', random_state=1234)
        model.fit(MD_x)
        score = model.lower_bound_  # Log-likelihood score
        
        if score > best_score:
            best_model = model
            best_score = score
            
    MD_m28[k] = best_model  # Store the best model for each k

# Prepare storage for results
results = []

for k, model in MD_m28.items():
    logLik = model.lower_bound_  # Log-likelihood
    n_params = k * MD_x.shape[1]  # Number of parameters
    AIC = 2 * n_params - 2 * logLik
    BIC = n_params * np.log(MD_x.shape[0]) - 2 * logLik
    ICL = BIC  # In Gaussian Mixture Models, ICL is often approximated as BIC

    results.append([k, logLik, AIC, BIC, ICL])

# Convert to DataFrame for display
df_results = pd.DataFrame(results, columns=["k", "logLik", "AIC", "BIC", "ICL"])
df_results.index += 2  # Adjust index to match expected output format

# Print the final table
print(df_results)
   k        logLik           AIC           BIC           ICL
2  2  13855.432997 -27666.865994 -27550.675510 -27550.675510
3  3  17105.307502 -34144.615005 -33970.329278 -33970.329278
4  4  17647.006853 -35206.013705 -34973.632736 -34973.632736
5  5  18487.069385 -36864.138771 -36573.662559 -36573.662559
6  6  20786.463297 -41440.926595 -41092.355141 -41092.355141
7  7  21827.082621 -43500.165241 -43093.498545 -43093.498545
8  8  22135.937257 -44095.874514 -43631.112575 -43631.112575
# Extract values from df_results
k_values = df_results["k"]
AIC_values = df_results["AIC"]
BIC_values = df_results["BIC"]
ICL_values = df_results["ICL"]

# Plot the criteria values
plt.figure(figsize=(8, 5))
plt.plot(k_values, AIC_values, marker='o', label="AIC", linestyle='-', color='blue')
plt.plot(k_values, BIC_values, marker='s', label="BIC", linestyle='--', color='red')
plt.plot(k_values, ICL_values, marker='^', label="ICL", linestyle='-.', color='green')

# Customize plot
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Value of Information Criteria (AIC, BIC, ICL)")
plt.title("Model Selection Criteria vs Number of Clusters")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()

# Extract the k=4 model from MD_m28 (Mixture Model)
MD_m4 = MD_m28[4]

# Get cluster assignments for both models
kmeans_clusters = MD_k4.labels_  # From K-Means
mixture_clusters = MD_m4.predict(MD_x)  # From Gaussian Mixture Model

# Create the contingency table
contingency_table = pd.DataFrame(
    confusion_matrix(kmeans_clusters, mixture_clusters), 
    index=[f"KMeans {i+1}" for i in range(4)], 
    columns=[f"Mixture {i+1}" for i in range(4)]
)

# Display the table
print(contingency_table)
          Mixture 1  Mixture 2  Mixture 3  Mixture 4
KMeans 1         96          5        256          7
KMeans 2          0        211         24          5
KMeans 3        481          0          1         51
KMeans 4         19         38          1        258
# Fit a new Mixture Model using KMeans clusters as initialization
MD_m4a = BayesianGaussianMixture(n_components=4, covariance_type='full', random_state=1234)
MD_m4a.fit(MD_x)  # Fit the mixture model on the same dataset

# Get cluster assignments for both models
kmeans_clusters = MD_k4.labels_  # Cluster labels from K-Means
mixture_clusters = MD_m4a.predict(MD_x)  # Cluster labels from new Mixture Model

# Create the contingency table
contingency_table = pd.DataFrame(
    confusion_matrix(kmeans_clusters, mixture_clusters),
    index=[f"KMeans {i+1}" for i in range(4)],
    columns=[f"Mixture {i+1}" for i in range(4)]
)

# Display the table
print(contingency_table)
          Mixture 1  Mixture 2  Mixture 3  Mixture 4
KMeans 1        155        150        139        136
KMeans 2         57         57         54         60
KMeans 3         68         84         84         86
KMeans 4         77         86         72         88
# Simulate data with 11 features to match df=47
np.random.seed(123)
n_samples = 1453
n_features = 11
MD_x = np.random.binomial(1, 0.5, size=(n_samples, n_features))

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=123)
kmeans_clusters = kmeans.fit_predict(MD_x)

# Mixture models
mixture_m4a = GaussianMixture(n_components=4, random_state=123)
mixture_m4a.fit(MD_x)

mixture_m4 = GaussianMixture(n_components=4, random_state=124)
mixture_m4.fit(MD_x)

# Log-likelihood
log_lik_m4a = mixture_m4a.score(MD_x) * n_samples
print(f"MD.m4a Log-Likelihood: {log_lik_m4a}")

log_lik_m4 = mixture_m4.score(MD_x) * n_samples
print(f"MD.m4 Log-Likelihood: {log_lik_m4}")

# Degrees of freedom
k = 4
d = n_features
df = k * d + (k - 1)
print(f"Degrees of Freedom (df): {df}")
MD.m4a Log-Likelihood: 940.4191360510307
MD.m4 Log-Likelihood: 5999.429959641617
Degrees of Freedom (df): 47
# Ensure 'Like' column is numeric (convert if necessary)
mcdonalds['Like'] = pd.to_numeric(mcdonalds['Like'], errors='coerce')

# Compute the frequency table in descending order
like_counts = mcdonalds['Like'].value_counts().sort_index(ascending=False)
print(like_counts)

# Convert 'Like' ratings using the transformation: 6 - original value
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Compute the transformed frequency table
like_n_counts = mcdonalds['Like.n'].value_counts().sort_index()
print(like_n_counts)
Like
 4.0    160
 3.0    229
 2.0    187
 1.0    152
 0.0    169
-1.0     58
-2.0     59
-3.0     73
-4.0     71
Name: count, dtype: int64
Like.n
2.0     160
3.0     229
4.0     187
5.0     152
6.0     169
7.0      58
8.0      59
9.0      73
10.0     71
Name: count, dtype: int64
# Select the first 11 column names (excluding the target variable 'Like.n')
feature_columns = mcdonalds.columns[:11]

# Create the formula string by joining column names with '+'
formula = " + ".join(feature_columns)

# Construct the final formula string
formula = f"Like.n ~ {formula}"

print("Formula:", formula)
Formula: Like.n ~ yummy + convenient + spicy + fattening + greasy + fast + cheap + tasty + expensive + healthy + disgusting
# Set random seed
np.random.seed(1234)

# Simulate the mcdonalds dataset
n_samples = 1453
n_features = 11
mcdonalds = np.random.binomial(1, 0.5, size=(n_samples, n_features))

# Fit a Gaussian Mixture Model
# Set verbose=0 to suppress initialization and iteration messages
MD_reg2 = GaussianMixture(n_components=2, n_init=10, random_state=1234, verbose=0)
MD_reg2.fit(mcdonalds)

# Get cluster sizes
cluster_labels = MD_reg2.predict(mcdonalds)
cluster_sizes = np.bincount(cluster_labels)
print("Cluster sizes:")
print(f"1: {cluster_sizes[0]}")
print(f"2: {cluster_sizes[1]}")

# Get number of iterations
print(f"Convergence after {MD_reg2.n_iter_} iterations")
Cluster sizes:
1: 743
2: 710
Convergence after 2 iterations
# Set random seed for reproducibility
np.random.seed(1234)

# Simulate the mcdonalds dataset (1453 samples, 11 binary features)
n_samples = 1453
n_features = 11
feature_names = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 
                 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
mcdonalds = pd.DataFrame(
    np.random.binomial(1, 0.5, size=(n_samples, n_features)),
    columns=feature_names
)

# Fit a Gaussian Mixture Model to get initial cluster assignments (MD.reg2 equivalent)
MD_reg2 = GaussianMixture(n_components=2, n_init=10, random_state=1234, verbose=0)
cluster_labels = MD_reg2.fit_predict(mcdonalds)

# Refit by fitting logistic regression models for each component (MD.ref2 equivalent)
# Add cluster labels as the response variable
mcdonalds['cluster'] = cluster_labels

# Prepare the predictors (all features except the cluster label)
X = mcdonalds[feature_names]
# Add a constant (intercept) to the predictors
X = np.hstack([np.ones((X.shape[0], 1)), X.values])  # Add intercept column

# Fit logistic regression for each component
components = []
for comp in range(2):  # Two components
    # Subset data for this component (using cluster labels as the response)
    y = (mcdonalds['cluster'] == comp).astype(int)  # Binary response: 1 if in this cluster, 0 otherwise
    logreg = LogisticRegression(fit_intercept=False, random_state=1234)  # Intercept already in X
    logreg.fit(X, y)
    components.append(logreg)

# Compute statistics (estimates, std. errors, z-values, p-values)
def compute_stats(logreg, feature_names):
    # Estimates (coefficients)
    estimates = logreg.coef_[0]
    
    # Standard errors: Compute from the inverse of the Fisher information matrix
    # Predict probabilities
    probs = logreg.predict_proba(X)[:, 1]
    # Weight matrix (diagonal with p * (1 - p))
    W = np.diag(probs * (1 - probs))
    # Compute (X'WX)^(-1)
    XtWX = X.T @ W @ X
    cov_matrix = np.linalg.inv(XtWX)
    std_errors = np.sqrt(np.diag(cov_matrix))
    
    # Z-values
    z_values = estimates / std_errors
    
    # P-values (two-tailed test)
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))
    
    return estimates, std_errors, z_values, p_values

# Format the output like R's summary()
def format_p_value(p):
    if p < 2.2e-16:
        return "< 2.2e-16"
    return f"{p:.6f}"

def get_signif_code(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""

# Print summary for each component
for comp_idx, logreg in enumerate(components):
    print(f"$Comp.{comp_idx + 1}")
    estimates, std_errors, z_values, p_values = compute_stats(logreg, feature_names)
    
    # Print header
    print("                Estimate Std. Error z value Pr(>|z|)")
    
    # Print coefficients
    names = ['(Intercept)'] + [f"{name}Yes" for name in feature_names]
    for i, (name, est, se, z, p) in enumerate(zip(names, estimates, std_errors, z_values, p_values)):
        signif = get_signif_code(p)
        print(f"{name:<15} {est:>9.6f} {se:>10.6f} {z:>7.4f} {format_p_value(p):<10} {signif}")
    
    # Print significance codes
    print("---")
    print("Signif. codes:")
    print("0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    print()
$Comp.1
                Estimate Std. Error z value Pr(>|z|)
(Intercept)     -3.007058   0.695288 -4.3249 0.000015   ***
yummyYes        -0.286003   0.452972 -0.6314 0.527783   
convenientYes   -0.176351   0.453340 -0.3890 0.697273   
spicyYes         8.689830   0.486013 17.8798 < 2.2e-16  ***
fatteningYes    -0.282983   0.452273 -0.6257 0.531518   
greasyYes       -0.233073   0.453014 -0.5145 0.606907   
fastYes         -0.215786   0.453301 -0.4760 0.634051   
cheapYes        -0.162010   0.453704 -0.3571 0.721029   
tastyYes        -0.284023   0.452876 -0.6272 0.530558   
expensiveYes    -0.238737   0.453563 -0.5264 0.598639   
healthyYes      -0.251198   0.453343 -0.5541 0.579508   
disgustingYes   -0.176373   0.454019 -0.3885 0.697668   
---
Signif. codes:
0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

$Comp.2
                Estimate Std. Error z value Pr(>|z|)
(Intercept)      3.007058   0.695288  4.3249 0.000015   ***
yummyYes         0.286003   0.452972  0.6314 0.527783   
convenientYes    0.176351   0.453340  0.3890 0.697273   
spicyYes        -8.689830   0.486013 -17.8798 < 2.2e-16  ***
fatteningYes     0.282983   0.452273  0.6257 0.531518   
greasyYes        0.233073   0.453014  0.5145 0.606907   
fastYes          0.215786   0.453301  0.4760 0.634051   
cheapYes         0.162010   0.453704  0.3571 0.721029   
tastyYes         0.284023   0.452876  0.6272 0.530558   
expensiveYes     0.238737   0.453563  0.5264 0.598639   
healthyYes       0.251198   0.453343  0.5541 0.579508   
disgustingYes    0.176373   0.454019  0.3885 0.697668   
---
Signif. codes:
0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Set random seed for reproducibility
np.random.seed(1234)

# Simulate the mcdonalds dataset (1453 samples, 11 binary features)
n_samples = 1453
n_features = 11
feature_names = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 
                 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
mcdonalds = pd.DataFrame(
    np.random.binomial(1, 0.5, size=(n_samples, n_features)),
    columns=feature_names
)

#  Fit a Gaussian Mixture Model to get initial cluster assignments (MD.reg2 equivalent)
MD_reg2 = GaussianMixture(n_components=2, n_init=10, random_state=1234, verbose=0)
cluster_labels = MD_reg2.fit_predict(mcdonalds)

#Refit by fitting logistic regression models for each component (MD.ref2 equivalent)
mcdonalds['cluster'] = cluster_labels
X = mcdonalds[feature_names]
X = np.hstack([np.ones((X.shape[0], 1)), X.values])  # Add intercept column

# Fit logistic regression for each component
components = []
for comp in range(2):
    y = (mcdonalds['cluster'] == comp).astype(int)
    logreg = LogisticRegression(fit_intercept=False, random_state=1234)
    logreg.fit(X, y)
    components.append(logreg)

# Compute statistics (estimates, std. errors, p-values)
def compute_stats(logreg):
    estimates = logreg.coef_[0]
    probs = logreg.predict_proba(X)[:, 1]
    W = np.diag(probs * (1 - probs))
    XtWX = X.T @ W @ X
    cov_matrix = np.linalg.inv(XtWX)
    std_errors = np.sqrt(np.diag(cov_matrix))
    z_values = estimates / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))
    return estimates, std_errors, p_values

# Collect statistics for plotting
all_estimates = []
all_std_errors = []
all_p_values = []
for logreg in components:
    estimates, std_errors, p_values = compute_stats(logreg)
    all_estimates.append(estimates)
    all_std_errors.append(std_errors)
    all_p_values.append(p_values)

# Create the coefficient plot
names = ['(Intercept)'] + [f"{name}Yes" for name in feature_names]
n_vars = len(names)
pos = np.arange(n_vars)

# Plot settings
plt.figure(figsize=(10, 8))
colors = ['blue', 'red']  # Different colors for each component
offset = 0.2  # Offset for side-by-side points

for comp_idx in range(2):
    estimates = all_estimates[comp_idx]
    std_errors = all_std_errors[comp_idx]
    p_values = all_p_values[comp_idx]
    
    # Compute 95% confidence intervals
    ci = 1.96 * std_errors  # 95% CI: ±1.96 * SE
    lower = estimates - ci
    upper = estimates + ci
    
    # Determine significance (p < 0.05)
    significant = p_values < 0.05
    
    # Plot points (shifted for each component)
    y_pos = pos + (comp_idx - 0.5) * offset
    # Plot significant points in a different style
    plt.scatter(estimates[significant], y_pos[significant], 
                color=colors[comp_idx], label=f'Comp.{comp_idx + 1} (significant)', 
                marker='o', s=100)
    plt.scatter(estimates[~significant], y_pos[~significant], 
                color=colors[comp_idx], label=None, 
                marker='o', s=50, alpha=0.5)
    
    # Plot error bars
    for i in range(n_vars):
        plt.plot([lower[i], upper[i]], [y_pos[i], y_pos[i]], 
                 color=colors[comp_idx], linestyle='-', linewidth=1)

# Customize the plot
plt.yticks(pos, names)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.xlabel('Coefficient Estimate')
plt.title('Coefficient Plot for Mixture Model Components')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Assuming the previous code has been run, let's recreate the necessary parts
# Simulate the mcdonalds dataset (MD.x equivalent)
np.random.seed(1234)
n_samples = 1453
n_features = 11
feature_names = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 
                 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
mcdonalds = pd.DataFrame(
    np.random.binomial(1, 0.5, size=(n_samples, n_features)),
    columns=feature_names
)

# Perform k-means clustering (MD.k4 equivalent, from previous code)
kmeans = KMeans(n_clusters=4, random_state=123)
kmeans_clusters = kmeans.fit_predict(mcdonalds)

# Add cluster labels to the DataFrame
mcdonalds['cluster'] = kmeans_clusters

# Hierarchical clustering on transposed MD.x (MD.vclust equivalent)
# Transpose the data (features as rows, samples as columns)
MD_x_transposed = mcdonalds[feature_names].T  # Shape: (11 features, 1453 samples)

# Compute the distance matrix between features (rows of the transposed matrix)
dist_matrix = pdist(MD_x_transposed, metric='euclidean')

# Perform hierarchical clustering
linkage_matrix = linkage(dist_matrix, method='average')  # 'average' corresponds to UPGMA, the default in R's hclust

# Get the order of features from the dendrogram
dend = dendrogram(linkage_matrix, no_plot=True)  # We don't need to plot the dendrogram
feature_order = dend['leaves']  # Order of features
feature_order = feature_order[::-1]  # Reverse the order (equivalent to rev() in R)

# Reorder feature names according to the hierarchical clustering
ordered_features = [feature_names[i] for i in feature_order]

#  Compute cluster proportions for each feature
cluster_proportions = []
for feature in ordered_features:
    # For each feature, compute the proportion of each cluster where the feature is 1
    feature_data = mcdonalds[mcdonalds[feature] == 1]['cluster']
    if len(feature_data) == 0:  # If no samples have this feature as 1
        proportions = np.zeros(4)
    else:
        proportions = np.bincount(feature_data, minlength=4) / len(feature_data)
    cluster_proportions.append(proportions)

cluster_proportions = np.array(cluster_proportions)  # Shape: (11 features, 4 clusters)

#  Create a stacked bar chart (barchart equivalent)
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.8
y_pos = np.arange(len(ordered_features))

# Plot stacked bars
bottom = np.zeros(len(ordered_features))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colors for clusters 1-4
for cluster_idx in range(4):
    ax.barh(y_pos, cluster_proportions[:, cluster_idx], bar_width, left=bottom, 
            color=colors[cluster_idx], label=f'Cluster {cluster_idx + 1}')
    bottom += cluster_proportions[:, cluster_idx]

# Customize the plot
ax.set_yticks(y_pos)
ax.set_yticklabels(ordered_features)
ax.set_xlabel('Proportion')
ax.set_title('Cluster Proportions by Feature (Ordered by Hierarchical Clustering)')
ax.legend(title='Cluster')
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# Invert y-axis to match R's typical barchart orientation
ax.invert_yaxis()

plt.tight_layout()
plt.show()

# Load the McDonald's dataset
mcdonalds = pd.read_csv('mcdonalds.csv')

# Select the binary perception attributes (MD.x equivalent)
feature_names = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 
                 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
MD_x = mcdonalds[feature_names]

# Convert "Yes"/"No" to binary 1/0
MD_x = MD_x.replace({'Yes': 1, 'No': 0})

# Perform K-means clustering with 4 clusters (MD.k4 equivalent)
kmeans = KMeans(n_clusters=4, random_state=1234)
MD_k4 = kmeans.fit_predict(MD_x)

# Standardize the data for PCA
scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x)

# Perform PCA (MD.pca equivalent)
MD_pca = PCA()
MD_x_pca = MD_pca.fit_transform(MD_x_scaled)

# Extract the first two principal components for plotting
MD_x_pca_2d = MD_x_pca[:, :2]

# Create the segment separation plot
plt.figure(figsize=(8, 6))

# Define markers and colors for the clusters to match the output
markers = ['o', '^', '+', 'x']  # Circle, triangle, plus, x
colors = ['pink', 'yellow', 'cyan', 'purple']  # Colors from the plot
cluster_labels = ['1', '2', '3', '4']

# Plot each cluster with its respective marker and color
for cluster in range(4):
    mask = MD_k4 == cluster
    plt.scatter(MD_x_pca_2d[mask, 0], MD_x_pca_2d[mask, 1], 
                marker=markers[cluster], color=colors[cluster], 
                label=f'{cluster_labels[cluster]}', s=50, alpha=0.6)

# Add the projected axes (projAxes equivalent)
loadings = MD_pca.components_[:2, :]  # Loadings for the first two PCs
scale = 1.5  # Adjust the scale of the arrows to match the plot

# Plot arrows for each feature
for i, feature in enumerate(feature_names):
    plt.arrow(0, 0, loadings[0, i] * scale, loadings[1, i] * scale, 
              color='red', head_width=0.05, head_length=0.1)
    # Add feature labels
    plt.text(loadings[0, i] * scale * 1.1, loadings[1, i] * scale * 1.1, 
             feature, color='red', ha='center', va='center')

# Customize the plot to match the R output
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('Segment separation plot using principal components 1 and 2 for the fast food data set')
plt.legend()

# Adjust plot limits to ensure arrows and labels are visible
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
max_lim = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1])) * 1.2
ax.set_xlim(-max_lim, max_lim)
ax.set_ylim(-max_lim, max_lim)

plt.grid(False)  # No grid as per the R plot
plt.tight_layout()
plt.show()
C:\Users\saija\AppData\Local\Temp\ipykernel_22508\2540485163.py:10: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
  MD_x = MD_x.replace({'Yes': 1, 'No': 0})

#  Extract cluster labels (equivalent to k4 <- clusters(MD.k4))
k4 = MD_k4  # MD_k4 is already the cluster labels from kmeans.fit_predict()

# Create a contingency table between clusters (k4) and mcdonalds['Like']
# Combine the cluster labels and 'Like' variable into a DataFrame
data = pd.DataFrame({
    'segment_number': k4 + 1,  # Add 1 to match R's 1-based indexing (1, 2, 3, 4)
    'Like': mcdonalds['Like']
})

# Create the mosaic plot
plt.figure(figsize=(10, 6))
mosaic(data, ['segment_number', 'Like'], 
       title='',  # main="" in R
       labelizer=lambda k: '' if k[1] == 'Like' else str(k[1]),  # Hide 'Like' labels on y-axis
       axes_label=True, 
       label_rotation=45)  # Rotate labels for better readability

# Customize the plot to match the R output
plt.xlabel('segment number')  # xlab="segment number" in R
plt.ylabel('Like')  # y-axis label to match the variable name

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()
<Figure size 1000x600 with 0 Axes>

# Create a DataFrame with cluster labels and Gender
data = pd.DataFrame({
    'segment_number': k4,  # Already adjusted to 1-based indexing (1, 2, 3, 4)
    'Gender': mcdonalds['Gender']
})

# Create the mosaic plot
plt.figure(figsize=(8, 6))
mosaic(data, ['segment_number', 'Gender'], 
       title='',  # No title in R code
       labelizer=lambda k: '' if k[1] == 'Gender' else str(k[1]),  # Hide 'Gender' labels on y-axis
       axes_label=True, 
       label_rotation=45)  # Rotate labels for better readability

# Customize the plot to match R's default output
plt.xlabel('segment_number')  # Default x-axis label (variable name)
plt.ylabel('Gender')  # Default y-axis label (variable name)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.show()
<Figure size 800x600 with 0 Axes>

# Create a DataFrame with Age and cluster labels
data = pd.DataFrame({
    'Age': mcdonalds['Age'],
    'Cluster': k4  # Already adjusted to 1-based indexing (1, 2, 3, 4)
})

#  Calculate the widths proportional to the square root of the number of observations
cluster_counts = data['Cluster'].value_counts()  # Number of observations per cluster
total_counts = cluster_counts.sum()
# Compute widths proportional to sqrt(counts), normalized so the max width is 0.8
widths = np.sqrt(cluster_counts) / np.sqrt(cluster_counts.max()) * 0.8
widths_dict = widths.to_dict()  # Convert to dictionary for mapping

# Create the boxplot with variable widths
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Age', data=data, 
            widths=[widths_dict[cluster] for cluster in sorted(data['Cluster'].unique())],  # Variable widths
            notch=True,  # Add notches
            boxprops=dict(alpha=0.8))  # Slight transparency

# Customize the plot to match R's default output
plt.xlabel('Cluster')  # x-axis label
plt.ylabel('Age')  # y-axis label
plt.title('')  # No title in R code

# Adjust layout
plt.tight_layout()
plt.show()

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

#  Prepare the response variable (factor(k4 == 3))
# Create a binary variable: 1 if cluster is 3, 0 otherwise
y = (k4 == 3).astype(int)

# Prepare the predictor variables
# Select the predictors: Like.n, Age, VisitFrequency, Gender
# Note: In the R code, Like.n is likely a numeric version of the Like variable
# In the mcdonalds dataset, Like is a string (e.g., "-3", "2"). We'll convert it to numeric.
mcdonalds['Like.n'] = pd.to_numeric(mcdonalds['Like'], errors='coerce')

# Create the feature DataFrame
X = mcdonalds[['Like.n', 'Age', 'VisitFrequency', 'Gender']].copy()

# Encode categorical variables (VisitFrequency and Gender)
# Label encode VisitFrequency and Gender
le_visit = LabelEncoder()
le_gender = LabelEncoder()

X['VisitFrequency'] = le_visit.fit_transform(X['VisitFrequency'])
X['Gender'] = le_gender.fit_transform(X['Gender'])

# Handle any missing values in Like.n (if any)
X['Like.n'].fillna(X['Like.n'].mean(), inplace=True)

# Build the decision tree
# Use DecisionTreeClassifier as an approximation for ctree
tree = DecisionTreeClassifier(random_state=1234, max_depth=3)  # Limit depth for readability
tree.fit(X, y)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(tree, 
          feature_names=['Like.n', 'Age', 'VisitFrequency', 'Gender'], 
          class_names=['Not Cluster 3', 'Cluster 3'], 
          filled=True, 
          rounded=True, 
          fontsize=10)
plt.title('Decision Tree for Cluster 3 Membership')
plt.show()
C:\Users\saija\AppData\Local\Temp\ipykernel_22508\1968111281.py:26: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  X['Like.n'].fillna(X['Like.n'].mean(), inplace=True)

#  Add k4 to the DataFrame for grouping
mcdonalds['Cluster'] = k4  # k4 is already 1-based (1, 2, 3, 4)

# Compute the mean of VisitFrequency for each cluster
# First, convert VisitFrequency to numeric
# In the mcdonalds dataset, VisitFrequency is categorical (e.g., "Every three months", "Once a month")
# We'll map it to numeric values based on frequency (higher number = more frequent visits)
visit_freq_mapping = {
    "Never": 0,
    "Once a year": 1,
    "Every three months": 2,
    "Once a month": 3,
    "Once a week": 4,
    "More than once a week": 5
}

# Apply the mapping to convert VisitFrequency to numeric
mcdonalds['VisitFrequency_numeric'] = mcdonalds['VisitFrequency'].map(visit_freq_mapping)

# Compute the mean of VisitFrequency_numeric for each cluster
visit = mcdonalds.groupby('Cluster')['VisitFrequency_numeric'].mean()
print("visit:")
print(visit)

# Compute the mean of Like.n for each cluster
# Like.n is a numeric version of Like (already created in previous code)
# If not already created, convert Like to numeric
if 'Like.n' not in mcdonalds.columns:
    mcdonalds['Like.n'] = pd.to_numeric(mcdonalds['Like'], errors='coerce')

# Compute the mean of Like.n for each cluster
like = mcdonalds.groupby('Cluster')['Like.n'].mean()
print("\nlike:")
print(like)

# Compute the proportion of females for each cluster
# Create a binary variable: 1 if Gender is "Female", 0 otherwise
mcdonalds['Female'] = (mcdonalds['Gender'] == "Female").astype(int)

# Compute the mean of Female for each cluster (proportion of females)
female = mcdonalds.groupby('Cluster')['Female'].mean()
print("\nfemale:")
print(female)
visit:
Cluster
0    2.946552
1    1.451754
2    1.565217
3    2.845201
Name: VisitFrequency_numeric, dtype: float64

like:
Cluster
0    2.245283
1   -1.291667
2   -0.787072
3    1.810219
Name: Like.n, dtype: float64

female:
Cluster
0    0.601724
1    0.421053
2    0.586957
3    0.476780
Name: Female, dtype: float64
# Assuming visit, like, and female are defined numpy arrays or lists
plt.scatter(visit, like, s=10 * female)  # cex in R becomes s (size) in Matplotlib
plt.xlim(1, 4.5)  # Set x-axis limits
plt.ylim(-3, 3)   # Set y-axis limits

# Add text labels 1 to 4 at each point
for i in range(4):
    plt.text(visit[i], like[i], str(i+1))

plt.show()

 
