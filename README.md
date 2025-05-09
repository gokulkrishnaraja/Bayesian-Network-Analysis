# Bayesian-Network-Analysis


In this small project, I compared network analysis versus traditional Bayesian inference in a simulated clinical dataset. The primary motivation was to understand how network analysis can be useful in the clinical data because a certain neurological condition often comes with comorbidities. Therefore, a graphical understanding is much more useful given the conditional (in)dependencies among the variables of interest. The Bayesian framework is more apt here given the frequentist procedures often fail to consider all the network dynamics and give wrong conclusions about the relationships between the nodes (variables). Please see () for thorough review of these problems.

### Simulation of the dataset:
I generated a dataset with the following variables: Attention, EEG beta power, Stress, Tremor, and Performance. The dependencies were simulated using a linear combination with gaussian noise (intrinsic to all psychological variables). The stress level was modeled as a function of attention (β = 0.5), while EEG beta power was influenced by both attention (β = 0.3) and stress (β = -0.4). Tremor was simulated to depend on stress (β = 0.4) and EEG beta activity (β = 0.3). Finally, task performance was modeled as being negatively affected by tremor (β = -0.5) and positively influenced by attention (β = 0.4). 

### After Network analysis:
The posterior probabilities showed strongest evidence (p = 1.00) for connections from Attention to Stress, EEG_Beta, Tremor, and Performance, perfectly matching our simulation where Attention was indeed a direct predictor of these variables (with β coefficients of 0.5, 0.3, and 0.4 respectively). The analysis also detected the relationship between Stress and EEG_Beta, and between Stress and Tremor with maximum certainty (p = 1.00), correctly identifying the simulated relationships (β = -0.4 and β = 0.4 respectively). The connection between Stress and Performance showed strong but not absolute evidence (p = 0.84), which aligns with our ground truth where Stress indirectly affected Performance through Tremor. The model also correctly identified the relationship between Tremor and Performance (p = 1.00), matching our simulated direct effect (β = -0.5). Overall, the Bayesian graph structure could recover most of the true relationships in our simulated data, with posterior probabilities reflecting the strength and certainty of these connections.

The network analysis revealed a highly connected structure with a density of 0.9, indicating that 90% of all possible connections are present in the network. The average degree of 3.6 suggests that each node maintains, on average, connections with approximately four other nodes. The network exhibits strong local clustering, as evidenced by a high clustering coefficient of 0.875, indicating substantial interconnectedness among neighboring nodes. The network consists of a single component (components = 1) containing all 5 nodes (largest_component_size = 5), demonstrating complete connectivity among all variables. The network diameter of 2 indicates that any two nodes in the network can be reached through at most two steps, suggesting efficient information flow and tight integration among the variables in the system.
BDgraph Analysis Report

### Model Information: 
- Number of nodes: 5 
- Number of samples: 1000 
- Number of iterations: 
- Burn-in: 

### Network Metrics:
- Density: 0.450 
- Average Degree: 3.600 
- Clustering Coefficient: 0.875 
- Number of Components: 1 
- Largest Component Size: 5 
- Diameter: 2 

| Node        | In-Degree | Out-Degree | Betweenness | Closeness  | Eigenvector |
|-------------|:---------:|:----------:|:-----------:|:----------:|:-----------:|
| Attention   | 0         | 4          | 0           | 0.2500    | 1.0000      |
| Stress      | 1         | 3          | 0           | 0.3333    | 1.0000      |
| EEG_Beta    | 2         | 1          | 0           | 0.3333    | 0.8229      |
| Tremor      | 3         | 1          | 1           | 1.0000    | 1.0000      |
| Performance | 3         | 0          | 0           | N/A       | 0.8229      |

### Key Findings
- Attention has the highest out-degree (4), suggesting it influences many other variables
- Tremor and Performance have the highest in-degree (3), indicating they are influenced by multiple variables
- Tremor is the only node with non-zero betweenness centrality
- Performance has no closeness centrality (N/A) due to its position in the network
- Attention, Stress, and Tremor show maximum eigenvector centrality (1.0000)

## Time Comparison Summary:
- BDGraph took: 0.74 seconds
- BRMS took: 370 seconds
- Difference: 369.26 seconds

## Average RMSE by variable and method:
|Variable| BDgraph_RMSE| BRMS_RMS|
|:------:|:------:|:------:|
|Stress  | 0.5055749 |0.5058432|
|EEG_Beta | 0.4886762 |0.4880422|
|Tremor  | 0.4961772 |0.4958868|
|Performance| 0.5015268 |0.5022121|

To conclude, this is nice way to showcase how comorbidities can exist in clinical data and how it can be extracted using Bayesian network analysis
which is found to be more efficient than the traditional way.
