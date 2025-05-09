
library(BDgraph)
library(tidyverse)
library(igraph)
library(ggplot2)
library(reshape2)
library(igraph)
# -----------------------------------------------------------------------
# Setting SEED
set.seed(123)

#Step 1: Simulating DATA
# We have 5 variables of interest: Attention, Stress, EEG Beta power, Tremor, and Performance

n_samples <- 1000
data      <- data.frame(
                        Attention = rnorm(n_samples, mean = 0, sd = 1),
                        Stress = NA,
                        EEG_Beta = NA,
                        Tremor = NA,
                        Performance = NA
                      )

# Making the variables dependent
data$Stress      <- 0.5 * data$Attention +                        rnorm(n_samples, 0, 0.5)
data$EEG_Beta    <- 0.3 * data$Attention - 0.4 * data$Stress    + rnorm(n_samples, 0, 0.5)
data$Tremor      <- 0.4 * data$Stress    + 0.3 * data$EEG_Beta  + rnorm(n_samples, 0, 0.5)
data$Performance <- -0.5 * data$Tremor   + 0.4 * data$Attention + rnorm(n_samples, 0, 0.5)
data_matrix      <- as.matrix(data)

# -----------------------------------------------------------------------

# STEP 2: Fitting a Bayesian Graphical Modele
# gcgm      - A method for graph estimation (gaussian copula graphical method)
# iter      - Number of iterations for MCMC sampling
# burnin    - Number of initial iterations to disregard
# save      - Whether to save the samples
# g.start   - Starting graph structure
# cores     - Number of cores
# threshold - Threshold for edge inclusion

bdgraph_obj <- bdgraph(
  data = data_matrix,
  method = "gcgm",    
  iter = 5000,        
  burnin = 1000,      
  save = TRUE,
  g.start = "empty",      
  cores = 2,             
  threshold = 0.5        
)       

            # Printin summary
            print(summary(bdgraph_obj))

            # SEE README FOR RESULTS

# ----------------------------------------------------------------------- 
            
# STEP 3: Getting posterior probability matrix
prob_matrix  <- plinks(bdgraph_obj)
print("Posterior Probability Matrix:")
print(round(prob_matrix, 3))

            # SEE README FOR RESULTS

# Heat map of posterior probabilities
# Posterior probability measures the likelihood of the presence 
# of an edge between two variables considering both prior and 
# the evidence

prob_df <- melt(prob_matrix)
p1      <- ggplot(prob_df, aes(x = Var1, y = Var2, fill = value)) +
                  geom_tile() +
                  scale_fill_gradient2(low = "white", high = "red", mid = "pink", 
                  midpoint = 0.5, limit = c(0,1)) +
                  theme_minimal() +
                  labs(title = "Posterior Probability of Edges",
                  x = "", y = "", fill = "Probability") +
                  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)

# Extracting the Selected graph structure
selected_graph <-  BDgraph::select(bdgraph_obj)
print("Selected Graph Structure:")
print(selected_graph)

# Plotting the graph Using qgraph
library(qgraph)
qgraph(selected_graph,
       layout       = "circle",
       labels       = colnames(data_matrix),
       edge.color   = "darkblue",
       node.color   = "lightblue",
       border.color = "darkblue",
       border.width = 2,
       edge.width   = 1.5,
       directed     = TRUE,    # =set to TRUE for directed edges
       arrows       = TRUE,    # Show arrows for directed edges
       title        = "Selected Graph Structure")

# -----------------------------------------------------------------------

# STEP 4:Extracting Network Metrics

# Density (edge_density): Calculates the proportion of actual connections
# relative to all possible connections in the network, indicating how densely 
# connected the network is.

# Average Degree (mean(degree)): Computes the mean number of connections 
# per node, representing the average connectivity of nodes in the network.
 
# Clustering Coefficient (transitivity): Measures the global transitivity 
# of the network, indicating the degree to which nodes tend to cluster together. 
# This is calculated as the ratio of triangles (three fully connected nodes) to 
# connected triples in the graph.

# Create igraph object
library(igraph)
g <- graph_from_adjacency_matrix(selected_graph, 
                                 mode = "directed", 
                                 weighted = TRUE)
network_metrics <- list()
                       
# Basic metrics
network_metrics$density         <- edge_density(g)
network_metrics$avg_degree      <- mean(igraph::degree(g))
network_metrics$clustering_coef <- transitivity(g, type = "global")

# degree distributions
network_metrics$in_degree       <- igraph::degree(g, mode = "in")
network_metrics$out_degree      <- igraph::degree(g, mode = "out")

# component information
comp                                   <- components(g, mode = "weak")
network_metrics$components             <- comp$no
network_metrics$largest_component_size <- max(comp$csize)

# Calculate diameter
# Calculate network metrics
network_metrics <- list()

# Calculate basic metrics
network_metrics$density <- edge_density(g)
network_metrics$avg_degree <- mean(igraph::degree(g))
network_metrics$clustering_coef <- transitivity(g, type = "global")

# Calculate degree distributions
network_metrics$in_degree <- igraph::degree(g, mode = "in")
network_metrics$out_degree <- igraph::degree(g, mode = "out")

# Calculate component information
comp                                   <- components(g, mode = "weak")
network_metrics$components             <- comp$no
network_metrics$largest_component_size <- max(comp$csize)

# Calculate diameter
network_metrics$diameter <- diameter(g, directed = TRUE)


# Print
cat("\nNetwork Metrics:\n")
cat("----------------------------------------\n")
cat("Density:",                sprintf("%.3f", network_metrics$density), "\n")
cat("Average Degree:",         sprintf("%.3f", network_metrics$avg_degree), "\n")
cat("Clustering Coefficient:", sprintf("%.3f", network_metrics$clustering_coef), "\n")
cat("Number of Components:",   network_metrics$components, "\n")
cat("Largest Component Size:", network_metrics$largest_component_size, "\n")
cat("Diameter:",               network_metrics$diameter, "\n")

cat("\nDegree Distributions:\n")
cat("----------------------------------------\n")
cat("In-Degree:\n")
print(network_metrics$in_degree)
cat("\nOut-Degree:\n")
print(network_metrics$out_degree)

# Additional centrality measures
cat("\nCentrality Measures:\n")
cat("----------------------------------------\n")
network_metrics$betweenness <- betweenness(g)
network_metrics$closeness <- closeness(g)
network_metrics$eigenvector <- eigen_centrality(g)$vector

cat("\nBetweenness Centrality:\n")
print(network_metrics$betweenness)
cat("\nCloseness Centrality:\n")
print(network_metrics$closeness)
cat("\nEigenvector Centrality:\n")
print(network_metrics$eigenvector)

# # Save 
# results <- list(
#                 model              = bdgraph_obj,
#                 probability_matrix = prob_matrix,
#                 selected_graph     = selected_graph,
#                 network_metrics    = network_metrics
# )
# 
# saveRDS(results, "bdgraph_results.rds")

# ______________________________________________________________________________
# Comparing both BDgraph and BRMS methods --------------------------------------
# ______________________________________________________________________________

library(brms)
library(BDgraph)
library(bnlearn)
library(tidyverse)
library(caret)

# Function to simulate data
simulate_data <- function(n_samples) {
    data <- data.frame(
                        Attention = rnorm(n_samples, mean = 0, sd = 1),
                        Stress = NA,
                        EEG_Beta = NA,
                        Tremor = NA,
                        Performance = NA
                       )
  
  data$Stress       <- 0.5 * data$Attention + rnorm(n_samples, 0, 0.5)
  data$EEG_Beta     <- 0.3 * data$Attention - 0.4 * data$Stress + rnorm(n_samples, 0, 0.5)
  data$Tremor       <- 0.4 * data$Stress    + 0.3 * data$EEG_Beta + rnorm(n_samples, 0, 0.5)
  data$Performance  <- -0.5 * data$Tremor   + 0.4 * data$Attention + rnorm(n_samples, 0, 0.5)
  return(data)
}

predict_bdgraph <- function(train_data, test_data) {
  
  bdg_fit              <- bdgraph(as.matrix(train_data), method = "gcgm") # Fitting BDgraph
  
  adj_matrix           <- (bdg_fit$p_links > 0.5) * 1
  colnames(adj_matrix) <- rownames(adj_matrix) <- names(train_data)
  
  dag                  <- empty.graph(nodes = names(train_data))
  amat(dag)            <- adj_matrix
  
  bn_fit               <- bn.fit(dag, train_data, method = "mle-g")
  
  # Predict each variable
  predictions          <- data.frame(matrix(NA, nrow = nrow(test_data), 
                                   ncol = ncol(test_data)))
  names(predictions)   <- names(test_data)
  
  for(var in names(test_data)) {
    parents <- parents(dag, var)
    if(length(parents) > 0) {
      pred <- predict(bn_fit, node = var, data = test_data[parents])
      predictions[[var]] <- pred
    } else {
      predictions[[var]] <- mean(train_data[[var]])
    }
  }
  return(predictions)
}


# Predict using BRMS
predict_brms <- function(train_data, test_data) {
      # Fit models
      stress_model      <- brm(Stress ~ Attention, data = train_data)
      eeg_model         <- brm(EEG_Beta ~ Attention + Stress, data = train_data)
      tremor_model      <- brm(Tremor ~ Stress + EEG_Beta, data = train_data)
      performance_model <- brm(Performance ~ Tremor + Attention, data = train_data)
      
      # Make predictions
      predictions <- data.frame(
        Attention   = test_data$Attention,
        Stress      = predict(stress_model, newdata = test_data)[,1],
        EEG_Beta    = predict(eeg_model, newdata = test_data)[,1],
        Tremor      = predict(tremor_model, newdata = test_data)[,1],
        Performance = predict(performance_model, newdata = test_data)[,1]
      )
      return(predictions)
}

#k-fold cross-validation
compare_methods <- function(n_samples = 1000, k_folds = 5) {
  data          <- simulate_data(n_samples)
  folds         <- createFolds(1:nrow(data), k = k_folds)
  
  results <- list(
    bdgraph = matrix(NA, nrow = k_folds, ncol = 4),  # 4 predicted variables
    brms    = matrix(NA, nrow = k_folds, ncol = 4)
  )
  colnames(results$bdgraph) <- colnames(results$brms) <- 
    c("Stress", "EEG_Beta", "Tremor", "Performance")
  
  for(i in 1:k_folds) {
    test_idx   <- folds[[i]]
    train_data <- data[-test_idx,]
    test_data  <- data[test_idx,]
    
    bdg_pred  <- predict_bdgraph(train_data, test_data)
    brms_pred <- predict_brms(train_data, test_data)
    
    # Calculate RMSE
    vars_to_predict <- c("Stress", "EEG_Beta", "Tremor", "Performance")
    for(j in seq_along(vars_to_predict)) {
      var                  <- vars_to_predict[j]
      results$bdgraph[i,j] <- sqrt(mean((test_data[[var]] - bdg_pred[[var]])^2))
      results$brms[i,j]    <- sqrt(mean((test_data[[var]] - brms_pred[[var]])^2))
    }
  }
  
  # Summarize results
  summary <- data.frame(
    Variable     = vars_to_predict,
    BDgraph_RMSE = colMeans(results$bdgraph),
    BRMS_RMSE    = colMeans(results$brms)
  )
  
  return(list(
    fold_results = results,
    summary      = summary
  ))
}

# Run comparison
set.seed(123)
results <- compare_methods(n_samples = 1000, k_folds = 5)

print("Average RMSE by variable and method:")
print(results$summary)
