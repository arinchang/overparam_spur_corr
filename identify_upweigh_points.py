"""
- run a round of model inference to compute learned embeddings for all data points, and measure distance 
to the average embedding of minority group. keep track of the top k closest points
- form a list containing the weights for each data point to be passed into WeightedRandomSampler
"""
   