namespace NeuronalNetSharp.Core.Optimization
{
    using System;
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;
    using NeuronalNetwork;

    public class GradientDescentAlgorithm : IOptimization
    {
        /// <summary>
        /// Initilizes a new instance of GradientDescentAlgorithm. 
        /// </summary>
        /// <param name="lambda">The lambda value to use.</param>
        /// <param name="alpha">The alpha value to use.</param>
        public GradientDescentAlgorithm(double lambda, double alpha)
        {
            Lambda = lambda;
            Alpha = alpha;
        }

        /// <summary>
        /// Gets or sets the alpha value.
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Gets or sets the lambda value.
        /// </summary>
        public double Lambda { get; set; }

        /// <summary>
        /// Event handler if an iteration is finished.
        /// </summary>
        public event EventHandler IterationFinished;

        /// <summary>
        /// Optimize the weights of a neuronal network.
        /// </summary>
        /// <param name="network">The neuronal network.</param>
        /// <param name="traingData">The training data.</param>
        /// <param name="results">The desired outputs for each label.</param>
        /// <param name="iterations">The number of iteraions.</param>
        /// <returns>An optimized neuronal network.</returns>
        public INeuronalNetwork OptimizeNetwork(INeuronalNetwork network, IList<IDataset> traingData, IDictionary<string, Matrix<double>> results, int iterations)
        {
            for (var i = 0; i < iterations; i++)
            {
                var cost = network.ComputeCostResultSet(traingData, results, Lambda);

                for (var j = 0; j < network.Weights.Count; j++)
                {
                    network.Weights[j] = network.Weights[j] - Alpha * (cost.Gradients.Gradients[j] + Lambda*network.Weights[j]);
                    network.BiasWeights[j] = network.BiasWeights[j] - Alpha * cost.Gradients.BiasGradients[j];
                }

                IterationFinished?.Invoke(this,
                    new IterationFinishedEventArgs
                    {
                        Cost = cost.Cost,
                        Iteration = i
                    });
            }

            return network;
        }
    }
}