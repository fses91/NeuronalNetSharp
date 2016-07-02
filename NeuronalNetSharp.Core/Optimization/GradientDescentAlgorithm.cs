using NeuronalNetSharp.Core.NeuronalNetwork;

namespace NeuronalNetSharp.Core.Optimization
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using Import;
    using MathNet.Numerics.LinearAlgebra;

    public class GradientDescentAlgorithm : IOptimization
    {
        public GradientDescentAlgorithm(double lambda, double alpha)
        {
            Lambda = lambda;
            Alpha = alpha;
        }

        public double Alpha { get; set; }

        public double Lambda { get; set; }

        public event EventHandler IterationFinished;

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
                    new IterationStartedEventArgs
                    {
                        Cost = cost.Cost,
                        Iteration = i
                    });
            }

            return network;
        }
    }
}