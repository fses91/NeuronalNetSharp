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

                Parallel.For(0, network.Weights.Count, i1 =>
                {
                    network.Weights[i1] = network.Weights[i1] - Alpha*cost.Gradients.Gradients[i1];
                    network.BiasWeights[i1] = network.BiasWeights[i1] - Alpha*cost.Gradients.BiasGradients[i1];
                });

                IterationFinished?.Invoke(this,
                    new IterationFinishedEventArgs
                    {
                        Cost = network.ComputeCostResultSet(traingData, results ,Lambda).Cost,
                        Iteration = i
                    });
            }

            return network;
        }
    }
}