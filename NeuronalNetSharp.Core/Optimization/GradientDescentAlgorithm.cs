namespace NeuronalNetSharp.Core.Optimization
{
    using System;
    using System.Collections.Generic;
    using System.Linq.Expressions;
    using System.Threading.Tasks;
    using EventArgs;
    using Import;
    using Interfaces;

    public class GradientDescentAlgorithm : IOptimization
    {
        public GradientDescentAlgorithm(double lambda, double alpha)
        {
            Lambda = lambda;
            Alpha = alpha;
        }

        public double Lambda { get; set; }

        public double Alpha { get; set; }

        public INeuronalNetwork OptimizeNetwork(
            INeuronalNetwork network,
            IBackpropagation backpropagation,
            IList<IDataset> traingData,
            int iterations)
        {
            for (var i = 0; i < iterations; i++)
            {
                var deltaMatrices = backpropagation.ComputeDerivatives(network, traingData);

                Parallel.For(0, deltaMatrices.Count, j =>
                {
                    network.Weights[j] = network.Weights[j] - Alpha * deltaMatrices[j];

                    var subDelta =
                        deltaMatrices[j].SubMatrix(0, network.Weights[j].RowCount, 1, network.Weights[j].ColumnCount - 1)
                            .Map(d => Lambda/traingData.Count*d);
                    var subWeights = network.Weights[j].SubMatrix(0, network.Weights[j].RowCount, 1,
                        network.Weights[j].ColumnCount - 1);
                    network.Weights[j].SetSubMatrix(0, network.Weights[j].RowCount, 1,
                        network.Weights[j].ColumnCount - 1, subWeights + subDelta);
                });

                IterationFinished?.Invoke(this,
                    new IterationFinishedEventArgs
                    {
                        Cost = backpropagation.ComputeCostRegularized(network, traingData, Lambda),
                        Iteration = i
                    });
            }

            //var test = backpropagation.ComputeNumericalGradients(network, traingData, backpropagation, Lambda);

            return network;
        }

        public event EventHandler IterationFinished;
    }
}