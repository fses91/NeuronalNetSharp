namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;

    public interface INeuronalNetwork
    {
        /// <summary>
        /// The size of the input layer.
        /// </summary>
        int SizeInputLayer { get; }

        /// <summary>
        /// The size of the output layer.
        /// </summary>
        int SizeOutputLayer { get; }

        /// <summary>
        /// The number of hidden layers.
        /// </summary>
        int NumberOfHiddenLayers { get; }

        /// <summary>
        /// The bias weights of the network.
        /// </summary>
        IList<Matrix<double>> BiasWeights { get; }

        /// <summary>
        /// The weights of the network.
        /// </summary>
        IList<Matrix<double>> Weights { get; }

        /// <summary>
        /// The layers of the network.
        /// </summary>
        IList<Matrix<double>> Layers { get; }

        /// <summary>
        /// Compute the cost of the network.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <param name="results">The desired results for each label.</param>
        /// <param name="lambda">The lambda value of the network.</param>
        /// <returns>The cost result set.</returns>
        CostResultSet ComputeCostResultSet(IList<IDataset> trainingData, IDictionary<string, Matrix<double>> results, double lambda = 0);

        /// <summary>
        /// Calcultes the output for a given input.
        /// </summary>
        /// <param name="input">Input for the network</param>
        /// <returns>Output for given input.</returns>
        Matrix<double> ComputeOutput(Matrix<double> input);

        /// <summary>
        /// Set the size of a specific layer in the network.
        /// </summary>
        /// <param name="layer">The layer you want to change.</param>
        /// <param name="size">The new size.</param>
        void SetLayerSize(int layer, int size);
    }
}