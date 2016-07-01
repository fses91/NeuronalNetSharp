namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public interface INeuronalNetwork
    {
        int SizeInputLayer { get; }

        int SizeOutputLayer { get; }

        int NumberOfHiddenLayers { get; }

        IList<Matrix<double>> BiasWeights { get; }

        IList<Matrix<double>> Weights { get; }

        IList<Matrix<double>> Layers { get; }

        CostResultSet ComputeCostResultSet(IList<IDataset> trainingData, IDictionary<string, Matrix<double>> results, double lambda);

        Matrix<double> ComputeOutput(Matrix<double> input);

        void SetLayerSize(int layer, int size);
    }
}