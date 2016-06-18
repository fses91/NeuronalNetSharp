namespace NeuronalNetSharp.Core.Interfaces
{
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;

    public interface IBackpropagation
    {
        double ComputeCost(INeuronalNetwork network, IList<IDataset> trainingData);

        double ComputeCostRegularized(INeuronalNetwork network, IList<IDataset> trainingData, double lambda);

        IList<double> ComputeNumericalGradients(INeuronalNetwork network, IList<IDataset> traingData, IBackpropagation backprob, double lambda);

        IList<Matrix<double>> ComputeDerivatives(INeuronalNetwork network, IList<IDataset> trainingData);
    }
}