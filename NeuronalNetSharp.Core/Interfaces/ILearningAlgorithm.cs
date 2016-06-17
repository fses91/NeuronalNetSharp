namespace NeuronalNetSharp.Core.Interfaces
{
    using System;
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public interface ILearningAlgorithm
    {
        IDictionary<string, Matrix> LabelMatrices { get; set; }

        IList<Matrix<double>> ComputeDerivatives(IList<IDataset> trainingData);

        INeuronalNetwork Network { get; set; }

        INeuronalNetwork TrainNetwork(int iterations, double alpha, double lambda, IList<IDataset> trainingData);

        double ComputeCost(IList<IDataset> trainingData);

        double ComputeCostRegularized(IList<IDataset> trainingData, double lambda);

        event EventHandler IterationFinished;
    }
}