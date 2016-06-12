namespace NeuronalNetSharp.Core.Interfaces
{
    using System;
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra.Double;

    public interface ILearningAlgorithm
    {
        IDictionary<string, Matrix> LabelMatrices { get; set; }
        INeuronalNetwork Network { get; set; }

        INeuronalNetwork TrainNetwork(int iterations, double alpha, double lambda, IList<IDataset> trainingData);

        double ComputeCost(IList<IDataset> trainingData);

        double ComputeCostRegularized(double lambda, IList<IDataset> trainingData);

        event EventHandler IterationFinished;
    }
}