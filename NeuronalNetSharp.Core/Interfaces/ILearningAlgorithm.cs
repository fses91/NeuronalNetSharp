namespace NeuronalNetSharp.Core.Interfaces
{
    using System;
    using System.Collections.Generic;
    using Import;

    public interface ILearningAlgorithm
    {
        IEnumerable<IDataset> TrainingData { get; set; }

        INeuronalNetwork TrainNetwork(int iterations, double alpha, double lambda);

        double ComputeCost();

        double ComputeCostRegularized(double lambda);

        event EventHandler IterationFinished;
    }
}