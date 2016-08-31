namespace NeuronalNetSharp.Core.Optimization
{
    using System;
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;
    using NeuronalNetwork;

    /// <summary>
    /// Iterface for optimization algorithms.
    /// </summary>
    public interface IOptimization
    {
        INeuronalNetwork OptimizeNetwork(
            INeuronalNetwork network, 
            IList<IDataset> traingData,
            IDictionary<string, Matrix<double>> results,
            int iterations);

        event EventHandler IterationFinished;
    }
}