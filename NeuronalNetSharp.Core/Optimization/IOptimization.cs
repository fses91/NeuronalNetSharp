using NeuronalNetSharp.Core.NeuronalNetwork;

namespace NeuronalNetSharp.Core.Optimization
{
    using System;
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;

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