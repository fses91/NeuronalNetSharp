namespace NeuronalNetSharp.Core
{
    using System;
    using System.Collections.Generic;
    using Import;
    using Interfaces;

    public interface IOptimization
    {
        INeuronalNetwork OptimizeNetwork(
            INeuronalNetwork network, 
            IBackpropagation backpropagation,
            IList<IDataset> traingData,
            int iterations);

        event EventHandler IterationFinished;
    }
}