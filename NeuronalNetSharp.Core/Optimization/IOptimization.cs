﻿namespace NeuronalNetSharp.Core.Optimization
{
    using System;
    using System.Collections.Generic;
    using Import;
    using Interfaces;

    public interface IOptimization
    {
        INeuronalNetwork OptimizeNetwork(
            INeuronalNetwork network, 
            IList<IDataset> traingData,
            int iterations);

        event EventHandler IterationFinished;
    }
}