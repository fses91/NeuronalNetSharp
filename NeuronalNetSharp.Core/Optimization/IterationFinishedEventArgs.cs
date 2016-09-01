namespace NeuronalNetSharp.Core.Optimization
{
    using System;

    /// <summary>
    /// Event args if an iteration of an IOptimization algorithm is finished.
    /// </summary>
    public class IterationFinishedEventArgs : EventArgs
    {
        /// <summary>
        /// The cost of the network.
        /// </summary>
        public double Cost { get; set; }

        /// <summary>
        /// The iteration the network is in the moment.
        /// </summary>
        public int Iteration { get; set; }
    }
}