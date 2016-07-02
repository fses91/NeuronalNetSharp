namespace NeuronalNetSharp.Core.Optimization
{
    using System;

    public class IterationStartedEventArgs : EventArgs
    {
        public double Cost { get; set; }

        public int Iteration { get; set; }
    }
}