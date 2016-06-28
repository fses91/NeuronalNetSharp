namespace NeuronalNetSharp.Core.Optimization
{
    using System;

    public class IterationFinishedEventArgs : EventArgs
    {
        public double Cost { get; set; }

        public int Iteration { get; set; }
    }
}