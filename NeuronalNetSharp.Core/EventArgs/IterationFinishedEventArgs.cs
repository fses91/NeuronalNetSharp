namespace NeuronalNetSharp.Core.EventArgs
{
    using System;

    public class IterationFinishedEventArgs : EventArgs
    {
        public double Cost { get; set; }
        public int Iteration { get; set; }
    }
}