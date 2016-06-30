namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System.Collections.Generic;
    using MathNet.Numerics.LinearAlgebra;

    public class CostResultSet
    {
        public double Cost { get; set; }

        public IList<Matrix<double>> Gradients { get; set; }
    }
}