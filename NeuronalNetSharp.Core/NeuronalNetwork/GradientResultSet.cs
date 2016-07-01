namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System.Collections.Generic;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class GradientResultSet
    {
        public IList<Matrix<double>> Gradients { get; set; }

        public IList<Matrix<double>> BiasGradients { get; set; }
    }
}