namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System.Collections.Generic;
    using MathNet.Numerics.LinearAlgebra;

    /// <summary>
    /// The gradient result set of the neuronal network (cost of each network weight).
    /// </summary>
    public class GradientResultSet
    {
        /// <summary>
        /// The gradients of the network.
        /// </summary>
        public IList<Matrix<double>> Gradients { get; set; }

        /// <summary>
        /// The gradients of the bias terms of the network.
        /// </summary>
        public IList<Matrix<double>> BiasGradients { get; set; }
    }
}