namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    /// <summary>
    /// The cost result set of the neuronal network.
    /// </summary>
    public class CostResultSet
    {
        /// <summary>
        /// The cost of the network.
        /// </summary>
        public double Cost { get; set; }

        /// <summary>
        /// The gradients of the network.
        /// </summary>
        public GradientResultSet Gradients { get; set; }
    }
}