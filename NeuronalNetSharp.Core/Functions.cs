namespace NeuronalNetSharp.Core
{
    using System;

    /// <summary>
    /// This class includes activation functions.
    /// </summary>
    public static class Functions
    {
        /// <summary>
        /// The sigmoid function.
        /// </summary>
        /// <param name="x">Input value.</param>
        /// <returns>The calculated value.</returns>
        public static double SigmoidFunction(double x)
        {
            return 1.0/(1.0 + Math.Exp(-x));
        }
    }
}