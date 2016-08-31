namespace NeuronalNetSharp.Core.Performance
{
    using System.Collections.Generic;
    using Import;
    using MathNet.Numerics.LinearAlgebra;
    using NeuronalNetwork;

    /// <summary>
    /// This class includes methods, to check the performance of a neuronal network.
    /// </summary>
    public static class NetworkTester
    {
        /// <summary>
        /// Tests the performance of a neuronal network.
        /// </summary>
        /// <param name="network">The network to test.</param>
        /// <param name="data">The data the network should be tested on.</param>
        /// <param name="labelMatrices">The wanted result matrix for a specific label.</param>
        /// <returns></returns>
        public static double TestNetwork(INeuronalNetwork network, IEnumerable<IDataset> data,
            IDictionary<string, Matrix<double>> labelMatrices)
        {
            var t = 0.0;
            var f = 0.0;

            foreach (var dataset in data)
            {
                var result = network.ComputeOutput(dataset.Data);
                var max = 0.0;
                var maxIndex = 0;

                for (var i = 0; i < result.RowCount; i++)
                {
                    if (result[i, 0] > max)
                    {
                        max = result[i, 0];
                        maxIndex = i;
                    }
                }

                if (labelMatrices[dataset.Label][maxIndex, 0] >= 1)
                    t += 1.0;
                else
                    f += 1.0;
            }

            return f/(f + t);
        }
    }
}