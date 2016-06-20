namespace NeuronalNetSharp.Core
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Import;
    using Interfaces;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public static class Visualizer
    {
        public static List<byte[]> VisualizeNetworkLayer(INeuronalNetwork network, IBackpropagation backprob, int layer, int width, int height, int iterations)
        {
            const int newMin = 0;
            const int newMax = 255;
            var list = new List<byte[]>();
            var matrix = DenseMatrix.CreateRandom(network.SizeInputLayer, 1, new ContinuousUniform(0, 255));
            var weights = network.Weights[layer];
            var squaredSum = Math.Sqrt(weights.Map(d => Math.Pow(d, 2)).ColumnSums().Sum());


            for (var i = 0; i < weights.RowCount; i++)
            {
                var activationValues = new double[weights.ColumnCount - 1];
                for (var j = 0; j < weights.ColumnCount - 1; j++)
                {
                    activationValues[j] = weights[i, j]/squaredSum;
                }

                var oldMax = activationValues.Max();
                var oldMin = activationValues.Min();

                // Scale between 0 - 255.
                        for (var j = 0; j < activationValues.Length; j++)
                            activationValues[j] = (activationValues[j] - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;
                var intTmp = activationValues.Select(Convert.ToInt32);
                var bytes = intTmp.Select(x => (byte)x).ToArray();


                list.Add(bytes);
            }

            return list;
        }
    }
}