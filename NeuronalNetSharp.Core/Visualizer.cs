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
        public static byte[] VisualizeNetwork(INeuronalNetwork network, IBackpropagation backprob, int layer, int unit, int width, int height, double alpha, int iterations)
        {
            var matrix = DenseMatrix.CreateRandom(network.SizeInputLayer, 1, new ContinuousUniform(0, 255));

            for (var i = 0; i < iterations; i++)
            {
                var activationAtUnit = network.HiddenLayers[layer][unit, 0];

                for (var j = 0; j < matrix.RowCount; j++)
                {
                    var tmp0 = alpha*(activationAtUnit/matrix[j, 0]);
                    var tmp1 = matrix[j, 0] + tmp0;
                    matrix[j, 0] = tmp1;
                }
            }

            var array = matrix.ToRowWiseArray().Select(Convert.ToInt32);

            return array.Select(x => (byte) x).ToArray();
        }
    }
}