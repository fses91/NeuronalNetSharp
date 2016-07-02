namespace NeuronalNetSharp.Core
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using Import;
    using MathNet.Numerics.LinearAlgebra.Double;

    public static class HelperFunctions
    {
        public static double SigmoidFunction(double x)
        {
            return 1/(1 + Math.Exp(-x));
        }

        public static IDictionary<string, Matrix> GetLabelMatrices(IEnumerable<IDataset> trainingData)
        {
            // Initialize Label Matrices
            IDictionary<string, Matrix> lablesMatrices = new ConcurrentDictionary<string, Matrix>();

            var distinctLabels = trainingData.Select(x => x.Label).Distinct().ToList();
            for (var i = 0; i < distinctLabels.Count; i++)
            {
                var matrix = DenseMatrix.OfColumnArrays(new double[distinctLabels.Count()]);
                matrix[i, 0] = 1;
                lablesMatrices.Add(distinctLabels[i], matrix);
            }

            return lablesMatrices;
        }
    }
}


