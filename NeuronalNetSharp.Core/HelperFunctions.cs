namespace NeuronalNetSharp.Core
{
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using Import;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public static class HelperFunctions
    {
        public static IList<Matrix<double>> InitializeMatricesWithSameDimensions(IList<Matrix<double>> matrices)
        {
            var result = new List<Matrix<double>>();

            foreach (var matrix in matrices)
                result.Add(new SparseMatrix(matrix.RowCount, matrix.ColumnCount));

            return result;
        }

        public static IDictionary<string, Matrix<double>> GetLabelMatrices(IEnumerable<IDataset> trainingData)
        {
            // Initialize Label Matrices
            IDictionary<string, Matrix<double>> lablesMatrices = new ConcurrentDictionary<string, Matrix<double>>();

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


