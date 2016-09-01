namespace NeuronalNetSharp.Core
{
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using Import;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// This class includes help functions for the framework.
    /// </summary>
    public static class HelperFunctions
    {
        /// <summary>
        /// Initialize a matrix with same dimensions of another matrix.
        /// </summary>
        /// <param name="matrices">The matrix you want to initilaize.</param>
        /// <returns>The new matrix.</returns>
        public static IList<Matrix<double>> InitializeMatricesWithSameDimensions(IList<Matrix<double>> matrices)
        {
            var result = new List<Matrix<double>>();

            foreach (var matrix in matrices)
                result.Add(new SparseMatrix(matrix.RowCount, matrix.ColumnCount));

            return result;
        }

        /// <summary>
        /// Get all possible labels out of the traing data.
        /// </summary>
        /// <param name="trainingData">The training data.</param>
        /// <returns>The distinct lables.</returns>
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

        /// <summary>
        /// Rescale a value into a specific value range.
        /// </summary>
        /// <param name="value">The value you want to rescale.</param>
        /// <param name="newMin">The minimum of the new range.</param>
        /// <param name="newMax">The maximum of the new range.</param>
        /// <param name="oldMin">The minimum of the old range.</param>
        /// <param name="oldMax">The maximum of the old range.</param>
        /// <returns>The value in the new range.</returns>
        public static double RescaleValue(double value, double newMin, double newMax, double oldMin, double oldMax)
        {
            return (value - oldMin) * (newMax - newMin) / (oldMax - oldMin) + newMin;
        }
    }
}


