namespace NeuronalNetSharp.Core
{
    using System.Collections.Generic;
    using MathNet.Numerics.LinearAlgebra;
    using System;

    /// <summary>
    /// This class includes extension methods.
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// Calculate the norm of the matrix, the matrix must have only one column.
        /// </summary>
        /// <param name="matrix">The matrix.</param>
        /// <returns>Norm of the matrix.</returns>
        public static double CalculateNorm(this Matrix<double> matrix)
        {
            if (matrix.ColumnCount != 1) throw new ArgumentException("The matrix must contain only one column.");
            matrix = matrix.Map(d => Math.Pow(d, 2));
            return Math.Sqrt(matrix.ColumnSums().Sum());
        }

        /// <summary>
        /// Shuffels the objects in a list.
        /// </summary>
        /// <typeparam name="T">The type.</typeparam>
        /// <param name="list">The name.</param>
        public static void Shuffle<T>(this IList<T> list)
        {
            var rng = new Random();

            var n = list.Count;
            while (n > 1)
            {
                n--;
                var k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }
}
