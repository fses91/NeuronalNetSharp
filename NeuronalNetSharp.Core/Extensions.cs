using System;
namespace NeuronalNetSharp.Core
{
    using System.Collections.Generic;
    using MathNet.Numerics.LinearAlgebra;

    public static class Extensions
    {
        public static double CalculateNorm(this Matrix<double> matrix)
        {
            matrix = matrix.Map(d => Math.Pow(d, 2));
            return Math.Sqrt(matrix.ColumnSums().Sum());
        }

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
