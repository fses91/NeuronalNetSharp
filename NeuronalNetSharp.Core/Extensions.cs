using System;
namespace NeuronalNetSharp.Core
{
    using MathNet.Numerics.LinearAlgebra;

    public static class Extensions
    {
        public static double CalculateNorm(this Matrix<double> matrix)
        {
            matrix = matrix.Map(d => Math.Pow(d, 2));
            return Math.Sqrt(matrix.ColumnSums().Sum());
        }
    }
}
