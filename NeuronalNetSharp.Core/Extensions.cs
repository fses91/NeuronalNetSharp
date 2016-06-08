using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronalNetSharp.Core
{
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Complex;

    public static class Extensions
    {
        public static double CalculateNorm(this Matrix<double> matrix)
        {
            matrix = matrix.Map(d => Math.Pow(d, 2));
            return Math.Sqrt(matrix.ColumnSums().Sum());
        }
    }
}
