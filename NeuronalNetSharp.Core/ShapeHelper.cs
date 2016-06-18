namespace NeuronalNetSharp.Core
{
    using System.Collections.Generic;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra;

    public static class ShapeHelper
    {
        public static double[] ShapeMatrices(IList<Matrix<double>> matrices)
        {
            var shapedMatrix = matrices[0].ToColumnWiseArray().ToList();
            for (var i = 0; i < matrices.Count - 1; i++)
                shapedMatrix = shapedMatrix.Concat(matrices[i + 1].ToColumnWiseArray()).ToList();

            return shapedMatrix.ToArray();
        }

        public static IList<Matrix<double>> ReshapeAndSetMatrices(double[] matrices, IList<Matrix<double>> matricesToSet)
        {
            var index = 0;

            foreach (var matrix in matricesToSet)
            {
                for (var i = 0; i < matrix.RowCount; i++)
                {
                    for (var j = 0; j < matrix.ColumnCount; j++)
                    {
                        matrix[i, j] = matrices[index];
                        index++;
                    }
                }
            }

            return matricesToSet;
        }
    }
}