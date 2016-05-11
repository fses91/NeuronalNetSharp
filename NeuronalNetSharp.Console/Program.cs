using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronalNetSharp.Console
{
    class Program
    {
        static void Main(string[] args)
        {
            var A = DenseMatrix.OfArray(new double[,] {
                    {1,1,1},
                    {1,2,3},
                    {4,3,2}});

            var B = DenseMatrix.OfArray(new double[,] {
                    {1,1,1},
                    {1,2,4},
                    {4,3,1}});

            var C = A * B;

        }
    }
}
