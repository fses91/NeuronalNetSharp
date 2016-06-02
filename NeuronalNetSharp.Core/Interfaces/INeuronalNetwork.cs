using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuronalNetSharp.Core.Interfaces
{
    public interface INeuronalNetwork
    {
        int SizeInputLayer { get; }

        int SizeOutputLayer { get; }

        List<Matrix<double>> Weights { get; }

        List<Matrix<double>> HiddenLayers { get; }

        Matrix<double> ComputeOutput(Matrix<double> input);
    }
}
