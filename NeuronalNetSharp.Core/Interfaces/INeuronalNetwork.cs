using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuronalNetSharp.Core.Interfaces
{
    interface INeuronalNetwork
    {
        ICollection<Matrix<double>> Weights { get; }

        ICollection<Matrix<double>> HiddenLayers { get; }
    }
}
