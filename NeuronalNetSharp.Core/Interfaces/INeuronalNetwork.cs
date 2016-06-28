using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuronalNetSharp.Core.Interfaces
{
    using System.Runtime.Serialization;
    using Import;

    public interface INeuronalNetwork
    {
        int SizeInputLayer { get; }

        int SizeOutputLayer { get; }

        IList<Matrix<double>> Weights { get; }

        IList<Matrix<double>> HiddenLayers { get; }

        Matrix<double> ComputeOutput(Matrix<double> input);

        void SetLayerSize(int layer, int size);

        double ComputeCost(IList<IDataset> trainingData, double lambda);

        IList<Matrix<double>> ComputeGradients(IList<IDataset> trainingData, double lambda);
    }
}
