namespace NeuronalNetSharp.Core
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Interfaces;
    using MathNet.Numerics;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class NeuronalNetwork : INeuronalNetwork
    {
        public NeuronalNetwork(int sizeInputLayer, int amountHiddenLayers, int sizeOutputLayer)
        {
            SizeInputLayer = sizeInputLayer;
            SizeOutputLayer = sizeOutputLayer;

            NumberOfHiddenLayers = amountHiddenLayers;
            HiddenLayers = new List<Matrix<double>>();

            Weights = new List<Matrix<double>>();

            InitializeLayers();
            InitializeWeights();
        }

        public int NumberOfHiddenLayers { get; }
        
        public Matrix<double> ComputeOutput(Matrix<double> input)
        {
            var currentLayer = DenseMatrix.Create(1, 1, 1).Append(input.Transpose()).Transpose();

            for (var i = 0; i < HiddenLayers.Count; i++)
            {
                HiddenLayers[i].SetSubMatrix(1, 0, Weights[i]*currentLayer);
                HiddenLayers[i].SetSubMatrix(1, 0,
                    HiddenLayers[i].SubMatrix(1, HiddenLayers[i].RowCount - 1, 0, HiddenLayers[i].ColumnCount)
                        .Map(SpecialFunctions.Logistic));

                currentLayer = HiddenLayers[i];
            }

            var outputlayer = Weights.Last()*HiddenLayers.Last();
            return outputlayer.Map(SpecialFunctions.Logistic);
        }

        public IList<Matrix<double>> HiddenLayers { get; }

        public void SetLayerSize(int layer, int size)
        {
            HiddenLayers[layer] = DenseMatrix.OfColumnArrays(new double[size + 1]);
            InitializeWeights();
        }

        public int SizeInputLayer { get; }

        public int SizeOutputLayer { get; }

        public IList<Matrix<double>> Weights { get; set; }

        /// <summary>
        ///     Initialize Layers.
        /// </summary>
        public void InitializeLayers()
        {
            for (var i = 0; i < NumberOfHiddenLayers; i++)
            {
                var matrix = DenseMatrix.OfColumnArrays(new double[SizeInputLayer + 1]);
                matrix[0, 0] = 1;
                HiddenLayers.Add(matrix);
            }
        }

        /// <summary>
        ///     Initialize weights for a full connected network where all hidden layers have the same size.
        /// </summary>
        public void InitializeWeights()
        {
            // TODO überarbeiten, Überlegung wegen Weight initialisierung welchen Wert für Epsilon.
            var epsilon = Math.Sqrt(6)/Math.Sqrt(SizeInputLayer + SizeOutputLayer);
            Weights.Clear();

            if (NumberOfHiddenLayers <= 0)
            {
                Weights.Add(DenseMatrix.CreateRandom(SizeOutputLayer, SizeInputLayer + 1, new ContinuousUniform(-epsilon, epsilon)));
                return;
            }

            for (var i = 0; i < NumberOfHiddenLayers; i++)
            {
                if (i <= 0)
                {
                    Weights.Add(DenseMatrix.CreateRandom(HiddenLayers[0].RowCount - 1, SizeInputLayer + 1, new ContinuousUniform(-epsilon, epsilon)));
                    continue;
                }

                Weights.Add(DenseMatrix.CreateRandom(HiddenLayers[i].RowCount - 1, HiddenLayers[i - 1].RowCount, new ContinuousUniform(-epsilon, epsilon)));
            }

            Weights.Add(DenseMatrix.CreateRandom(SizeOutputLayer, HiddenLayers.Last().RowCount,
                new ContinuousUniform(-epsilon, epsilon)));
        }
    }
}