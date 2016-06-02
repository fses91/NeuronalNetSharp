using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuronalNetSharp.Core.Interfaces;

namespace NeuronalNetSharp.Core
{
    public class NeuronalNetwork : INeuronalNetwork
    {
        public NeuronalNetwork(int sizeInputLayer, int amountHiddenLayers, int sizeOutputLayer)
        {
            SizeInputLayer = sizeInputLayer;
            SizeOutputLayer = sizeOutputLayer;

            AmountHiddenLayers = amountHiddenLayers;
            HiddenLayers = new List<Matrix<double>>();

            Weights = new List<Matrix<double>>();

            InitializeWeights();
            InitializeLayers();
        }

        public bool Bias { get; set; }

        public int SizeInputLayer { get; }

        public int SizeOutputLayer { get; }

        public int AmountHiddenLayers { get; }

        public List<Matrix<double>> Weights { get; }

        public List<Matrix<double>> HiddenLayers { get; }

        /// <summary>
        ///     Initialize Layers.
        /// </summary>
        public void InitializeLayers()
        {
            for (var i = 0; i < AmountHiddenLayers; i++)
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

            for (var i = 0; i < AmountHiddenLayers; i++)
                Weights.Add(DenseMatrix.CreateRandom(SizeInputLayer, SizeInputLayer + 1,
                    new ContinuousUniform(-epsilon, epsilon)));

            Weights.Add(DenseMatrix.CreateRandom(SizeOutputLayer, SizeInputLayer + 1,
                new ContinuousUniform(-epsilon, epsilon)));
        }

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

            var outputLayer = Weights.Last()*HiddenLayers.Last();
            return outputLayer.Map(SpecialFunctions.Logistic);
        }
    }
}