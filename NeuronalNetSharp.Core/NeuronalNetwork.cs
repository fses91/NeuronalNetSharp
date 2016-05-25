using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuronalNetSharp.Core
{
    public class NeuronalNetwork
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

        public int AmountHiddenLayers { get; set; }

        public List<Matrix<double>> Weights { get; }

        public List<Matrix<double>> HiddenLayers { get; }

        /// <summary>
        ///     Initialize Layers.
        /// </summary>
        public void InitializeLayers()
        {
            for (var i = 0; i < AmountHiddenLayers; i++)
                HiddenLayers.Add(DenseMatrix.OfColumnArrays(new double[SizeInputLayer + 1]));
        }

        /// <summary>
        ///     Initialize weights for a full connected network where all hidden layers have the same size.
        /// </summary>
        public void InitializeWeights()
        {
            // TODO überarbeiten, Überlegung wegen Weight initialisierung welchen Wert für Epsilon.
            var bias = Bias ? 1 : 0;
            var epsilon = Math.Sqrt(6)/Math.Sqrt(SizeInputLayer + SizeOutputLayer);

            for (var i = 0; i < AmountHiddenLayers; i++)
                Weights.Add(DenseMatrix.CreateRandom(SizeInputLayer, SizeInputLayer + 1,
                    new ContinuousUniform(-epsilon, epsilon)));

            Weights.Add(DenseMatrix.CreateRandom(SizeOutputLayer, SizeInputLayer + 1,
                new ContinuousUniform(-epsilon, epsilon)));
        }

        public Matrix<double> ComputeOutput(Matrix<double> input)
        {
            // TODO Überlegen wie man bias term anders adden kann.
            var currentLayer = DenseMatrix.Create(1, 1, 1).Append(input.Transpose()).Transpose();

            for (var i = 0; i < AmountHiddenLayers; i++)
            {
                HiddenLayers[i] = Weights[i]*currentLayer;
                HiddenLayers[i] = HiddenLayers[i].Map(SpecialFunctions.Logistic);
                currentLayer = DenseMatrix.Create(1, 1, 1).Append(HiddenLayers[i].Transpose()).Transpose();
            }

            var lastLayer = DenseMatrix.Create(1, 1, 1).Append(HiddenLayers.Last().Transpose()).Transpose();
            var output = Weights.Last()*lastLayer;
            return output.Map(SpecialFunctions.Logistic);
        }
    }
}