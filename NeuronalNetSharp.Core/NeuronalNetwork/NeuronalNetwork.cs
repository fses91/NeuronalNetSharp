namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Import;
    using MathNet.Numerics;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class NeuronalNetwork
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

        public double ComputeCost(IList<IDataset> trainingData, double lambda)
        {
            var labelMatrices = GetLabelMatrices(trainingData);

            // Cost
            var result = 0.0;
            foreach (var dataset in trainingData)
            {
                var difference = ComputeOutput(dataset.Data) - labelMatrices[dataset.Label];
                var norm = difference.CalculateNorm();
                result += 1.0/2.0*Math.Pow(norm, 2);
            }

            result = result/trainingData.Count;

            var reg = 0.0;
            foreach (var weightVector in Weights)
            {
                reg +=
                    weightVector.SubMatrix(0, weightVector.RowCount, 1, weightVector.ColumnCount - 1)
                        .Map(d => Math.Pow(d, 2))
                        .ColumnSums()
                        .Sum();
            }

            reg = lambda/2.0*reg;

            return reg + result;
        }

        public IList<Matrix<double>> ComputeGradients(IList<IDataset> trainingData, double lambda)
        {
            var labelMatrices = GetLabelMatrices(trainingData);
            var deltaMatrices = InitilizeDeltaMatrices();

            foreach (var dataset in trainingData)
            {
                var tmpDeltaVectors = new List<Matrix<double>>();
                var output = ComputeOutput(dataset.Data);
                var deltaLast = output - labelMatrices[dataset.Label];
                //-(labelMatrices[dataset.Label] - output).PointwiseMultiply(output.PointwiseMultiply(1 - output));

                tmpDeltaVectors.Add(deltaLast);

                for (var j = Weights.Count - 1; j >= 1; j--)
                {
                    var tmp1 = Weights[j].Transpose()*tmpDeltaVectors.Last();
                    var tmp2 = HiddenLayers[j - 1].Map(d => d*(1 - d));
                    var delta = tmp1.PointwiseMultiply(tmp2);

                    tmpDeltaVectors.Add(delta.SubMatrix(1, delta.RowCount - 1, 0, delta.ColumnCount));
                }

                // For the input layer.
                tmpDeltaVectors.Reverse();
                deltaMatrices[0] = deltaMatrices[0] +
                                   tmpDeltaVectors[0]*DenseMatrix.Create(1, 1, 1).Append(dataset.Data.Transpose());

                for (var j = 1; j < deltaMatrices.Count; j++)
                    deltaMatrices[j] = deltaMatrices[j] + tmpDeltaVectors[j]*HiddenLayers[j - 1].Transpose();
            }

            Parallel.ForEach(deltaMatrices, matrix => matrix.Map(d => d/trainingData.Count));

            return deltaMatrices;
        }

        public Matrix<double> ComputeOutput(Matrix<double> input)
        {
            var currentLayer = DenseMatrix.Create(1, 1, 1).Stack(input); //     .Append(input.Transpose()).Transpose();

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
                Weights.Add(DenseMatrix.CreateRandom(SizeOutputLayer, SizeInputLayer + 1,
                    new ContinuousUniform(-epsilon, epsilon)));
                return;
            }

            for (var i = 0; i < NumberOfHiddenLayers; i++)
            {
                if (i <= 0)
                {
                    Weights.Add(DenseMatrix.CreateRandom(HiddenLayers[0].RowCount - 1, SizeInputLayer + 1,
                        new ContinuousUniform(-epsilon, epsilon)));
                    continue;
                }

                Weights.Add(DenseMatrix.CreateRandom(HiddenLayers[i].RowCount - 1, HiddenLayers[i - 1].RowCount,
                    new ContinuousUniform(-epsilon, epsilon)));
            }

            Weights.Add(DenseMatrix.CreateRandom(SizeOutputLayer, HiddenLayers.Last().RowCount,
                new ContinuousUniform(-epsilon, epsilon)));
        }

        private IList<Matrix<double>> InitilizeDeltaMatrices()
        {
            IList<Matrix<double>> deltaVectors = new List<Matrix<double>>();
            foreach (var weights in Weights)
                deltaVectors.Add(new SparseMatrix(weights.RowCount, weights.ColumnCount));

            return deltaVectors;
        }

        public static IDictionary<string, Matrix> GetLabelMatrices(IEnumerable<IDataset> trainingData)
        {
            // Initialize Label Matrices
            IDictionary<string, Matrix> lablesMatrices = new ConcurrentDictionary<string, Matrix>();

            var distinctLabels = trainingData.Select(x => x.Label).Distinct().ToList();
            for (var i = 0; i < distinctLabels.Count; i++)
            {
                var matrix = DenseMatrix.OfColumnArrays(new double[distinctLabels.Count()]);
                matrix[i, 0] = 1;
                lablesMatrices.Add(distinctLabels[i], matrix);
            }

            return lablesMatrices;
        }
    }
}