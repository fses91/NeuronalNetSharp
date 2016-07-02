namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.InteropServices.ComTypes;
    using System.Threading;
    using System.Threading.Tasks;
    using Import;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class NeuronalNetwork : INeuronalNetwork
    {
        public NeuronalNetwork(int sizeInputLayer, int sizeOutputLayer, int numberOfHiddenLayers = 0, double lambda = 0.01)
        {
            SizeInputLayer = sizeInputLayer;
            SizeOutputLayer = sizeOutputLayer;
            NumberOfHiddenLayers = numberOfHiddenLayers;
            Lambda = lambda;

            Layers = new List<Matrix<double>>();
            Weights = new List<Matrix<double>>();
            BiasWeights = new List<Matrix<double>>();

            InitializeLayers();
            InitializeBiases();
            InitializeWeights();
        }

        public IList<Matrix<double>> BiasWeights { get; }

        public int SizeInputLayer { get; }

        public int SizeOutputLayer { get; }

        public int NumberOfHiddenLayers { get; }

        public double Lambda { get; }

        public IList<Matrix<double>> Layers { get; }

        public IList<Matrix<double>> Weights { get; }

        public CostResultSet ComputeCostResultSet(IList<IDataset> trainingData, IDictionary<string, Matrix<double>> results, double lambda)
        {
            var costResultSet = new CostResultSet
            {
                Cost = ComputeCost(trainingData, results, lambda),
                Gradients = ComputeGradients(trainingData, results)
            };

            return costResultSet;
        }

        public Matrix<double> ComputeOutput(Matrix<double> input)
        {
            if (input.RowCount != SizeInputLayer || input.ColumnCount > 1)
                throw new ArgumentException("Dimensions have to agree with the size of the input layer.");

            Layers[0] = input;

            for (var i = 0; i < Weights.Count; i++)
            {
                Layers[i + 1] = Weights[i]*Layers[i] + BiasWeights[i];
                Layers[i + 1].MapInplace(Functions.SigmoidFunction);
            }

            return Layers.Last();
        }

        public void SetLayerSize(int layer, int size)
        {
            if (layer == 0 || layer == Layers.Count - 1)
                throw new ArgumentException("You can't resize the input or output layer.");

            Layers[layer] = DenseMatrix.OfColumnArrays(new double[size]);

            InitializeBiases();
            InitializeWeights();
        }

        private void InitializeBiases()
        {
            BiasWeights.Clear();
            var epsilon = Math.Sqrt(6)/Math.Sqrt(SizeInputLayer + SizeOutputLayer);

            for (var i = 0; i < Layers.Count - 1; i++)
                BiasWeights.Add(DenseMatrix.CreateRandom(Layers[i + 1].RowCount, 1, new ContinuousUniform(-epsilon, epsilon)));
        }

        private void InitializeLayers()
        {
            Layers.Add(DenseMatrix.OfColumnArrays(new double[SizeInputLayer]));

            for (var i = 0; i < NumberOfHiddenLayers; i++)
                Layers.Add(DenseMatrix.OfColumnArrays(new double[SizeInputLayer]));

            Layers.Add(DenseMatrix.OfColumnArrays(new double[SizeOutputLayer]));
        }

        private void InitializeWeights()
        {
            Weights.Clear();
            var epsilon = Math.Sqrt(6)/Math.Sqrt(SizeInputLayer + SizeOutputLayer);

            for (var i = 0; i < NumberOfHiddenLayers; i++)
                Weights.Add(DenseMatrix.CreateRandom(Layers[i + 1].RowCount, Layers[i].RowCount,
                    new ContinuousUniform(-epsilon, epsilon)));

            Weights.Add(DenseMatrix.CreateRandom(SizeOutputLayer, Layers.Reverse().Skip(1).FirstOrDefault().RowCount,
                new ContinuousUniform(-epsilon, epsilon)));
        }

        private double ComputeCost(IList<IDataset> trainingData, IDictionary<string, Matrix<double>> results, double lambda)
        {
            var cost = 0.0;
            var reg = 0.0;

            foreach (var dataset in trainingData)
            {
                var output = ComputeOutput(dataset.Data);
                var error = output - results[dataset.Label];
                cost += 1.0 / 2.0 * Math.Pow(error.CalculateNorm(), 2);
            }
            cost = 1.0 / trainingData.Count * cost;

            Parallel.ForEach(Weights, matrix =>
            {
                reg += matrix.Map(x => Math.Pow(x, 2)).RowSums().Sum();
            });
            reg = lambda / 2 * reg;

            return cost + reg;
        }

        private GradientResultSet ComputeGradients(IList<IDataset> trainingData, IDictionary<string, Matrix<double>> results)
        {
            var deltas = HelperFunctions.InitializeMatricesWithSameDimensions(Weights);
            var biasDeltas = HelperFunctions.InitializeMatricesWithSameDimensions(BiasWeights);

            foreach (var dataset in trainingData)
            {
                var tmpDeltas = new List<Matrix<double>>();

                var output = ComputeOutput(dataset.Data);
                var error = -(results[dataset.Label] - output).PointwiseMultiply(output.Map(d => d*(1 - d)));
                tmpDeltas.Add(error);

                for (var i = Weights.Count - 1; i > 0; i--)
                {
                    var delta = (Weights[i].Transpose()*tmpDeltas.Last()).PointwiseMultiply(Layers[i].Map(d => d*(1 - d)));
                    tmpDeltas.Add(delta);
                }
                tmpDeltas.Reverse();

                for (var i = 0; i < tmpDeltas.Count; i++)
                {
                    deltas[i] = deltas[i] + tmpDeltas[i]*Layers[i].Transpose();
                    biasDeltas[i] = biasDeltas[i] + tmpDeltas[i];
                }
            }

            Parallel.ForEach(deltas, matrix => matrix.MapInplace(v => 1.0/trainingData.Count*v));
            Parallel.ForEach(biasDeltas, matrix => matrix.MapInplace(v => 1.0/trainingData.Count*v));

            return new GradientResultSet { BiasGradients = biasDeltas, Gradients = deltas};
        }
    }
}