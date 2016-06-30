namespace NeuronalNetSharp.Core.NeuronalNetwork
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading.Tasks;
    using Import;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class NewNeuronalNetwork : INeuronalNetwork
    {
        public NewNeuronalNetwork(int sizeInputLayer, int sizeOutputLayer, int numberOfHiddenLayers = 0)
        {
            SizeInputLayer = sizeInputLayer;
            SizeOutputLayer = sizeOutputLayer;
            NumberOfHiddenLayers = numberOfHiddenLayers;

            Layers = new List<Matrix<double>>();
            Weights = new List<Matrix<double>>();
            Biases = new List<Matrix<double>>();

            InitializeLayers();
            InitializeBiases();
            InitializeWeights();
        }

        public IList<Matrix<double>> Biases { get; }

        public int SizeInputLayer { get; }

        public int SizeOutputLayer { get; }

        public int NumberOfHiddenLayers { get; }

        public IList<Matrix<double>> Layers { get; }

        public IList<Matrix<double>> Weights { get; }

        public CostResultSet ComputeCostResultSet(IList<IDataset> trainingData, IDictionary<string, Matrix> results, double lambda)
        {
            var costResultSet = new CostResultSet
            {
                Cost = ComputeCost(trainingData, results, lambda)
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
                Layers[i + 1] = Weights[i]*Layers[i];
                Layers[i + 1] = Layers[i + 1] + Biases[i];
                Layers[i + 1].MapInplace(Functions.SigmoidFunction);
            }

            return Layers.Last();
        }

        public void SetLayerSize(int layer, int size)
        {
            throw new NotImplementedException();
        }

        private void InitializeBiases()
        {
            Weights.Clear();
            var epsilon = Math.Sqrt(6)/Math.Sqrt(SizeInputLayer + SizeOutputLayer);

            for (var i = 0; i < Layers.Count - 1; i++)
                Biases.Add(DenseMatrix.CreateRandom(Layers[i + 1].RowCount, 1, new ContinuousUniform(-epsilon, epsilon)));
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

        private double ComputeCost(IList<IDataset> trainingData, IDictionary<string, Matrix> results, double lambda)
        {
            var cost = 0.0;
            var reg = 0.0;

            foreach (var dataset in trainingData)
            {
                var output = ComputeOutput(dataset.Data);
                var error = output - results[dataset.Label];
                cost += 1.0 / 2.0 * Math.Pow(error.L2Norm(), 2);
            }
            cost = 1.0 / trainingData.Count * cost;

            Parallel.ForEach(Weights, matrix =>
            {
                reg += matrix.Map(x => Math.Pow(x, 2)).RowSums().Sum();
            });
            reg = lambda / 2 * reg;

            return cost + reg;
        }

        private double ComputeGradients(IList<IDataset> trainingData, IDictionary<string, Matrix> results)
        {
            throw new NotImplementedException();
        }
    }
}