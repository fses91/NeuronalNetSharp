namespace NeuronalNetSharp.Core.LearningAlgorithms
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Import;
    using Interfaces;
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Double;

    public class BackpropagationLearningAlgorithm
    {
        public BackpropagationLearningAlgorithm(INeuronalNetwork network, IEnumerable<IDataset> traningData)
        {
            Network = network;
            LabelMatrices = new Dictionary<string, Matrix>();
            TrainingData = traningData;

            InitilizeLabelMatrices();
        }

        public IDictionary<string, Matrix> LabelMatrices { get; set; }

        public INeuronalNetwork Network { get; set; }

        public IEnumerable<IDataset> TrainingData { get; set; }

        public INeuronalNetwork TrainNetwork(int iterations)
        {
            IList<Matrix<double>> deltaVectors = new List<Matrix<double>>();
            foreach (var dataset in TrainingData)
            {
                var output = Network.ComputeOutput(dataset.Data);
                var deltaLast = output - LabelMatrices[dataset.Label];
                deltaVectors.Add(deltaLast);


                // Beispielimplementierung für 1 Layer, TODO umsetzten auf n - Layer.
                var tmp1 = Network.Weights[1].Transpose()*deltaLast;
                var tmp2 = Network.HiddenLayers[0].Map(d => d*(1 - 1));
                var delta2 = tmp1.PointwiseMultiply(tmp2);


                //for (var i = Network.Weights.Count - 2; i >= 0; i++)
                //{
                //    var test = Network.Weights[i + 1].Transpose()*deltaVectors.Last();
                //    var test2 = Network.HiddenLayers[i].Map(d => d*(1 - d));


                //    deltaVectors.Add((Network.Weights[i + 1].Transpose() * deltaVectors.Last()).PointwiseMultiply(Network.HiddenLayers[i].Map(d => d*(1 - d))));
                //}


                foreach (var layer in Network.HiddenLayers.Reverse())
                {
                    deltaVectors.Add(layer.Transpose() * deltaVectors.Last());
                }
            }

            

            throw new NotImplementedException();

        }

        public double ComputeCostRegularized(double lambda)
        {
            // Calculate cost.
            var cost = 0.0;
            foreach (var dataset in TrainingData)
            {
                var result = Network.ComputeOutput(dataset.Data);
                var labelmatrix = LabelMatrices[dataset.Label];

                var tmpCost =
                    -labelmatrix.PointwiseMultiply(result.Map(Math.Log)) -
                    (1 - labelmatrix).PointwiseMultiply(result.Map(d => Math.Log(1 - d)));
                cost = tmpCost.RowSums().Sum();
            }

            // Calculate regularization term.
            var reg = 0.0;
            foreach (var weightVector in Network.Weights)
            {
                reg +=
                    weightVector.SubMatrix(0, weightVector.RowCount, 1, weightVector.ColumnCount - 1)
                        .Map(d => Math.Pow(d, 2))
                        .ColumnSums()
                        .Sum();
            }

            reg = reg*(lambda/(2*TrainingData.Count()));

            return cost + reg;
        }

        public void InitilizeLabelMatrices()
        {
            // Initialize Label Matrices
            var distinctLabels = TrainingData.Select(x => x.Label).Distinct().ToList();
            for (var i = 0; i < distinctLabels.Count; i++)
            {
                var matrix = DenseMatrix.OfColumnArrays(new double[distinctLabels.Count()]);
                matrix[i, 0] = 1;
                LabelMatrices.Add(distinctLabels[i], matrix);
            }
        }

        #region MapIndexed

        //NeuronalNetwork.Weights[0].MapIndexed((i, i1, arg3) =>

        //{
        //    return arg3;
        //});

        #endregion
    }
}