namespace NeuronalNetSharp.Core.LearningAlgorithms
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Linq;
    using Import;
    using Interfaces;
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

        public INeuronalNetwork TrainNetwork(int iterations, double alpha, double lambda)
        {
            var deltaMatrices = InitilizeDeltaMatrices();

            for (var i = 0; i < iterations; i++)
            {
                foreach (var dataset in TrainingData)
                {
                    var tmpDeltaVectors = new List<Matrix<double>>();
                    var output = Network.ComputeOutput(dataset.Data);
                    var deltaLast = output - LabelMatrices[dataset.Label];
                    tmpDeltaVectors.Add(deltaLast);

                    for (var j = Network.Weights.Count - 1; j >= 1; j--)
                    {
                        var tmp1 = Network.Weights[j].Transpose() * tmpDeltaVectors.Last();
                        var tmp2 = Network.HiddenLayers[j - 1].Map(d => d * (1 - d));
                        var delta = tmp1.PointwiseMultiply(tmp2);

                        tmpDeltaVectors.Add(delta.SubMatrix(1, delta.RowCount - 1, 0, delta.ColumnCount));
                    }

                    tmpDeltaVectors.Reverse();
                    deltaMatrices[0] = deltaMatrices[0] + tmpDeltaVectors[0] * DenseMatrix.Create(1, 1, 1).Append(dataset.Data.Transpose());
                    for (var j = 1; j < deltaMatrices.Count; j++)
                        deltaMatrices[j] = deltaMatrices[j] + tmpDeltaVectors[j] * Network.HiddenLayers[j - 1].Transpose();
                }

                for (var j = 0; j < TrainingData.Count(); j++)
                {
                    deltaMatrices[j] = deltaMatrices[j].Map(d => d / TrainingData.Count());
                    Network.Weights[j] = Network.Weights[j] - alpha * deltaMatrices[j];
                }

                Console.WriteLine(ComputeCost());
            }

            return Network;
        }

        public double ComputeCost()
        {
            //var difference = (output - targetOutput).NormalizeRows(2.0);
            //difference = difference.Map(d => 1.0/2.0 * Math.Pow(d, 2));
            //var cost = 1.0/TrainingData.Count() * difference.ColumnSums().Sum();
            var result = 0.0;

            foreach (var dataset in TrainingData)
            {
                var difference = Network.ComputeOutput(dataset.Data) - LabelMatrices[dataset.Label];
                var norm = difference.CalculateNorm();
                result += 1.0/2.0 * Math.Pow(norm, 2);
            }



            // Calculate regularization term.
            //var reg = 0.0;
            //foreach (var weightVector in Network.Weights)
            //{
            //    reg += weightVector.SubMatrix(0, weightVector.RowCount, 1, weightVector.ColumnCount - 1)
            //            .Map(d => Math.Pow(d, 2))
            //            .ColumnSums()
            //            .Sum();
            //}

            //reg = lambda / 2 * reg;

            return result/TrainingData.Count();
        }

        //public double ComputeCostRegularized(double lambda)
        //{
        //    // Calculate cost.
        //    var cost = 0.0;
        //    foreach (var dataset in TrainingData)
        //    {
        //        var result = Network.ComputeOutput(dataset.Data);
        //        var labelmatrix = LabelMatrices[dataset.Label];

        //        var tmpCost =
        //            -labelmatrix.PointwiseMultiply(result.Map(Math.Log)) -
        //            (1 - labelmatrix).PointwiseMultiply(result.Map(d => Math.Log(1 - d)));
        //        cost = tmpCost.RowSums().Sum();
        //    }

        //    // Calculate regularization term.
        //    var reg = 0.0;
        //    foreach (var weightVector in Network.Weights)
        //    {
        //        reg +=
        //            weightVector.SubMatrix(0, weightVector.RowCount, 1, weightVector.ColumnCount - 1)
        //                .Map(d => Math.Pow(d, 2))
        //                .ColumnSums()
        //                .Sum();
        //    }

        //    reg = reg*(lambda/(2*TrainingData.Count()));

        //    return cost + reg;
        //}

        private void InitilizeLabelMatrices()
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

        private IList<Matrix<double>> InitilizeDeltaMatrices()
        {
            IList<Matrix<double>> deltaVectors = new List<Matrix<double>>();
            foreach (var weights in Network.Weights)
            {
                deltaVectors.Add(new SparseMatrix(weights.RowCount, weights.ColumnCount));
            }

            return deltaVectors;
        }

        #region MapIndexed

        //NeuronalNetwork.Weights[0].MapIndexed((i, i1, arg3) =>

        //{
        //    return arg3;
        //});

        #endregion
    }
}