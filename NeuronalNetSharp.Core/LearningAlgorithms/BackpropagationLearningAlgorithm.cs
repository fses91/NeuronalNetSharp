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

        public INeuronalNetwork TrainNetwork(int iterations)
        {
            var deltaVectors = InitilizeDeltaMatrices();
            var tmpDeltaVectors = new List<Matrix<double>>();

            foreach (var dataset in TrainingData)
            {
                var output = Network.ComputeOutput(dataset.Data);
                var deltaLast = output - LabelMatrices[dataset.Label];
                tmpDeltaVectors.Add(deltaLast);

                for (var i = Network.Weights.Count - 1; i >= 1; i--)
                {
                    var tmp1 = Network.Weights[i].Transpose() * tmpDeltaVectors.Last();
                    var tmp2 = Network.HiddenLayers[i - 1].Map(d => d * (1 - d));
                    var delta = tmp1.PointwiseMultiply(tmp2);

                    tmpDeltaVectors.Add(delta.SubMatrix(1, delta.RowCount - 1, 0, delta.ColumnCount));
                }

                tmpDeltaVectors.Reverse();


                // TODO Umlegen auf Schleife.
                deltaVectors[0] = deltaVectors[0] + tmpDeltaVectors[0] * DenseMatrix.Create(1, 1, 1).Append(dataset.Data.Transpose());
                deltaVectors[1] = deltaVectors[1] + tmpDeltaVectors[1]*Network.HiddenLayers[0].Transpose();



                for (var i = 0; i < deltaVectors.Count - 1; i++)
                {
                    deltaVectors[i] = deltaVectors[i] + tmpDeltaVectors[i] * Network.HiddenLayers[i].Transpose();
                }
                var last = deltaVectors.Last();
                last = last + tmpDeltaVectors.Last()*output.Transpose();
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