namespace NeuronalNetSharp.Core
{
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using Import;
    using MathNet.Numerics.LinearAlgebra.Double;

    public static class HelperFunctions
    {
        //public double ComputeCost(INeuronalNetwork network, IList<IDataset> trainingData)
        //{
        //    var labelMatrices = GetLabelMatrices(trainingData);

        //    // Cost
        //    var result = 0.0;
        //    foreach (var dataset in trainingData)
        //    {
        //        var difference = network.ComputeOutput(dataset.Data) - labelMatrices[dataset.Label];
        //        var norm = difference.CalculateNorm();
        //        result += 1.0 / 2.0 * Math.Pow(norm, 2);
        //    }

        //    return result / trainingData.Count;
        //}

        //public double ComputeCostRegularized(INeuronalNetwork network, IList<IDataset> trainingData, double lambda)
        //{
        //    var reg = 0.0;
        //    foreach (var weightVector in network.Weights)
        //    {
        //        reg +=
        //            weightVector.SubMatrix(0, weightVector.RowCount, 1, weightVector.ColumnCount - 1)
        //                .Map(d => Math.Pow(d, 2))
        //                .ColumnSums()
        //                .Sum();
        //    }

        //    reg = lambda / 2.0 * reg;

        //    return reg + ComputeCost(network, trainingData);
        //}

        ////public double ComputeCostRegularized(INeuronalNetwork network, IList<IDataset> trainingData, double lambda)
        ////{
        ////    // Calculate cost.
        ////    var labelMatrices = GetLabelMatrices(trainingData);

        ////    var cost = 0.0;
        ////    foreach (var dataset in trainingData)
        ////    {
        ////        var result = network.ComputeOutput(dataset.Data);
        ////        var labelmatrix = labelMatrices[dataset.Label];

        ////        var tmpCost =
        ////            -labelmatrix.PointwiseMultiply(result.Map(Math.Log)) -
        ////            (1 - labelmatrix).PointwiseMultiply(result.Map(d => Math.Log(1 - d)));
        ////        cost = tmpCost.RowSums().Sum();
        ////    }

        ////    // Calculate regularization term.
        ////    var reg = 0.0;
        ////    foreach (var weightVector in network.Weights)
        ////    {
        ////        reg +=
        ////            weightVector.SubMatrix(0, weightVector.RowCount, 1, weightVector.ColumnCount - 1)
        ////                .Map(d => Math.Pow(d, 2))
        ////                .ColumnSums()
        ////                .Sum();
        ////    }

        ////    reg = reg * (lambda / (2 * trainingData.Count()));

        ////    return cost + reg;
        ////}

        //public IList<double> ComputeNumericalGradients(INeuronalNetwork network, IList<IDataset> traingData, IBackpropagation backprob, double lambda)
        //{
        //    var epsilon = 0.0001;
        //    var numGrad = new List<double>();

        //    foreach (var weightMatrix in network.Weights)
        //    {
        //        for (var i = 0; i < weightMatrix.RowCount; i++)
        //        {
        //            for (var j = 0; j < weightMatrix.ColumnCount; j++)
        //            {
        //                weightMatrix[i, j] = weightMatrix[i, j] - epsilon;
        //                var loss1 = backprob.ComputeCostRegularized(network, traingData, lambda);
        //                weightMatrix[i, j] = weightMatrix[i, j] + 2*epsilon;
        //                var loss2 = backprob.ComputeCostRegularized(network, traingData, lambda);

        //                weightMatrix[i, j] = weightMatrix[i, j] - epsilon;

        //                numGrad.Add((loss2 - loss1)/(2*epsilon));
        //            }
        //        }
        //    }

        //    return numGrad;
        //}

        //public IList<Matrix<double>> ComputeDerivatives(INeuronalNetwork network, IList<IDataset> trainingData)
        //{
        //    var labelMatrices = GetLabelMatrices(trainingData);
        //    var deltaMatrices = InitilizeDeltaMatrices(network);

        //    foreach (var dataset in trainingData)
        //    {
        //        var tmpDeltaVectors = new List<Matrix<double>>();
        //        var output = network.ComputeOutput(dataset.Data);
        //        var deltaLast = output - labelMatrices[dataset.Label];
        //        //-(labelMatrices[dataset.Label] - output).PointwiseMultiply(output.PointwiseMultiply(1 - output));

        //        tmpDeltaVectors.Add(deltaLast);

        //        for (var j = network.Weights.Count - 1; j >= 1; j--)
        //        {
        //            var tmp1 = network.Weights[j].Transpose()*tmpDeltaVectors.Last();
        //            var tmp2 = network.HiddenLayers[j - 1].Map(d => d*(1 - d));
        //            var delta = tmp1.PointwiseMultiply(tmp2);

        //            tmpDeltaVectors.Add(delta.SubMatrix(1, delta.RowCount - 1, 0, delta.ColumnCount));
        //        }

        //        // For the input layer.
        //        tmpDeltaVectors.Reverse();
        //        deltaMatrices[0] = deltaMatrices[0] +
        //                           tmpDeltaVectors[0]*DenseMatrix.Create(1, 1, 1).Append(dataset.Data.Transpose());

        //        for (var j = 1; j < deltaMatrices.Count; j++)
        //            deltaMatrices[j] = deltaMatrices[j] + tmpDeltaVectors[j]*network.HiddenLayers[j - 1].Transpose();
        //    }

        //    Parallel.ForEach(deltaMatrices, matrix => matrix.Map(d => d/trainingData.Count));

        //    return deltaMatrices;
        //}

        //private IList<Matrix<double>> InitilizeDeltaMatrices(INeuronalNetwork network)
        //{
        //    IList<Matrix<double>> deltaVectors = new List<Matrix<double>>();
        //    foreach (var weights in network.Weights)
        //    {
        //        deltaVectors.Add(new SparseMatrix(weights.RowCount, weights.ColumnCount));
        //    }

        //    return deltaVectors;
        //}

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


