using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuronalNetSharp.Import.Interfaces;

namespace NeuronalNetSharp.Core.LearningAlgorithms
{
    // TODO
    // Label Matrix erstellen
    // 
    public class BackpropagationLearningAlgorithm
    {
        public BackpropagationLearningAlgorithm(NeuronalNetwork neuronalNetwork, ICollection<IDataset> traningData)
        {
            NeuronalNetwork = neuronalNetwork;
            LabelMatrieMatrices = new Dictionary<string, Matrix>();
            TrainingData = traningData;


            // Initialize Label Matrices
            var distinctLabels = traningData.Select(x => x.Label).Distinct().ToList();
            for (var i = 0; i < distinctLabels.Count; i++)
            {
                var matrix = DenseMatrix.OfColumnArrays(new double[distinctLabels.Count()]);
                matrix[i, 0] = 1;
                LabelMatrieMatrices.Add(distinctLabels[i], matrix);
            }
        }

        public NeuronalNetwork NeuronalNetwork { get; set; }

        public IEnumerable<IDataset> TrainingData { get; set; }

        public IDictionary<string, Matrix> LabelMatrieMatrices { get; set; }

        public NeuronalNetwork TrainNetwork()
        {
            throw new NotImplementedException();
        }


        public double ComputeCost()
        {
            var cost = 0.0;
            foreach (var dataset in TrainingData)
            {
                var result = NeuronalNetwork.ComputeOutput(dataset.Data);
                var labelmatrix = LabelMatrieMatrices[dataset.Label];

                var tmpCost = 
                    (-labelmatrix.PointwiseMultiply(result.Map(Math.Log))) - 
                    (1 - labelmatrix).PointwiseMultiply(result.Map(d => Math.Log(1 - d)));
                cost = tmpCost.RowSums().Sum();
            }


            return cost;
        }

        #region MapIndexed

        //NeuronalNetwork.Weights[0].MapIndexed((i, i1, arg3) =>

        //{
        //    return arg3;
        //});

        #endregion
    }
}