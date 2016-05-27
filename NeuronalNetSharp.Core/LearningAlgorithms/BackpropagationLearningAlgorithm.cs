using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuronalNetSharp.Import.Interfaces;

namespace NeuronalNetSharp.Core.LearningAlgorithms
{
    // TODO
    // Label Matrix erstellen
    // 
    public class BackpropagationLearningAlgorithm
    {
        public BackpropagationLearningAlgorithm(NeuronalNetwork neuronalNetwork, List<IDataset> traningData)
        {
            NeuronalNetwork = neuronalNetwork;
            LabelMatrieMatrices = new Dictionary<string, Matrix>();


            // Initialize Label Matrix
            var distinctLabels = traningData.Select(x => x.Label).Distinct().ToList();
            for (int i = 0; i < distinctLabels.Count; i++)
            {
                var matrix = DenseMatrix.OfColumnArrays(new double[distinctLabels.Count()]);
                matrix[i, 0] = 1;
                LabelMatrieMatrices.Add(traningData[i].Label, matrix);
            }
        }

        public NeuronalNetwork NeuronalNetwork { get; set; }

        public IEnumerable<IDataset> TrainingData { get; set; }

        public Dictionary<string, Matrix> LabelMatrieMatrices { get; set; }

        public NeuronalNetwork TrainNetwork()
        {
            throw new NotImplementedException();
        }


        //public double ComputeCost()
        //{
        //    foreach (var dataset in TrainingData)
        //    {
        //        var result = NeuronalNetwork.ComputeOutput(dataset.Data);
        //    }


        //}

        #region MapIndexed

        //NeuronalNetwork.Weights[0].MapIndexed((i, i1, arg3) =>

        //{
        //    return arg3;
        //});

        #endregion
    }
}