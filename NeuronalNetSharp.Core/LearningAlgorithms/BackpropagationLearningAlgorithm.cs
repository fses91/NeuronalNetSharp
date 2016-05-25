using System;
using System.Collections.Generic;
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
        }

        public NeuronalNetwork NeuronalNetwork { get; set; }

        public List<IDataset> TrainingData { get; set; }

        public NeuronalNetwork TrainNetwork()
        {
            throw new NotImplementedException();
        }

        //    //NeuronalNetwork.Weights[0].MapIndexed((i, i1, arg3) =>
        //#region MapIndexed
        //{

        //public double ComputeCost()
        //    //{
        //    //    return arg3;
        //    //});
        //#endregion


        //    foreach (var dataset in TrainingData)
        //    {
        //        var result = NeuronalNetwork.ComputeOutput(dataset.Data);


        //    }

        //}
    }
}