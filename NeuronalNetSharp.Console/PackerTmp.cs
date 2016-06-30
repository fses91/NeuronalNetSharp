using NeuronalNetSharp.Core.NeuronalNetwork;

namespace NeuronalNetSharp.Console
{
    using System.Collections.Generic;
    using Import;

    public class PackerTmp
    {
        //public ILearningAlgorithm LearningAlgorithm { get; set; }

        public INeuronalNetwork Network { get; set; }

        public IList<IDataset> Datasets { get; set; }
    }
}