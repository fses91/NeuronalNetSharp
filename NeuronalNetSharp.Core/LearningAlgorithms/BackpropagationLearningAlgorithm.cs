using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;

namespace NeuronalNetSharp.Core.LearningAlgorithms
{
    public class BackpropagationLearningAlgorithm
    {
        public BackpropagationLearningAlgorithm(NeuronalNetwork neuronalNetwork)
        {
            NeuronalNetwork = neuronalNetwork;

        }

        public NeuronalNetwork NeuronalNetwork { get; set; }    

        public NeuronalNetwork TrainNetwork()
        {
            throw new NotImplementedException();
        }

        public double ComputeCost()
        {
            return 0;
        }
    }
}
