using System.Security.Cryptography.X509Certificates;

namespace NeuronalNetSharp.Core.LearningAlgorithms
{
    internal interface ILearningAlgorithm
    {
        NeuronalNetwork TrainNetwork(NeuronalNetwork neuronalNetwork);
    }
}
