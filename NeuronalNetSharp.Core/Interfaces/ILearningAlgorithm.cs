namespace NeuronalNetSharp.Core.Interfaces
{
    interface ILearningAlgorithm
    {
        NeuronalNetwork TrainNetwork(NeuronalNetwork neuronalNetwork);
    }
}
