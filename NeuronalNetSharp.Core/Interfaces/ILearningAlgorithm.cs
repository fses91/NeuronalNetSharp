namespace NeuronalNetSharp.Core.Interfaces
{
    interface ILearningAlgorithm
    {
        INeuronalNetwork TrainNetwork(int iterations);
    }
}
