namespace NeuronalNetSharp.Console
{
    using System;
    using System.IO;
    using System.Linq;
    using Core;
    using Core.LearningAlgorithms;
    using Import;

    internal class Program
    {
        private static void Main()
        {
            var importer = new MinstImporter();
            var rawData = importer.ImportData(@"train-images-idx3-ubyte",
                @"train-labels-idx1-ubyte").ToList();

            var network = new NeuronalNetwork(784, 1, 10);
            var backprob = new BackpropagationLearningAlgorithm(network, rawData.Take(100).ToList());

            var test = backprob.TrainNetwork(10);

            Console.WriteLine(Directory.GetCurrentDirectory());
            Console.ReadLine();
        }
    }
}