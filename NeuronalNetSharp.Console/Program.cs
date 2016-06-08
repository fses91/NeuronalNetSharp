namespace NeuronalNetSharp.Console
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Core;
    using Core.LearningAlgorithms;
    using Import;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra.Double;

    internal class Program
    {
        private static void Main()
        {
            var importer = new MinstImporter();
            var rawData = importer.ImportData(@"train-images-idx3-ubyte",
                @"train-labels-idx1-ubyte").ToList();

            var network = new NeuronalNetwork(784, 1, 10);
            var backprob = new BackpropagationLearningAlgorithm(network, rawData.Take(50));

            var test = backprob.TrainNetwork(40, 0.05, 4);

            var test5 = network.ComputeOutput(rawData[25].Data);
            var test6 = network.ComputeOutput(rawData[30].Data);


            Console.WriteLine(Directory.GetCurrentDirectory());
            Console.ReadLine();
        }
    }
}