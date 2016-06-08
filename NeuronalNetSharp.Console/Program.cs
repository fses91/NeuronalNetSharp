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



            var rand = DenseMatrix.CreateRandom(5, 5, new ContinuousUniform(0, 255));
            var tali = 1000*rand;


            var dataset = new MinstDataset(DenseMatrix.CreateRandom(5, 1, new ContinuousUniform(0, 255)), "0", 5, 5);
            var dataset1 = new MinstDataset(DenseMatrix.CreateRandom(5, 1, new ContinuousUniform(0, 255)), "1", 5, 5);
            var dataset2 = new MinstDataset(DenseMatrix.CreateRandom(5, 1, new ContinuousUniform(0, 255)), "2", 5, 5);
            var dataset3 = new MinstDataset(DenseMatrix.CreateRandom(5, 1, new ContinuousUniform(0, 255)), "3", 5, 5);
            var dataset4 = new MinstDataset(DenseMatrix.CreateRandom(5, 1, new ContinuousUniform(0, 255)), "4", 5, 5);


            var list = new List<IDataset> { dataset, dataset1, dataset2, dataset3, dataset4 };

            var network = new NeuronalNetwork(784, 1, 10);
            var backprob = new BackpropagationLearningAlgorithm(network, rawData.Take(50));

            var test = backprob.TrainNetwork(200, 0.1, 0.000001);

            var test5 = network.ComputeOutput(rawData[25].Data);
            var test6 = network.ComputeOutput(rawData[501].Data);


            Console.WriteLine(Directory.GetCurrentDirectory());
            Console.ReadLine();
        }
    }
}