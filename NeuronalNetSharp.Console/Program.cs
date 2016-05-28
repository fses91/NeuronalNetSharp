using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuronalNetSharp.Core;
using NeuronalNetSharp.Core.LearningAlgorithms;
using NeuronalNetSharp.Import.Importer;

namespace NeuronalNetSharp.Console
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var importer = new MinstImporter();
            var rawData = importer.ImportData(
                @"C:\Users\flori\Desktop\train-images-idx3-ubyte",
                @"C:\Users\flori\Desktop\train-labels-idx1-ubyte").ToList();
            
            var network = new NeuronalNetwork(784, 1, 12);

            var result = network.ComputeOutput(rawData[0].Data);

            var backprob = new BackpropagationLearningAlgorithm(network, rawData);
            backprob.ComputeCost();

            System.Console.WriteLine(Directory.GetCurrentDirectory());
            System.Console.ReadLine();
        }
    }
}