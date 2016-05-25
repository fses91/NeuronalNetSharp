using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;
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
                @"C:\Users\flori\Desktop\train-labels-idx1-ubyte");

            System.Console.WriteLine(Directory.GetCurrentDirectory());
            System.Console.ReadLine();
        }
    }
}