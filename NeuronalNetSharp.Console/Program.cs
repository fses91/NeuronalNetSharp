using System;
using System.ComponentModel;
using System.IO;
using System.Linq;
using NeuronalNetSharp.Import.Importer;

namespace NeuronalNetSharp.Console
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            var test = new MinstImporter();

            var test2 = test.ImportData(@"C:\Users\flori\Desktop\train-images-idx3-ubyte",
                @"C:\Users\flori\Desktop\train-labels-idx1-ubyte", 20);


        }
    }
}