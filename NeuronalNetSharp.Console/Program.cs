namespace NeuronalNetSharp.Console
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using Core.Optimization;
    using Core.Performance;
    using Import;
    using MathNet.Numerics;
    using MathNet.Numerics.Providers.LinearAlgebra.Mkl;
    using Core;
    using Core.NeuronalNetwork;

    /// <summary>
    /// This class contains the gradient check, performance check and is used for testing the network more interactive.
    /// </summary>
    internal class Program
    {
        public NeuronalNetwork Network { get; set; }

        /// <summary>
        /// This functions contains a gradient check for the network.
        /// </summary>
        private static void Main()
        {
            string path = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory()))) + @"\data";
            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            Control.UseNativeMKL();

            var importer = new MinstSmallImporter();
            var rawData = importer.ImportData(path + @"\trainingData.csv",
                path +  @"\labelData.csv").ToList();

            var data = rawData.ToList();
            data = data.Take(200).ToList();

            var lambda = 0.00;
            var alpha = 0.5;
            var labelMatrices = HelperFunctions.GetLabelMatrices(rawData);

            var network = new NeuronalNetwork(400, 10, 1, lambda);
            network.SetLayerSize(1, 25);

            var optimizer = new GradientDescentAlgorithm(lambda, alpha); 
            optimizer.OptimizeNetwork(network, data, labelMatrices, 10);

            var watch = new Stopwatch();
            watch.Start();
            var cost = network.ComputeCostResultSet(data, labelMatrices, lambda);
            watch.Stop();
            var elapsedN = watch.Elapsed;


            watch.Reset();
            watch.Start();
            var numCost = network.ComputeNumericalGradients(data, labelMatrices, lambda, 0.0001);
            watch.Stop();
            var elapsedA = watch.Elapsed;

            for (int i = 0; i < numCost.Gradients.Count; i++)
            {
                for (int j = 0; j < numCost.Gradients[i].RowCount; j++)
                {
                    for (int k = 0; k < numCost.Gradients[i].ColumnCount; k++)
                    {
                        Console.WriteLine(cost.Gradients.Gradients[i][j, k] + "     " + numCost.Gradients[i][j, k]);
                    }
                }
            }

            Console.WriteLine("#####");

            var test = NetworkTester.TestNetwork(network, data, labelMatrices);

            Console.ReadLine();

            return;            
        }
    }
}
