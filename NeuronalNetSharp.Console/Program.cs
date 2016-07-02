using NeuronalNetSharp.Core;
using NeuronalNetSharp.Core.NeuronalNetwork;

namespace NeuronalNetSharp.Console
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using Core.Optimization;
    using Core.Performance;
    using Import;
    using MathNet.Numerics;
    using MathNet.Numerics.LinearAlgebra;
    using MathNet.Numerics.LinearAlgebra.Single;
    using MathNet.Numerics.Providers.LinearAlgebra.Mkl;

    internal class Program
    {
        public NeuronalNetwork Network { get; set; }

        private static void Main()
        {
            var testan = new double[] {4, 3, 6, 3};

            string path = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Directory.GetCurrentDirectory()))) + @"\data";




            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            Control.UseNativeMKL();

            var importer = new MinstSmallImporter();
            var rawData = importer.ImportData(path + @"\trainingData.csv",
                path +  @"\labelData.csv").ToList();

            //network.SetLayerSize(0, 10);
            //var backprob = new BackpropagationAlgorithm(network);
            //var packer = new PackerTmp { Network = network, LearningAlgorithm = backprob, Datasets = rawData.Take(1000).ToList()};
            //Console.WriteLine("Cost: " + backprob.ComputeCostRegularized(0.01, rawData.Take(100).ToList())); 
            //Console.WriteLine("Cost: " + backprob.ComputeCostRegularized(0.01, rawData.Take(100).ToList()));

            var data = rawData.Take(20).ToList();

            //INeuronalNetwork network =  new NeuronalNetwork(784, 1, 10);
            ////var backprob = new BackpropagationAlgorithm();
            ////var optimizer = new GradientDescentAlgorithm(0.01, 0.01);
            //network.SetLayerSize(0, 200);
            //network = optimizer.OptimizeNetwork(network, data, 1);
            ////var result = Visualizer.VisualizeNetworkLayer(network, backprob, 0, 30, 28, 28, 500);
            ////var grad1 = backprob.ComputeDerivatives(network, data);
            ////var numGrad1 = /*backprob*/.ComputeNumericalGradients(network, data, backprob, 0.01);
            //network = optimizer.OptimizeNetwork(network, data, 2);
            //network = optimizer.OptimizeNetwork(network, data, 3);
            //network = optimizer.OptimizeNetwork(network, data, 4);

            var labelMatrices = HelperFunctions.GetLabelMatrices(rawData);
            var lambda = 0.00;
            var alpha = 0.1;

            var network = new NeuronalNetwork(400, 10, 1, lambda);

            network.SetLayerSize(1, 25);

            var optimizer = new GradientDescentAlgorithm(lambda, alpha);
            optimizer.IterationFinished += UpdateCostFunctionPlot;
            optimizer.OptimizeNetwork(network, rawData, labelMatrices, 100);

            var test = NetworkTester.TestNetwork(network, rawData, labelMatrices);


            double[] x = null;
            double epsg = 0.0000000001;
            double epsf = 0;
            double epsx = 0;
            int maxits = 0;
            alglib.mincgstate state;
            alglib.mincgreport rep;

            alglib.mincgcreate(x, out state);
            alglib.mincgsetcond(state, epsg, epsf, epsx, maxits);
            //alglib.mincgoptimize(state, function2_grad, null, packer);
            alglib.mincgresults(state, out x, out rep);

            //Console.WriteLine("Cost: " + backprob.ComputeCostRegularized(rawData.Take(100).ToList(), 0.01));


            //Console.WriteLine($"{rep.terminationtype}"); // EXPECTED: 4
            //Console.WriteLine($"{alglib.ap.format(x, 8)}"); // EXPECTED: [-3,3]
            Console.ReadLine();

            return;            
        }

        public static void UpdateCostFunctionPlot(object sender, EventArgs e)
        {
            var args = (IterationStartedEventArgs) e;
            Console.WriteLine(args.Cost);
        }


        public static void function1_grad(double[] x, ref double func, double[] grad, object obj)
        {
            // this callback calculates f(x0,x1) = 100*(x0+3)^4 + (x1-3)^4
            // and its derivatives df/d0 and df/dx1


            func = 100*Math.Pow(x[0] + 3, 4) + Math.Pow(x[1] - 3, 4);
            grad[0] = 400*Math.Pow(x[0] + 3, 3);
            grad[1] = 4*Math.Pow(x[1] - 3, 3);
        }


    }
}
