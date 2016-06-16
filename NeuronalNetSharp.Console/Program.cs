using NeuronalNetSharp.Core;
using NeuronalNetSharp.Core.LearningAlgorithms;

namespace NeuronalNetSharp.Console
{
    using System;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using Import;
    using MathNet.Numerics;
    using MathNet.Numerics.Providers.LinearAlgebra.Mkl;

    internal class Program
    {
        private static void Main()
        {
            double[] x = {0, 0};
            const double epsg = 0.0000000001;
            const double epsf = 0;
            const double epsx = 0;
            const int maxits = 0;
            alglib.minlbfgsstate state;
            alglib.minlbfgsreport rep;


            alglib.minlbfgscreate(1, x, out state);
            alglib.minlbfgssetcond(state, epsg, epsf, epsx, maxits);
            alglib.minlbfgsoptimize(state, function1_grad, null, null);
            alglib.minlbfgsresults(state, out x, out rep);

            Console.WriteLine($"{rep.terminationtype}"); // EXPECTED: 4
            Console.WriteLine($"{alglib.ap.format(x, 2)}"); // EXPECTED: [-3,3]
            Console.ReadLine();
     
            return;

            //Control.NativeProviderPath = "x64/";

            //Control.NativeProviderPath = "x64";
            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            Control.UseNativeMKL();

            var importer = new MinstImporter();
            var rawData = importer.ImportData(@"train-images-idx3-ubyte",
                @"train-labels-idx1-ubyte").ToList();

            var network = new NeuronalNetwork(784, 1, 10);

            network.SetLayerSize(0, 5);

            var backprob = new BackpropagationLearningAlgorithm(network);
            var test = backprob.TrainNetwork(100, 0.01, 0, rawData.Take(100).ToList());


            var stopwatch = new Stopwatch();
            stopwatch.Start();

            stopwatch.Stop();
            Console.WriteLine(stopwatch.Elapsed);


            var tmp1 = rawData[6000].Label;
            var tmp2 = rawData[5323].Label;

            var test1 = network.ComputeOutput(rawData[25].Data);
            var test2 = network.ComputeOutput(rawData[33].Data);
            var test3 = network.ComputeOutput(rawData[37].Data);
            var test4 = network.ComputeOutput(rawData[12].Data);
            var test5 = network.ComputeOutput(rawData[6000].Data);
            var test6 = network.ComputeOutput(rawData[25].Data);
            var test7 = network.ComputeOutput(rawData[33].Data);
            var test8 = network.ComputeOutput(rawData[37].Data);
            var test9 = network.ComputeOutput(rawData[12].Data);
            var test10 = network.ComputeOutput(rawData[5323].Data);


            Console.WriteLine(Directory.GetCurrentDirectory());
            Console.ReadLine();
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