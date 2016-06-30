namespace NeuronalNetSharp.Core
{
    using System;

    public  static class Functions
    {
        public static double SigmoidFunction(double x)
        {
            return 1.0/(1.0 + Math.Exp(-x));
        }
    }
}