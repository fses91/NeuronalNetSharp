namespace NeuronalNetSharp.WPF
{
    using System;
    using System.Windows;
    using MathNet.Numerics;
    using MathNet.Numerics.Providers.LinearAlgebra.Mkl;

    /// <summary>
    ///     Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnActivated(EventArgs e)
        {
            base.OnActivated(e);
            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            Control.UseNativeMKL();
            Control.MaxDegreeOfParallelism = 1024;
            Control.UseMultiThreading();
        }
    }
}