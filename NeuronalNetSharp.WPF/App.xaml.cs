namespace NeuronalNetSharp.WPF
{
    using System.Windows;
    using MathNet.Numerics;
    using MathNet.Numerics.Providers.LinearAlgebra.Mkl;

    /// <summary>
    ///     Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public App()
        {
            Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            Control.UseNativeMKL();
        }
    }
}