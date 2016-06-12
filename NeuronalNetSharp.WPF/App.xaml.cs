
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

namespace NeuronalNetSharp.WPF
{
    using System;
    using System.Windows.Controls;
    using MathNet.Numerics.Providers.LinearAlgebra.Mkl;

    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnActivated(EventArgs e)
        {
            base.OnActivated(e);
            MathNet.Numerics.Control.LinearAlgebraProvider = new MklLinearAlgebraProvider();
            MathNet.Numerics.Control.UseNativeMKL();
        }
    }
}
