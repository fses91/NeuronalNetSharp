using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuronalNetSharp.WPF
{
    using System.IO;
    using Core.Interfaces;
    using Import;
    using MathNet.Numerics.Providers.LinearAlgebra.Mkl;
    using Microsoft.Win32;
    using OxyPlot.Wpf;

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            DataContext = new MainViewModel();
        }

        private void LoadTrainingData_Click(object sender, RoutedEventArgs e)
        {
            var dataFile = string.Empty;
            var labelFile = string.Empty;

            var openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "Select training data file.";
            if (openFileDialog.ShowDialog() == false)
                return;

            dataFile = openFileDialog.FileName;

            openFileDialog.Title = "Select label data file.";
            if (openFileDialog.ShowDialog() == false)
                return;

            labelFile = openFileDialog.FileName;

            var importer = new MinstImporter();
            ((MainViewModel)DataContext).TrainingData = importer.ImportData(dataFile, labelFile);
        }

        private void TrainNetwork_Click(object sender, RoutedEventArgs e)
        {
            var model = DataContext as MainViewModel;

            if (model?.TrainingData == null ||
                !model.TrainingData.Any())
            {
                MessageBox.Show(this, "No training data was loaded.", ContentStringFormat, MessageBoxButton.OK);
                return;
            }
            model.TrainNetwork();
        }

        private void TestNetworkButton_Click(object sender, RoutedEventArgs e)
        {
            var model = DataContext as MainViewModel;

            if (model != null && !model.TrainingTask.IsCompleted)
                MessageBox.Show(this, "Network is not finished with training.");
            else
                model?.TestNetwork();
        }

        private void TestNetworkOnCrossValidationDataButton_Click(object sender, RoutedEventArgs e)
        {
            var model = DataContext as MainViewModel;

            if (model != null && !model.TrainingTask.IsCompleted)
                MessageBox.Show(this, "Network is not finished with training.");
            else
                model?.TestNetworkWithCrossValidation();
        }
    }
}
