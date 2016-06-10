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
        }

        private INeuronalNetwork Network { get; set; }

        private ICollection<IDataset> TrainingData { get; set; }

        private ICollection<IDataset> CrossValidationData { get; set; }

        private ICollection<IDataset> TestData { get; set; }

        private void LoadTrainingData_Click(object sender, RoutedEventArgs e)
        {
            var dataFile = string.Empty;
            var labelFile = string.Empty;
            var openFileDialog = new OpenFileDialog();

            openFileDialog.Title = "Select training data file.";
            if (openFileDialog.ShowDialog() == true)
                dataFile = openFileDialog.FileName;

            openFileDialog.Title = "Select label data file.";
            if (openFileDialog.ShowDialog() == true)
                labelFile = openFileDialog.FileName;

            var importer = new MinstImporter();
            TrainingData = importer.ImportData(dataFile, labelFile);
        }

        private void LoadTestData_OnClick(object sender, RoutedEventArgs e)
        {
            throw new NotImplementedException();
        }

        private void LoadNetwork_OnClick(object sender, RoutedEventArgs e)
        {
            throw new NotImplementedException();
        }
    }
}
