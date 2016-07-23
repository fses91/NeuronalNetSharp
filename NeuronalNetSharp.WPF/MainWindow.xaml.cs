namespace NeuronalNetSharp.WPF
{
    using System;
    using System.Linq;
    using System.Windows;
    using System.Windows.Controls;
    using Core;
    using Import;
    using Microsoft.Win32;
    using OxyPlot;
    using OxyPlot.Axes;
    using OxyPlot.Series;

    /// <summary>
    ///     Interaction logic for MainWindow.xaml
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
            var model = (MainViewModel)DataContext;

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

            var importer = new MinstSmallImporter();
            model.TrainingData = importer.ImportData(dataFile, labelFile).ToList();
            model.TrainingData.Shuffle();
            model.Results = HelperFunctions.GetLabelMatrices(model.TrainingData);

        }

        private void TrainNetwork_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            if (model?.Network == null)
                MessageBox.Show(this, "Not network was loaded");

            if (model?.TrainingData == null || !model.TrainingData.Any())
                MessageBox.Show(this, "No training data was loaded.", ContentStringFormat, MessageBoxButton.OK);

            model?.TrainNetwork();
        }

        private void TestNetworkButton_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            if (model?.TrainingTask != null && !model.TrainingTask.IsCompleted)
                MessageBox.Show(this, "Network is not finished with training.");
            else
                model?.TestNetwork();
        }

        private void TestNetworkOnCrossValidationDataButton_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            if (model != null && !model.TrainingTask.IsCompleted)
                MessageBox.Show(this, "Network is not finished with training.");
            else
                model?.TestNetworkWithCrossValidation();
        }

        private void VisualizeNodesButton_Click(object sender, RoutedEventArgs e)
        {
            VisalizationListBox.Items.Clear();
            var model = (MainViewModel) DataContext;
            var btmap = VisualizerTmp.VisualizeLayerGrayscale(model.Network.Weights[Convert.ToInt32(LayerToVisualizeTextBox.Text)], 20, 20);
            //var btmap = VisualizerTmp.VisualizeLayerGrayscale(model.Network, model.BackpropagationAlgorithm, Convert.ToInt32(LayerToVisualizeTextBox.Text), 28, 28, 10);

            foreach (var imageSource in btmap)
                VisalizationListBox.Items.Add(new Image {Source = imageSource, Width = 100, Height = 100});
        }

        private void CreateNewNetworkButton_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;
            model.CostFunctionLineSeries = new LineSeries();
            model.CostFunctionPlotModel = new PlotModel
            {
                Axes =
                {
                    new LinearAxis {Position = AxisPosition.Left, Minimum = 0, Maximum = 30},
                    new LinearAxis {Position = AxisPosition.Bottom, Minimum = 0, Maximum = 120}
                }
            };
            model.CreateNewNetwork();
        }

        private void SetLayerSizeButton_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            var layer = Convert.ToInt32(LayerToChangeTextBox.Text);
            var size = Convert.ToInt32(SizeToChangeTextBox.Text);

            model.Network.SetLayerSize(layer, size);
        }
    }
}