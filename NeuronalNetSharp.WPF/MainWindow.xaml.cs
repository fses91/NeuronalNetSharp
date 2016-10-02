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

        /// <summary>
        /// Load the traing data.
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
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

        /// <summary>
        /// Train the network.
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
        private void TrainNetwork_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            if (model?.Network == null)
                MessageBox.Show(this, "No network was loaded.");

            if (model?.TrainingData == null || !model.TrainingData.Any())
                MessageBox.Show(this, "No training data was loaded.", ContentStringFormat, MessageBoxButton.OK);

            model.TrainNetwork();
        }
        
        /// <summary>
        /// Test the network on the traings set.
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
        private void TestNetworkButton_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            if (model?.TrainingTask != null && !model.TrainingTask.IsCompleted)
                MessageBox.Show(this, "Network is not finished with training.");
            else if (model?.Network == null)
                MessageBox.Show(this, "No network was loaded.");
            else if (model?.TrainingData == null)
                MessageBox.Show(this, "No traingdata was loaded");
            else
                model?.TestNetwork();
        }
        
        /// <summary>
        /// Test network on the cross validation set.
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
        private void TestNetworkOnCrossValidationDataButton_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            if (model != null && !model.TrainingTask.IsCompleted)
                MessageBox.Show(this, "Network is not finished with training.");
            else
                model?.TestNetworkWithCrossValidation();
        }
       
        /// <summary>
        /// Test network on the test set.
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
        private void TestNetworkOnTestData_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            if (model != null && !model.TrainingTask.IsCompleted)
                MessageBox.Show(this, "Network is not finished with training");
            else
                model?.TestNetworkWithTestSet();
        }
        
        /// <summary>
        /// Visualize the weights of the network
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
        private void VisualizeNodesButton_Click(object sender, RoutedEventArgs e)
        {
            VisalizationListBox.Items.Clear();
            var model = (MainViewModel) DataContext;
            var btmap = Visualizer.VisualizeLayerGrayscale(model.Network.Weights[Convert.ToInt32(LayerToVisualizeTextBox.Text)], 20, 20);

            foreach (var imageSource in btmap)
                VisalizationListBox.Items.Add(new Image {Source = imageSource, Width = 100, Height = 100});
        }
        
        /// <summary>
        /// Create a new neuronal network.
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
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
        
        /// <summary>
        /// Set the layer size of a network.
        /// </summary>
        /// <param name="sender">Button klicked.</param>
        /// <param name="e">The event args.</param>
        private void SetLayerSizeButton_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

            var layer = Convert.ToInt32(LayerToChangeTextBox.Text);
            var size = Convert.ToInt32(SizeToChangeTextBox.Text);

            model.Network.SetLayerSize(layer, size);
        }
    }
}