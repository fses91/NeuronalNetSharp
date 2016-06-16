namespace NeuronalNetSharp.WPF
{
    using System.Linq;
    using System.Windows;
    using System.Windows.Controls;
    using Import;
    using Microsoft.Win32;

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
            ((MainViewModel) DataContext).TrainingData = importer.ImportData(dataFile, labelFile);
        }

        private void TrainNetwork_Click(object sender, RoutedEventArgs e)
        {
            var model = (MainViewModel) DataContext;

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
            var model = (MainViewModel) DataContext;

            var btmap = VisualizerTmp.VisualizeLayerGrayscale(model.Network.Weights[0], 28, 28);

            foreach (var imageSource in btmap)
                VisalizationListBox.Items.Add(new Image {Source = imageSource, Width = 100, Height = 100});
        }
    }
}