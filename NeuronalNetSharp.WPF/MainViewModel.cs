
namespace NeuronalNetSharp.WPF
{
    using System;
    using System.Collections.Generic;
    using System.ComponentModel;
    using System.Linq;
    using System.Runtime.CompilerServices;
    using System.Threading.Tasks;
    using Core;
    using Core.NeuronalNetwork;
    using Core.Optimization;
    using Core.Performance;
    using Import;
    using Annotations;
    using OxyPlot;
    using OxyPlot.Axes;
    using OxyPlot.Series;
    using MathNet.Numerics.LinearAlgebra;

    /// <summary>
    /// This class is the viewmodel of the application.
    /// </summary>
    public class MainViewModel : INotifyPropertyChanged
    {
        /// <summary>
        /// The plotmodel of the cost function.
        /// </summary>
        private PlotModel _costFunctionPlotModel;

        /// <summary>
        /// The error on the crossvalidationset.
        /// </summary>
        private double _crossValidationError;

        /// <summary>
        /// The error on the traingset.
        /// </summary>
        private double _trainingError;

        /// <summary>
        /// The error on the testset.
        /// </summary>
        private double _testError;

        /// <summary>
        /// The cost of the network.
        /// </summary>
        private double _cost;

        /// <summary>
        /// Initializes a new instance of the MainViewModel.
        /// </summary>
        public MainViewModel()
        {
            Alpha = 0.0001;
            Lambda = 0.0001;
            Iterations = 100;

            CostFunctionLineSeries = new LineSeries();
            CostFunctionPlotModel = new PlotModel
            {
                Axes =
                {
                    new LinearAxis {Position = AxisPosition.Left, Minimum = 0, Maximum = 30},
                    new LinearAxis {Position = AxisPosition.Bottom, Minimum = 0, Maximum = 120}
                }
            };
        }

        /// <summary>
        /// Gets or sets the alpha value.
        /// </summary>
        public double Alpha { get; set; }

        /// <summary>
        /// Gets or sets the cost value.
        /// </summary>
        public double Cost
        {
            get { return _cost; }
            set
            {
                if (value == _cost) return;
                _cost = value;
                OnPropertyChanged(nameof(Cost));
            }
        }

        /// <summary>
        /// Gets or sets the cost function line series.
        /// </summary>
        public LineSeries CostFunctionLineSeries { get; set; }

        /// <summary>
        /// Gets or sets the cost function plot model.
        /// </summary>
        public PlotModel CostFunctionPlotModel
        {
            get { return _costFunctionPlotModel; }
            set
            {
                if (value == _costFunctionPlotModel) return;
                _costFunctionPlotModel = value;
                OnPropertyChanged(nameof(CostFunctionPlotModel));
            }
        }

        /// <summary>
        /// Gets or sets the cross validation data.
        /// </summary>
        public IList<IDataset> CrossValidationData { get; set; }

        /// <summary>
        /// Gets or sets the cross validation data which get used for calcualtion.
        /// </summary>
        public int CrossValidationDataToUse { get; set; }

        /// <summary>
        /// Gets or sets the crossvalidationerror.
        /// </summary>
        public double CrossValidationError
        {
            get { return _crossValidationError; }
            set
            {
                if (value == _crossValidationError) return;
                _crossValidationError = value;
                OnPropertyChanged(nameof(CrossValidationError));
            }
        }

        /// <summary>
        /// Gets or sets the the input size.
        /// </summary>
        public int InputLayerSize { get; set; }

        /// <summary>
        /// Gets or sets the iterations counter.
        /// </summary>
        public int IterationCount { get; set; }

        /// <summary>
        /// Gets or sets the amount of iterations.
        /// </summary>
        public int Iterations { get; set; }

        /// <summary>
        /// Gets or sets the lambda value.
        /// </summary>
        public double Lambda { get; set; }

        /// <summary>
        /// Gets or sets the result matrices with labels.
        /// </summary>
        public IDictionary<string, Matrix<double>> Results { get; set; }

        /// <summary>
        /// Gets or sets the network.
        /// </summary>
        public INeuronalNetwork Network { get; set; }

        /// <summary>
        /// Gets or sets the optimizer.
        /// </summary>
        public IOptimization Optimizer { get; set; }

        /// <summary>
        /// Gets or sets the number of hidden layers.
        /// </summary>
        public int NumberOfHiddenLayers { get; set; }

        /// <summary>
        /// Gets or sets the size of the output layer.
        /// </summary>
        public int OutputLayerSize { get; set; }

        /// <summary>
        /// Gets or sets the testdata.
        /// </summary>
        public IList<IDataset> TestData { get; set; }

        /// <summary>
        /// Gets or sets the testdata which get used for calculations.
        /// </summary>
        public int TestDataToUse { get; set; }

        /// <summary>
        /// Gets or sets the traing error.
        /// </summary>
        public double TrainingError
        {
            get { return _trainingError; }
            set
            {
                if (value == _trainingError) return;
                _trainingError = value;
                OnPropertyChanged(nameof(TrainingError));
            }
        }

        /// <summary>
        /// Gets or sets the test error.
        /// </summary>
        public double TestError
        {
            get { return _testError; }
            set
            {
                if (value == _testError) return;
                _testError = value;
                OnPropertyChanged(nameof(TestError));
            }
        }

        /// <summary>
        /// Gets or sets the traingdata which get used for calcualtion.
        /// </summary>
        public int TraingDataToUse { get; set; }

        /// <summary>
        /// Gets or sets the traingdata. 
        /// </summary>
        public IList<IDataset> TrainingData { get; set; }

        /// <summary>
        /// Gets or sets the traingstask.
        /// </summary>
        public Task TrainingTask { get; set; }

        /// <summary>
        /// Property changed event handler.
        /// </summary>
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// Create a new network.
        /// </summary>
        public void CreateNewNetwork()
        {
            Network = new NeuronalNetwork(InputLayerSize, OutputLayerSize, NumberOfHiddenLayers, Lambda);
            IterationCount = 0;
        }

        /// <summary>
        /// Train the network.
        /// </summary>
        public void TrainNetwork()
        {
            if(Network == null)
                throw new NullReferenceException("Network is null");

            if (TrainingTask == null || TrainingTask.IsCompleted)
            {
                TrainingTask = Task.Run(() =>
                {
                    Optimizer = new GradientDescentAlgorithm(Lambda, Alpha);
                    Optimizer.IterationFinished += UpdateCostFunctionPlot;
                    Optimizer.OptimizeNetwork(Network, TrainingData.ToList(), HelperFunctions.GetLabelMatrices(TrainingData), Iterations);
                });
            }
        }

        /// <summary>
        /// Test the network on traingdata.
        /// </summary>
        public void TestNetwork()
        {
            TrainingError = NetworkTester.TestNetwork(
                Network,
                TrainingData.Take(TraingDataToUse),
                Results);
        }

        /// <summary>
        /// Test the network on crossvalidatondata.
        /// </summary>
        public void TestNetworkWithCrossValidation()
        {
            CrossValidationError = NetworkTester.TestNetwork(Network,
                TrainingData.Skip(TraingDataToUse).Take(CrossValidationDataToUse),
                Results);
        }

        /// <summary>
        /// Test the network on testdata.
        /// </summary>
        public void TestNetworkWithTestSet()
        {
            TestError = NetworkTester.TestNetwork(Network,
                TrainingData.Skip(TraingDataToUse).Skip(CrossValidationDataToUse).Take(TestDataToUse), Results);
        }

        /// <summary>
        /// Update the plot of the cost functions if a traingsiteraions is finished.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        public void UpdateCostFunctionPlot(object sender, EventArgs e)
        {
            var args = (IterationFinishedEventArgs) e;
            var plotModel = new PlotModel();

            Cost = args.Cost;

            CostFunctionLineSeries.Points.Add(new DataPoint(IterationCount, args.Cost));

            plotModel.Axes.Add(new LinearAxis {Position = AxisPosition.Left, Minimum = 0, Maximum = args.Cost + 2});
            plotModel.Axes.Add(new LinearAxis {Position = AxisPosition.Bottom, Minimum = 0, Maximum = IterationCount + 5});

            CostFunctionPlotModel.Series.Clear();
            plotModel.Series.Add(CostFunctionLineSeries);
            CostFunctionPlotModel = plotModel;
            IterationCount++;
        }

        /// <summary>
        /// PropertyChangedInvocator.
        /// </summary>
        /// <param name="propertyName"></param>
        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}