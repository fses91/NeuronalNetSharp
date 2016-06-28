using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using NeuronalNetSharp.Core;
using NeuronalNetSharp.Core.Interfaces;
using NeuronalNetSharp.Core.Optimization;
using NeuronalNetSharp.Core.Performance;
using NeuronalNetSharp.Import;
using NeuronalNetSharp.WPF.Annotations;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace NeuronalNetSharp.WPF
{
    public class MainViewModel : INotifyPropertyChanged
    {
        private PlotModel _costFunctionPlotModel;
        private double _crossValidationError;
        private double _testError;

        public MainViewModel()
        {
            Alpha = 0.0001;
            Lambda = 0.0001;
            Iterations = 100;

            Network = new NeuronalNetwork(784, 1, 10);

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

        public double Alpha { get; set; }

        public LineSeries CostFunctionLineSeries { get; set; }

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

        public ICollection<IDataset> CrossValidationData { get; set; }

        public int CrossValidationDataToUse { get; set; }

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

        public int InputLayerSize { get; set; }

        public int IterationCount { get; set; }

        public int Iterations { get; set; }

        public double Lambda { get; set; }

        public INeuronalNetwork Network { get; set; }

        public IOptimization Optimizer { get; set; }

        public int NumberOfHiddenLayers { get; set; }

        public int OutputLayerSize { get; set; }

        public ICollection<IDataset> TestData { get; set; }

        public int TestDataToUse { get; set; }

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

        public int TraingDataToUse { get; set; }

        public IEnumerable<IDataset> TrainingData { get; set; }

        public Task TrainingTask { get; set; }

        public event PropertyChangedEventHandler PropertyChanged;

        public void CreateNewNetwork()
        {
            Network = new NeuronalNetwork(InputLayerSize, NumberOfHiddenLayers, OutputLayerSize);
            Network.SetLayerSize(0, 25);
            IterationCount = 0;
        }

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
                    Optimizer.OptimizeNetwork(Network, TrainingData.Take(TraingDataToUse).ToList(), Iterations);
                });
            }
        }

        public void TestNetwork()
        {
            TestError = NetworkTester.TestNetwork(
                Network,
                TrainingData.Take(TraingDataToUse),
                Core.HelperFunctions.GetLabelMatrices(TrainingData.Take(TraingDataToUse)));
        }

        public void TestNetworkWithCrossValidation()
        {
            CrossValidationError = NetworkTester.TestNetwork(Network,
                TrainingData.Skip(TraingDataToUse).Take(CrossValidationDataToUse),
                Core.HelperFunctions.GetLabelMatrices(TrainingData.Take(TraingDataToUse)));
        }

        public void UpdateCostFunctionPlot(object sender, EventArgs e)
        {
            var args = (IterationFinishedEventArgs) e;
            var plotModel = new PlotModel();

            CostFunctionLineSeries.Points.Add(new DataPoint(IterationCount, args.Cost));

            plotModel.Axes.Add(new LinearAxis {Position = AxisPosition.Left, Minimum = 0, Maximum = args.Cost + 2});
            plotModel.Axes.Add(new LinearAxis {Position = AxisPosition.Bottom, Minimum = 0, Maximum = IterationCount + 40});

            CostFunctionPlotModel.Series.Clear();
            plotModel.Series.Add(CostFunctionLineSeries);
            CostFunctionPlotModel = plotModel;
            IterationCount++;
        }

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}