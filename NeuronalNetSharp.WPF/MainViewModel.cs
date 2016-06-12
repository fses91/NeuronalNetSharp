namespace NeuronalNetSharp.WPF
{
    using System;
    using System.Collections.Generic;
    using System.ComponentModel;
    using System.Linq;
    using System.Runtime.CompilerServices;
    using System.Threading.Tasks;
    using Annotations;
    using Core;
    using Core.EventArgs;
    using Core.Interfaces;
    using Core.LearningAlgorithms;
    using Import;
    using OxyPlot;
    using OxyPlot.Axes;
    using OxyPlot.Series;

    public class MainViewModel : INotifyPropertyChanged
    {
        private PlotModel _costFunctionPlotModel;

        public MainViewModel()
        {
            Network = new NeuronalNetwork(784, 1, 10);
            LearningAlgorithm = new BackpropagationLearningAlgorithm(Network);
            LearningAlgorithm.IterationFinished += UpdateCostFunctionPlot;

            CostFunctionLineSeries = new LineSeries();
            CostFunctionPlotModel = new PlotModel
            {
                Axes =
                {
                    new LinearAxis {Position = AxisPosition.Left, Minimum = 0, Maximum = 5},
                    new LinearAxis {Position = AxisPosition.Bottom, Minimum = 0, Maximum = 120}
                }
            };

            Alpha = 0.01;
            Lambda = 0.01;
            Iterations = 100;
        }

        public int TraingDataToUse { get; set; }

        public int CrossValidationDataToUse { get; set; }

        public int TestDataToUse { get; set; }

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

        public int Iterations { get; set; }

        public double Lambda { get; set; }

        public ILearningAlgorithm LearningAlgorithm { get; set; }

        public INeuronalNetwork Network { get; set; }

        public ICollection<IDataset> TestData { get; set; }

        public IEnumerable<IDataset> TrainingData { get; set; }

        public Task TrainingTask { get; set; }

        public event PropertyChangedEventHandler PropertyChanged;

        public void TrainNetwork()
        {
            if (TrainingTask == null || TrainingTask.IsCompleted)
            {
                TrainingTask = Task.Run(() => LearningAlgorithm.TrainNetwork(Iterations, Alpha, Lambda, TrainingData.Take(TraingDataToUse).ToList()));
            }
        }

        public void TestNetwork()
        {
            var result = NetworkTester.TestNetwork(LearningAlgorithm.Network, TrainingData.Take(TraingDataToUse), LearningAlgorithm.LabelMatrices);
        }

        public void TestNetworkWithCrossValidation()
        {
            NetworkTester.TestNetwork(LearningAlgorithm.Network, TrainingData.Skip(TraingDataToUse).Take(CrossValidationDataToUse), LearningAlgorithm.LabelMatrices);
        }

        public void UpdateCostFunctionPlot(object sender, EventArgs e)
        {
            var args = (IterationFinishedEventArgs) e;
            CostFunctionLineSeries.Points.Add(new DataPoint(args.Iteration, args.Cost));

            var plotModel = new PlotModel();
            plotModel.Axes.Add(new LinearAxis {Position = AxisPosition.Left, Minimum = 0, Maximum = 5});
            plotModel.Axes.Add(new LinearAxis {Position = AxisPosition.Bottom, Minimum = 0, Maximum = 120});
            CostFunctionPlotModel.Series.Clear();
            plotModel.Series.Add(CostFunctionLineSeries);
            CostFunctionPlotModel = plotModel;
        }

        [NotifyPropertyChangedInvocator]
        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}