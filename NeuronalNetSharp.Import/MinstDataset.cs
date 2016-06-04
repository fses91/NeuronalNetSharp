namespace NeuronalNetSharp.Import
{
    using MathNet.Numerics.LinearAlgebra.Double;

    public class MinstDataset : IDataset
    {
        public MinstDataset(Matrix data, string label, int rows, int columns)
        {
            Data = data;
            Label = label;
            NumberOfRows = rows;
            NumberOfColumns = columns;
        }

        public Matrix Data { get; set; }

        public string Label { get; }

        public int NumberOfColumns { get; }

        public int NumberOfRows { get; }
    }
}