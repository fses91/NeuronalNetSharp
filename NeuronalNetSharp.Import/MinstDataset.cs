namespace NeuronalNetSharp.Import
{
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// This class is a dataset for the mnist data.
    /// </summary>
    public class MinstDataset : IDataset
    {
        /// <summary>
        /// Initializes a new instance of MnistDataset.
        /// </summary>
        /// <param name="data">The data.</param>
        /// <param name="label">The lable of the data.</param>
        /// <param name="rows">The number of rows of the data.</param>
        /// <param name="columns">The numbe of columns of the data.</param>
        public MinstDataset(Matrix data, string label, int rows, int columns)
        {
            Data = data;
            Label = label;
            NumberOfRows = rows;
            NumberOfColumns = columns;
        }

        /// <summary>
        /// Gets of sets the data.
        /// </summary>
        public Matrix Data { get; set; }

        /// <summary>
        /// Gets of sets the label.
        /// </summary>
        public string Label { get; }

        /// <summary>
        /// Gets of sets the number of columns.
        /// </summary>
        public int NumberOfColumns { get; }

        /// <summary>
        /// Gets or sets the number of rows. 
        /// </summary>
        public int NumberOfRows { get; }
    }
}