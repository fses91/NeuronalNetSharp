namespace NeuronalNetSharp.Import
{
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    ///     An interface for the dataset.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    public interface IDataset
    {
        /// <summary>
        /// Gets of sets the data.
        /// </summary>
        Matrix Data { get; }

        /// <summary>
        /// Gets of sets the label.
        /// </summary>
        string Label { get; }

        /// <summary>
        /// Gets of sets the number of columns.
        /// </summary>
        int NumberOfColumns { get; }

        /// <summary>
        /// Gets or sets the number of rows. 
        /// </summary>
        int NumberOfRows { get; }
    }
}