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
        ///     The data.
        /// </summary>
        Matrix Data { get; }

        /// <summary>
        ///     The label for the data.
        /// </summary>
        string Label { get; }

        /// <summary>
        ///     The number of columns of the input data.
        /// </summary>
        int NumberOfColumns { get; }

        /// <summary>
        ///     The number of rows of the input data.
        /// </summary>
        int NumberOfRows { get; }
    }
}