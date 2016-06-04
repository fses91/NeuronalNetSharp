using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuronalNetSharp.Import.Interfaces
{
    /// <summary>
    /// An interface for the dataset.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    public interface IDataset
    {
        /// <summary>
        /// The label for the data.
        /// </summary>
        string Label { get; }

        /// <summary>
        /// The number of rows of the input data. 
        /// </summary>
        int NumberOfRows { get; }

        /// <summary>
        /// The number of columns of the input data.
        /// </summary>
        int NumberOfColumns { get; }

        /// <summary>
        /// The data.
        /// </summary>
        Matrix Data { get; }
    }
}
