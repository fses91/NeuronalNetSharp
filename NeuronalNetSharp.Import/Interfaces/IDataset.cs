namespace NeuronalNetSharp.Import.Interfaces
{
    /// <summary>
    /// An interface for the dataset.
    /// </summary>
    /// <typeparam name="T">The type of the data.</typeparam>
    public interface IDataset<T>
    {
        /// <summary>
        /// The label for the data.
        /// </summary>
        string Label { get; set; }

        /// <summary>
        /// The data.
        /// </summary>
        T Data { get; set; }
    }
}
