namespace NeuronalNetSharp.Import.Interfaces
{
    public interface IDataset<T>
    {
        string Label { get; set; }

        T Data { get; set; }
    }
}
