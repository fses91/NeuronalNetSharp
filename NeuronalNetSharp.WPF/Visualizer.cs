namespace NeuronalNetSharp.WPF
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Windows.Media;
    using System.Windows.Media.Imaging;
    using Core;
    using MathNet.Numerics.LinearAlgebra;

    public static class Visualizer
    {
        /// <summary>
        /// Creates a imagesource for the visualization of the network nodes.
        /// </summary>
        /// <param name="layer">The layer to visualize.</param>
        /// <param name="width">The width of the imagesource.</param>
        /// <param name="height">The height of the imagesource.</param>
        /// <returns>The image source to visualize the weights.</returns>
        public static IList<ImageSource> VisualizeLayerGrayscale(Matrix<double> layer, int width, int height)
        {
            const int newMin = 0;
            const int newMax = 255;

            var images = new List<ImageSource>();

            for (var i = 0; i < layer.RowCount; i++)
            {
                var unit = layer.SubMatrix(i, 1, 0, layer.ColumnCount).ToColumnWiseArray();
                var oldMax = unit.Max();
                var oldMin = unit.Min();

                // Scale between 0 - 255.
                for (var j = 0; j < unit.Length; j++)
                    unit[j] = HelperFunctions.RescaleValue(unit[j], newMin, newMax, oldMin, oldMax);

                var intTmp = unit.Select(Convert.ToInt32);
                var byteTmp = intTmp.Select(x => (byte)x).ToArray();
                var btmap = CreateBitmap(byteTmp, width, height);
                images.Add(btmap);
            }

            return images;
        }
        
        /// <summary>
        /// Creates a bitmap source.
        /// </summary>
        /// <param name="bytes">The byte array.</param>
        /// <param name="width">The width of the bitmap source.</param>
        /// <param name="height">The heights of the bitmap source.</param>
        /// <returns>The bitmap source.</returns>
        public static BitmapSource CreateBitmap(byte[] bytes, int width, int height)
        {
            return BitmapSource.Create(width, height, 96, 96, PixelFormats.Gray8, BitmapPalettes.Gray256, bytes, 20);
        }
    }
}