using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp3
{
    /// <summary>
    /// 
    /// </summary>
    class CrossValidationRegression
    {
        /// <summary>
        /// 
        /// </summary>
        private readonly List<int> basisFunctionIndex;

        /// <summary>
        /// 
        /// </summary>
        private readonly double[] indexCoeff;

        /// <summary>
        /// 
        /// </summary>
        private readonly Func<double[], double>[] f;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="bf"></param>
        /// <param name="crossValFactor"></param>
        /// <param name="runs"></param>
        /// <returns></returns>
        double crossvalidate(double[][] x, double[] y, List<Func<double[], double>> bf, double crossValFactor = 0.1, int runs = 5, bool useAll = false)
        {          
            var forAll = useAll ? Enumerable.Range(0, y.Length) : Enumerable.Range(0, (int)Math.Pow(y.Length, -0.15));

            //TODO remove 
            forAll = Enumerable.Range(0, y.Length);


            var mses = new List<double>();

            Random random = new Random(Seed: 1);

            for (int r = 0; r < runs; ++r)
            {
                double[] weights = forAll.Select(i => random.NextDouble() < crossValFactor ? 0.0 : 1.0).ToArray();
                var xx = forAll.Select(i => x[i]).ToArray();
                var yy = forAll.Select(i => y[i]).ToArray();
                var reg = new LinearRegression(xx, yy, bf.ToArray(), weights: weights);
                var mse = forAll.Select(i => weights[i] == 0 ? reg.Result(x[i]) - y[i] : 0).Select(a => a * a).Sum() / weights.Sum();
                mses.Add(mse);
            }

            return mses.Average();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="which"></param>
        /// <param name="allfunc"></param>
        /// <returns></returns>
        List<Func<double[], double>> takeFunc(List<int> which, List<Func<double[], double>> allfunc)
        {
            var liste = new List<Func<double[], double>>();

            foreach (var ii in which)
                liste.Add(allfunc[ii]);

            return liste;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="funcUse"></param>
        /// <param name="allFunc"></param>
        /// <returns></returns>
        int addFunction(double[][] x, double[] y, List<int> funcUse, List<Func<double[], double>> allFunc, bool useAll = false)
        {
            var abf = takeFunc(funcUse, allFunc);
            double actualcross = funcUse.Count() == 0 ? double.PositiveInfinity : crossvalidate(x, y, takeFunc(funcUse, allFunc));

            Console.WriteLine("actual cross validation: " + actualcross);

            var results = new List<Tuple<int, double>>() { new Tuple<int, double>(-1, actualcross) };

            for (int i = 0; i < allFunc.Count(); i++)
                if (!funcUse.Contains(i))
                {
                    var fu = takeFunc(funcUse, allFunc);
                    fu.Add(allFunc[i]);

                    double mse = crossvalidate(x, y, fu, useAll: useAll);

                    results.Add(new Tuple<int, double>(i, mse));

                    Console.WriteLine("With function i: " + mse);
                }

            return results.OrderBy(item => item.Item2).First().Item1;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="LinearRegression"/> class.
        /// </summary>
        /// <param name="x">The realizations of the explanatory variables.</param>
        /// <param name="y">The realizations of the dependent variable.</param>
        /// <param name="f">The lambda functions to be used for regression.</param>
        /// <param name="weights">The weights of the observations</param>
        public CrossValidationRegression(double[][] x, double[] y, Func<double[], double>[] f, double[] weights = null, bool useAll = true)
        {
            this.f = f;

            this.basisFunctionIndex = new List<int>();

            while (true)
            {
                int next = addFunction(x, y, basisFunctionIndex, f.ToList(), useAll: useAll);
                Console.WriteLine("next: " + next + "\n");

                if (next == -1)
                    break;
                else
                    basisFunctionIndex.Add(next);
            }

            //TODOF weights
            this.indexCoeff = new LinearRegression(x, y, takeFunc(basisFunctionIndex, f.ToList()).ToArray(), weights: weights).Coeff;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double Result(double[] x)
        {
            double sum = 0;
            for (int i = 0; i < this.indexCoeff.Length; ++i)
                sum += this.indexCoeff[i] * this.f[basisFunctionIndex.ElementAt(i)](x);
            return sum;
        }
    }
}
