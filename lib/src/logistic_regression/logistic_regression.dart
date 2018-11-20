part of grizzly.classify;

/// Encloses a Logistic regression problem for given exogenous variables [x] and
/// endogenous variable [y]
class LogisticRegressionProblem extends RegressionProblem {
  /// Exogenous variables
  final Double2DView x;

  /// Binary endogenous variable
  final Double1DView y;

  // TODO add documentation
  final Double1DView c;

  /// Constructs a Logistic regression problem from given [x], [y] and [c]
  LogisticRegressionProblem(this.x, this.y, this.c);

  /// Computes the cost function with parameters [w]
  double costFunction(Double1DView w) {
    final Double1DView wx = x.dot(w);

    double ret = 0.0;

    for (int i = 0; i < x.numCols; i++) ret += w[i] * w[i];
    ret /= 2;

    for (int i = 0; i < y.length; i++) {
      final double yz = y[i] * wx[i];
      if (yz >= 0)
        ret += c[i] * math.log(1 + math.exp(-yz));
      else
        ret += c[i] * (-yz + math.log(1 + math.exp(yz)));
    }

    return ret;
  }

  /// Computes the gradient with parameters [w]
  Double1DFix gradient(Double1DView w) {
    final g = new Double1D(w);
    for (int i = 0; i < y.length; i++) {
      final double jTheta = 1 / (1 + math.exp(-y[i] * x[i].dot(w)));
      final double temp = c[i] * (jTheta - 1) * y[i];
      for (int j = 0; j < g.length; j++) {
        g[j] += temp * x[i][j];
      }
    }
    return g;
  }
}
