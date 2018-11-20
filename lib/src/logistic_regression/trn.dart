part of grizzly.classify;

/// A model that is regress-able, aka. Has a cost function and a gradient
abstract class RegressionProblem {
  /// Exogenous variables
  Double2DView get x;

  /// Binary endogenous variable
  Double1DView get y;

  /// Computes the cost function with parameters [w]
  double costFunction(Double1DView w);

  /// Computes the gradient for with parameters [w]
  Double1DFix gradient(Double1DView w);
}

/// Solves a regress-able problem using Trust region newton method
class TrustRegionNewtonSolver {
  /// The regress-able problem which needs to be solved
  final RegressionProblem problem;

  final double tolerance;

  /// Exogenous variable
  Double2DView get x => problem.x;

  /// Binary endogenous variable
  Double1DView get y => problem.y;

  TrustRegionNewtonSolver(this.problem, {this.tolerance});  // TODO set tolerance

  void trcg(double delta, Double1DFix g, Double1DFix s, Double1DFix r) {
    // TODO
  }

  void learn_(Double1DFix w, double eps, int maxIter) {
    // TODO check that w's length is same as x's

    // Calculate gradient norm at w=0 for stopping condition.
    Double1DFix g = problem.gradient(new Double1DView.sized(w.length));
    double gnorm0 = norm2(g);

    double f = problem.costFunction(w);
    g = problem.gradient(w);
    double gnorm = norm2(g);

    if(gnorm > eps * gnorm0) return;

    for(int i = 0; i < maxIter; i++) {
      // TODO
    }

    // TODO
  }

  Double1DFix learn(int maxIter, double eps) {
    final w = new Double1DFix.sized(x.numCols); // The parameters to be found!

    final s = new Double1DFix.sized(x.numCols); // TODO what is this?
    final r = new Double1DFix.sized(x.numCols); // TODO what is this?

    // Calculate gradient norm at w=0 for stopping condition.
    double f = problem.costFunction(w);
    Double1DFix g = problem.gradient(w);
    double delta = norm2(g);
    double gNorm0 = delta;

    final bool search = gNorm0 <= eps * gnorm1;

    int iter = 1;

    while (iter <= maxIter && search) {
      trcg(delta, g, s, r);

      final wNew = w + s;

      final double gs = g.dot(s);
      final double prered = -0.5 * (gs - s.dot(r));

      final double fnew = problem.costFunction(wNew);

      // Compute the actual reduction.
      final double actred = f - fnew;

      // On the first iteration, adjust the initial step bound.
      double snorm = norm2(s);
      if (iter == 1) delta = math.min(delta, snorm);

      // Compute prediction alpha*snorm of the step.
      double alpha;
      if (fnew - f - gs <= 0)
        alpha = _sigma3;
      else
        alpha = math.max(_sigma1, -0.5 * (gs / (fnew - f - gs)));

      // Update the trust region bound according to the ratio of actual to predicted reduction.
      if (actred < _eta0 * prered)
        delta = math.min(math.max(alpha, _sigma1) * snorm, _sigma2 * delta);
      else if (actred < _eta1 * prered)
        delta =
            math.max(_sigma1 * delta, math.min(alpha * snorm, _sigma2 * delta));
      else if (actred < _eta2 * prered)
        delta =
            math.max(_sigma1 * delta, math.min(alpha * snorm, _sigma3 * delta));
      else
        delta = math.max(delta, math.min(alpha * snorm, _sigma3 * delta));

      if (actred > _eta0 * prered) {
        iter++;
        w.assign(wNew);
        f = fnew;
        g.assign(problem.gradient(w));

        gNorm0 = norm2(g);
        if (gNorm0 <= eps * gnorm1) break;
      }

      if (f < -1.0e+32) {
        // TODO info("WARNING: f < -1.0e+32\n");
        break;
      }

      if (actred.abs() <= 0 && prered <= 0) {
        // TODO info("WARNING: actred and prered <= 0\n");
        break;
      }

      if (actred.abs() <= 1.0e-12 * f.abs() &&
          prered.abs() <= 1.0e-12 * f.abs()) {
        // TODO info("WARNING: actred and prered too small\n");
        break;
      }
    }

    return w;
  }

  // Parameters for updating the iterates.
  static const double _eta0 = 1e-4;
  static const double _eta1 = 0.25;
  static const double _eta2 = 0.75;

  // Parameters for updating the trust region size delta.
  static const double _sigma1 = 0.25;
  static const double _sigma2 = 0.5;
  static const double _sigma3 = 4.0;
}
