library grizzly.classify;

import 'dart:math' as math;
import 'package:grizzly_linalg/grizzly_linalg.dart';
import 'package:grizzly_series/grizzly_series.dart';

part 'logistic_regression/logistic_regression.dart';
part 'logistic_regression/trn.dart';

lrl2r(Bool1DView y, double eps, double cp, double cn) {
  // TODO

  int pos = 0;

  for (int i = 0; i < y.length; i++) if (y[i]) pos++;
  final int neg = y.length - pos;

  final double primalSolverTol =
      eps * math.max(math.min(pos, neg), 1) / y.length;

  final c = new Double1D.shapedLike(y);
  for (int i = 0; i < y.length; i++) {
    if (y[i])
      c[i] = cp;
    else
      c[i] = cn;
  }

  // TODO

  // TODO
}