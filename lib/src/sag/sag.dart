import 'dart:math';
import 'package:grizzly_array/grizzly_array.dart';

class SagState {
  final Bool1DFix seen;

  int numSeen;
}

/// Stochastic Average Gradient (SAG) and SAGA solvers.
class Sag {
  final int maxIter;

  final double tol;

  final double step_size;

  final double alpha;

  final double beta;

  final bool isSaga;

  final Double2DView x;

  final Double1DView y;

  final Double1DView weights;

  final SagState state;

  /// Coefficients or Weights
  final Double1DFix coef;

  /// Sum of gradient for each feature
  final Double1DFix sum_gradient;

  /// Previously seen gradient for each sample
  final Double1DFix gradient_memory;

  final double intercept_decay;

  bool fitIntercept;

  double intercept;

  double intercept_sum_gradient;

  int get numSamples => x.numRows;

  int get numFeatures => x.numCols;

  void learn() {
    /// Precomputation scale update (since the step size does not change in this implementation)
    final double wscale_update = 1.0 - step_size * alpha;

    bool prox = beta > 0 && isSaga;

    /*
    # Loss function to optimize
    cdef LossFunction loss
    */
    final loss; // TODO

    final featureHist = new Int1DFix.sized(numFeatures);

    // Cumulative sums needed for JIT params
    final cumulative_sums = new Double1DFix.sized(numSamples);
    // Multipliative scale needed for JIT params
    Double1DFix cumulative_sums_prox;
    if (prox) cumulative_sums_prox = new Double1DFix.sized(numSamples);

    final rnd = new Random();

    // Weights from previous iteration used to check if stop creteria is reached
    final Double1DFix previousWeights = new Double1DFix.shapedLike(coef);
    double wscale = 1.0; // Scalar used for multiplying z
    double gradient = 0.0;

    for (int n_iter = 0; n_iter < maxIter; n_iter++) {
      for (int sample_itr = 0; sample_itr < numSamples; sample_itr++) {
        // Index (row number) of the current random sample
        final int sample_ind = rnd.nextInt(numSamples);

        final Double1DView sampleX = x[sample_ind];
        final double sampleY = y[sample_ind];
        final double sampleWeight = weights[sample_ind];

        // Update the number of samples seen and the seen array
        if (state.seen[sample_ind]) {
          state.numSeen++;
          state.seen[sample_ind] = true;
        }

        // Update the coef
        if (sample_itr > 0) {
          _laggedUpdate(
              wscale,
              sample_itr,
              cumulative_sums,
              cumulative_sums_prox,
              featureHist,
              prox,
              sum_gradient,
              false,
              n_iter);
        }

        double prediction = _predictSample(sampleX, coef, wscale, intercept);

        // Compute the gradient for this sample, given the prediction
        gradient = loss.dloss(prediction, sampleY) * sampleWeight;

        // L2 regularization by simply rescaling the coef
        wscale *= wscale_update;

        // Make the updates to the sum of gradients
        for (int j = 0; j < numFeatures; j++) {
          double gradient_correction =
              sampleX[j] * (gradient - gradient_memory[sample_ind]);

          if (isSaga) {
            coef[j] -=
                (gradient_correction * step_size * (1 - (1 / state.numSeen))) /
                    wscale;
          }
          sum_gradient[j] += gradient_correction;
        }

        // Fit the intercept
        if (fitIntercept) {
          double gradient_correction = gradient - gradient_memory[sample_ind];
          intercept_sum_gradient += gradient_correction;
          gradient_correction *= step_size * (1 - (1 / state.numSeen));

          if (isSaga) {
            intercept -= (step_size *
                    intercept_sum_gradient /
                    state.numSeen *
                    intercept_decay) +
                gradient_correction;
          } else {
            intercept -= (step_size *
                intercept_sum_gradient /
                state.numSeen *
                intercept_decay);
          }

          if (intercept.isInfinite) {
            throw new Exception('Infinite error!');
          }
        }

        gradient_memory[sample_ind] = gradient;

        if (sample_itr == 0) {
          cumulative_sums[0] = step_size / (wscale * state.numSeen);
          if (prox) cumulative_sums_prox[0] = step_size * beta / wscale;
        } else {
          cumulative_sums[sample_itr] = (cumulative_sums[sample_itr - 1] +
              step_size / (wscale * state.numSeen));
          if (prox) {
            cumulative_sums_prox[sample_itr] =
                (cumulative_sums_prox[sample_itr - 1] +
                    step_size * beta / wscale);
          }
        }

        // If wscale gets too small, we need to reset the scale.
        if (wscale < 1e-9) {
          // TODO
        }
      }
      // TODO wscale

      // Check if the stopping criteria is reached
      double maxChange = 0.0;
      double maxWeight = 0.0;
      for (int idx = 0; idx < numFeatures; idx++) {
        maxWeight = max(maxWeight, coef[idx].abs());
        maxChange = max(maxChange, (coef[idx] - previousWeights[idx]).abs());
        previousWeights[idx] = coef[idx];
      }

      if ((maxWeight != 0 && ((maxChange / maxWeight) <= tol)) ||
          maxWeight == 0 && maxChange == 0) {
        // Finished!
        break;
      }
    }
  }

  /// Compute the prediction given [sample], [coef], [intercept] and [scale].
  double _predictSample(Double1DView sample, Double1DView coef, double scale,
          double intercept) =>
      (scale * sample.dot(coef)) + intercept;

  double _laggedUpdate(
      double scale,
      int sample_iter,
      Double1DFix cumulativeSums,
      Double1DFix cumulativeSumsProx,
      Int1DFix featureHist,
      bool prox,
      Double1DFix sumGradients,
      bool reset,
      int n_iter) {
    for (int feature_ind = 0; feature_ind < numFeatures; feature_ind++) {
      double cumSum = cumulativeSums[sample_iter - 1];
      double cumSumProx;
      if (prox) cumSumProx = cumulativeSumsProx[sample_iter - 1];
      if (featureHist[feature_ind] != 0) {
        cumSum -= cumulativeSums[featureHist[feature_ind] - 1];
        if (prox)
          cumSumProx -= cumulativeSumsProx[featureHist[feature_ind] - 1];
      }

      if (!prox) {
        coef[feature_ind] -= cumSum * sumGradients[feature_ind];
        if (reset) {
          coef[feature_ind] *= scale;
          if (coef[feature_ind].isInfinite) throw new Exception();
        }
      } else {
        if ((sumGradients[feature_ind] * cumSum).abs() < cumSumProx) {
          coef[feature_ind] -= cumSum * sumGradients[feature_ind];
          // TODO coef[feature_ind] =
        } else {
          int last_update_ind = featureHist[feature_ind] - 1;
          if (last_update_ind == -1) last_update_ind = sample_iter - 1;
          for (int laggedInd = sample_iter - 1;
              laggedInd >= last_update_ind - 1;
              laggedInd--) {
            double gradStep;
            double proxStep;
            if (laggedInd > 0) {
              gradStep =
                  (cumulativeSums[laggedInd] - cumulativeSums[laggedInd - 1]);
              proxStep = (cumulativeSumsProx[laggedInd] -
                  cumulativeSumsProx[laggedInd - 1]);
            } else {
              gradStep = cumulativeSums[laggedInd];
              proxStep = cumulativeSumsProx[laggedInd];
            }
            coef[feature_ind] -= sum_gradient[feature_ind] * gradStep;
            // TODO coef[feature_ind] =
          }
        }

        if (reset) {
          coef[feature_ind] *= scale;
          if (coef[feature_ind].isInfinite) throw new Exception();
        }
      }

      if (reset) {
        featureHist[feature_ind] = sample_iter % numSamples;
      } else {
        featureHist[feature_ind] = sample_iter;
      }
    }

    if (reset) {
      cumulativeSums[sample_iter - 1] = 0.0;
      if (prox) {
        cumulativeSumsProx[sample_iter - 1] = 0.0;
      }
    }
  }
}

/*
    # the index for the last time this feature was updated
    cdef np.ndarray[int, ndim=1] feature_hist_array = \
    np.zeros(n_features, dtype=np.int32, order="c")
    cdef int* feature_hist = <int*> feature_hist_array.data
*/
