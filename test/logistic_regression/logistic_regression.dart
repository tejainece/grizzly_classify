import 'package:grizzly_classify/grizzly_classify.dart';
import 'package:grizzly_series/grizzly_series.dart';
import 'package:test/test.dart';

void main() {
  group('A group of tests', () {

    setUp(() {
    });

    test('First Test', () {
      final x = new Double2D.genRows(10, (int i) => [i/10, i/10]);
      print(x);
      final Double1D w = double1D([2, 5]);
      final Double1D z = x.dot(w);
      print(z);
      // TODO 1/(1 + );
    });
  });
}
