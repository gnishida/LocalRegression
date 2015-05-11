#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "MLUtils.h"
#include "LocalLinearRegression.h"

using namespace std;

int main() {
	cv::Mat_<double> X, Y;
	ml::loadDataset("samplesX.txt", X);
	ml::loadDataset("samplesY.txt", Y);

	LocalLinearRegression llr(X, Y, 250);

	for (double x = 1700; x < 3800; x+=100) {
		cv::Mat_<double> m = (cv::Mat_<double>(1, 1) << x);
		cv::Mat_<double> y = llr.predict(m);
		cout << y << endl;
	}

	return 0;
}
