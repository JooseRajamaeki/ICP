/*

MIT License

Copyright (c) 2017 Joose Rajamäki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


*/

#include <stdio.h>
#include <tchar.h>
#include <chrono>
#include <direct.h>


#include "ANN.h"

static inline float randu() {
	return (float)rand() / (float)RAND_MAX;
}

static void write_vector_to_file(std::string filename, std::vector<std::vector<float> >& vector_to_write) {
	std::ofstream myfile;
	myfile.open(filename);

	for (std::vector<float>& vect : vector_to_write) {

		for (int i = 0; i < (int)vect.size(); i++) {

			myfile << vect[i];

			if (i < (int)vect.size() - 1) {
				myfile << ",";
			}

		}
		myfile << std::endl;

	}

	myfile.close();
}

static void write_vector_to_file(std::string filename, std::vector<Eigen::VectorXf>& vector_to_write) {
	std::ofstream myfile;
	myfile.open(filename);

	for (Eigen::VectorXf& vect : vector_to_write) {

		for (int i = 0; i < (int)vect.size(); i++) {

			myfile << vect[i];

			if (i < (int)vect.size() - 1) {
				myfile << ",";
			}

		}
		myfile << std::endl;

	}

	myfile.close();
}


enum NoiseType
{
	UNIFORM, GAUSSIAN, MIXED, MIXED_MULTICHOISE
};

static NoiseType noise_type = NoiseType::MIXED;

Eigen::VectorXf sample_noise(int dim) {

	Eigen::VectorXf noise;
	noise.resize(dim);

	switch (noise_type)
	{
	case UNIFORM:
		noise.setRandom();
		break;
	case GAUSSIAN:
		BoxMuller<float>(noise);
		break;
	case MIXED:
		noise.setRandom();

		for (int i = 0; i < dim / 2; i++) {
			if (rand() % 2 == 0) {
				noise[i] = 1.0f;
			}
			else {
				noise[i] = -1.0f;
			}
		}

		break;
	case MIXED_MULTICHOISE:
	{
		noise.setRandom();

		const int amount_of_choises = 20;

		for (int i = 0; i < dim / 2; i++) {

			int rand_choise = rand() % amount_of_choises;
			noise[i] = (float)rand_choise;

		}
	}
	break;
	default:
		noise.setRandom();
		break;
	}

	return noise;

}


void synthetic_data_test() {

	int noise_dim = 6;
	int data_dim = 2;
	//If you want to condition on the first dimension set this to 1.
	int conditioning_dim = 0;

	float conditioned_dimension_min = 1.0f;
	float conditioned_dimension_max = 5.0f;

	//int data_amount = 1000;
	//const int minibatch_size = data_amount/2;
	//const int supervised_minibatch_size = 100;

	int data_amount = 2000;
	const int minibatch_size = 500;
	const int supervised_minibatch_size = 100;


	enum TestCase
	{
		THREE_GAUSSIAN, SERPENT, SWISS_ROLL
	};

	TestCase test_case = TestCase::THREE_GAUSSIAN;

	std::vector<Eigen::VectorXf> data;

	switch (test_case)
	{
	case THREE_GAUSSIAN:

		for (unsigned i = 0; i < data_amount; i++) {

			Eigen::VectorXf sample_datum;
			sample_datum.resize(data_dim);
			BoxMuller<float>(sample_datum);

			if (i <= data_amount / 3) {
				sample_datum[0] += 3.0f;
				sample_datum[1] += 7.5f;
			}

			if (i > data_amount / 3) {
				sample_datum[0] += 4.5f;
				sample_datum[1] *= 3.0f;
				sample_datum[1] -= 10.0f;
			}

			if (i > (data_amount * 2 / 3)) {
				BoxMuller<float>(sample_datum);
				sample_datum[0] -= 2.0f;
				sample_datum[1] *= 0.5f;
				sample_datum[1] -= 20.0f;
			}

			data.push_back(sample_datum);

		}
		break;
	case SERPENT:
		for (unsigned i = 0; i < data_amount; i++) {

			Eigen::VectorXf sample_datum;
			sample_datum.resize(data_dim);
			sample_datum.setRandom();
			sample_datum *= 2.0f*M_PI;

			float gaussian_noise = 0.0f;
			BoxMuller<float>(&gaussian_noise, 1);
			gaussian_noise *= 0.1f;

			sample_datum[1] = std::sin(sample_datum[0]) + gaussian_noise;

			data.push_back(sample_datum);

		}
		break;
	case SWISS_ROLL:
	{
		for (unsigned i = 0; i < data_amount; i++) {

			Eigen::VectorXf sample_datum;
			sample_datum.resize(data_dim);
	
			float x = randu()*8.0f+2.0f;

			sample_datum[0] = x*std::cos(x);
			sample_datum[1] = x*std::sin(x);

			Eigen::VectorXf gaussian_noise;
			gaussian_noise.resize(data_dim);
			BoxMuller<float>(gaussian_noise);
			gaussian_noise *= 0.5f;

			sample_datum += gaussian_noise;

			data.push_back(sample_datum);

		}
	}
	break;
	default:
		break;
	}





	std::vector<unsigned>  layers;
	layers.push_back(noise_dim + conditioning_dim);
	layers.push_back(50);
	layers.push_back(50);
	layers.push_back(50);
	layers.push_back(data_dim);

	std::unique_ptr<MultiLayerPerceptron> mlp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
	mlp->build_network(layers);

	float min_val = -0.5f;
	float max_val = 0.5f;

	mlp->randomize_weights(min_val, max_val);

	mlp->drop_out_stdev_ = 0.0f;
	mlp->max_gradient_norm_ = 0.1f;
	mlp->error_drop_out_prob_ = 0.0f;

	int epochs = 50000;

	std::string time_str = get_time_string();




	for (int epoch = 0; epoch < epochs; epoch++) {

		if (epoch % 50 == 0 || epoch == 5) {


			std::string filename = "true_data.csv";
			write_vector_to_file(filename, data);

			std::vector<Eigen::VectorXf> noise_inputs;
			for (unsigned i = 0; i < data_amount; i++) {

				Eigen::VectorXf sample = Eigen::VectorXf::Zero(noise_dim + conditioning_dim);
				if (conditioning_dim > 0) {
					int rand_idx = rand() % data.size();
					float conditioning_variable = data[rand_idx][0];

					while (conditioning_variable < conditioned_dimension_min || conditioning_variable > conditioned_dimension_max) {
						rand_idx = rand() % data.size();
						conditioning_variable = data[rand_idx][0];
					}

					sample.head(conditioning_dim) = data[rand_idx].head(conditioning_dim);
				}
				sample.tail(noise_dim) = sample_noise(noise_dim);

				noise_inputs.push_back(sample);
			}

			std::vector<Eigen::VectorXf > data_copy = data;

			for (int i = 0; i < data_copy.size(); i++) {
				mlp->run(noise_inputs[i].data(), data_copy[i].data());

				if (conditioning_dim > 0) {
					data_copy[i].head(conditioning_dim) = noise_inputs[i].head(conditioning_dim);
				}

			}

			filename = "predicted.csv";
			write_vector_to_file(filename, data_copy);

			if (epoch == 0) {
				filename = "initial.csv";
				write_vector_to_file(filename, data_copy);
			}

			if (epoch == 5) {
				filename = "early.csv";
				write_vector_to_file(filename, data_copy);
			}

			std::system("python scatter_plotter.py");
			std::system("python scatter_plotter_two_figures.py");
		}

		auto start = std::chrono::system_clock::now();


		{
			std::vector<Eigen::VectorXf> noise;

			for (int i = 0; i < data_amount; ++i) {
				noise.push_back(sample_noise(noise_dim));
			}

			std::vector<float*> data_ptrs = vector_to_ptrs(data);
			std::vector<float*> noise_ptrs = vector_to_ptrs(noise);

			mlp->alternating_incremental_matching((const float**)noise_ptrs.data(), (const float**)data_ptrs.data(), conditioning_dim, noise_dim, data_ptrs.size(), minibatch_size, supervised_minibatch_size);

		}



		auto end = std::chrono::system_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Time elapsed: " << (float)elapsed.count() << " ms" << std::endl;


	}



}

static const int continuous_to_discrete_data_dim = 8;

float cross_entropy_dist(const float* pt1, const float* pt2) {

	float pt1_tmp[continuous_to_discrete_data_dim];
	softmax(pt1, pt1_tmp, continuous_to_discrete_data_dim);

	float pt2_tmp[continuous_to_discrete_data_dim];
	softmax(pt2, pt2_tmp, continuous_to_discrete_data_dim);

	float dist = 0.0f;

	for (int i = 0; i < continuous_to_discrete_data_dim; ++i) {

		float val2 = pt2_tmp[i];
		if (val2 < std::numeric_limits<float>::lowest()) {
			val2 = std::numeric_limits<float>::lowest();
		}

		dist -= pt1_tmp[i] * std::logf(val2);

	}

	return dist;

}


void cross_entropy_dist_gradient(const float* prediction, const float* true_val, float* gradient) {

	float pred_tmp[continuous_to_discrete_data_dim];
	softmax(prediction, pred_tmp, continuous_to_discrete_data_dim);


	float tau = 0.0f;
	for (int i = 0; i < continuous_to_discrete_data_dim; ++i) {
		tau += true_val[i];
	}

	for (int i = 0; i < continuous_to_discrete_data_dim; ++i) {
		gradient[i] = pred_tmp[i] * tau - true_val[i];
	}


}




float cross_entropy_no_softmax(const float* pt1, const float* pt2) {

	float dist = 0.0f;

	for (int i = 0; i < continuous_to_discrete_data_dim; ++i) {

		float val2 = pt2[i];
		if (val2 < std::numeric_limits<float>::lowest()) {
			val2 = std::numeric_limits<float>::lowest();
		}

		dist -= pt1[i] * std::logf(val2);

	}

	return dist;

}


void discrete_test() {

	std::vector<Eigen::VectorXf> convergence_results;
	std::vector<Eigen::VectorXf> empirical_errors;

	int runs = 5;
	int epochs = 200;

	for (int run = 0; run < runs; ++run) {

		Eigen::VectorXf convergence_curve;
		convergence_curve.resize(epochs);


		int noise_dim = 20;
		int data_dim = continuous_to_discrete_data_dim;

		int data_amount = 1000;
		const int minibatch_size = data_amount;
		const int supervised_minibatch_size = minibatch_size;

		//noise_type = NoiseType::UNIFORM;

		enum TestCase
		{
			CATEGORICAL
		};

		TestCase test_case = TestCase::CATEGORICAL;

		std::vector<Eigen::VectorXf> data;

		float probabilities[continuous_to_discrete_data_dim];

		float prob_sum = 0.0f;
		for (float& prob : probabilities) {
			prob = randu();
			prob_sum += prob;
		}

		for (float& prob : probabilities) {
			prob /= prob_sum;
		}


		switch (test_case)
		{
		case CATEGORICAL:
		{


			for (unsigned i = 0; i < data_amount; i++) {

				Eigen::VectorXf sample_datum;
				sample_datum.resize(data_dim);
				sample_datum.setZero();

				float u = randu();
				float accumulator = 0.0f;

				int j = 0;
				for (const float& prob : probabilities) {

					accumulator += prob;

					if (u <= accumulator) {
						break;
					}

					++j;
				}

				sample_datum[j] = 1.0f;

				data.push_back(sample_datum);

			}
		}
		break;
		default:
			break;
		}


		auto pmf = [&data_dim](const std::vector<Eigen::VectorXf>& hist_data) {

			Eigen::VectorXf hist;
			hist.resize(data_dim);
			hist.setZero();


			for (const Eigen::VectorXf& datum : hist_data) {

				int max_idx = 0;

				for (int j = 0; j < datum.size(); ++j) {
					if (datum[j] > datum[max_idx]) {
						max_idx = j;
					}
				}

				hist[max_idx] += 1.0f;

			}

			float sum = hist.cwiseAbs().sum();
			hist /= sum;

			return hist;

		};



		Eigen::VectorXf true_hist = Eigen::Map<Eigen::VectorXf>(probabilities, continuous_to_discrete_data_dim);
		Eigen::VectorXf empirical_hist = pmf(data);

		const float empirical_error = (empirical_hist - true_hist).cwiseAbs().sum();


		std::vector<unsigned>  layers;
		layers.push_back(noise_dim);
		layers.push_back(50);
		layers.push_back(50);
		layers.push_back(50);
		layers.push_back(data_dim);

		std::unique_ptr<MultiLayerPerceptron> mlp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
		mlp->build_network(layers);

		float min_val = -0.5f;
		float max_val = 0.5f;

		mlp->randomize_weights(min_val, max_val);


		mlp->drop_out_stdev_ = 0.0f;
		mlp->max_gradient_norm_ = 0.1f;
		mlp->error_drop_out_prob_ = 0.0f;

		std::string time_str = get_time_string();




		for (int epoch = 0; epoch < epochs; epoch++) {

			float error = 0.0f;
			{

				Eigen::VectorXf noise;
				std::vector<Eigen::VectorXf> predicted;

				const int predicted_data_amount = 10 * data_amount;

				for (int i = 0; i < predicted_data_amount; ++i) {
					noise = sample_noise(noise_dim);

					Eigen::VectorXf pred;
					pred.resize(data_dim);

					mlp->run(noise.data(), pred.data());

					predicted.push_back(pred);

				}

				Eigen::VectorXf predicted_hist = pmf(predicted);



				if (runs == 1 && epoch % 10 == 0) {

					std::vector<Eigen::VectorXf> true_data;
					true_data.push_back(empirical_hist);

					std::string filename = "empirical_hist.csv";
					write_vector_to_file(filename, true_data);

					predicted.clear();
					predicted.push_back(predicted_hist);

					filename = "predicted_hist.csv";
					write_vector_to_file(filename, predicted);

					predicted.clear();
					predicted.push_back(true_hist);

					filename = "true_hist.csv";
					write_vector_to_file(filename, predicted);

					std::system("python hist_plotter.py");

				}


				error = (empirical_hist - predicted_hist).cwiseAbs().sum();

				convergence_curve[epoch] = error;


			}


			auto start = std::chrono::system_clock::now();


			{
				std::vector<Eigen::VectorXf> noise;

				for (int i = 0; i < data_amount; ++i) {
					noise.push_back(sample_noise(noise_dim));
				}

				std::vector<float*> data_ptrs = vector_to_ptrs(data);
				std::vector<float*> noise_ptrs = vector_to_ptrs(noise);

				const int conditioning_dim = 0;
				mlp->incremental_matching((const float**)noise_ptrs.data(), (const float**)data_ptrs.data(), conditioning_dim, noise_dim, data_ptrs.size(), minibatch_size, supervised_minibatch_size, cross_entropy_dist, cross_entropy_dist_gradient);

			}



			auto end = std::chrono::system_clock::now();

			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			std::cout << "Epoch: " << epoch << " Time elapsed: " << (float)elapsed.count() << " ms. Error: " << error << " Empirical error: " << empirical_error << std::endl;;


		}

		convergence_results.push_back(convergence_curve);

		empirical_errors.push_back(Eigen::VectorXf::Ones(1)*empirical_error);

	}

	std::string filename = "continuous_to_discrete_convergence.csv";
	write_vector_to_file(filename, convergence_results);

	filename = "empirical_errors.csv";
	write_vector_to_file(filename, empirical_errors);


}



int ReverseInt(int i)
{
	// author : Eric Yuan 
	// my blog: http://eric-yuan.me/
	// part of this code is stolen from http://compvisionlab.wordpress.com/
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(std::string filename, std::vector<std::vector<double> > &vec, int num_images)
{
	// author : Eric Yuan 
	// my blog: http://eric-yuan.me/
	// part of this code is stolen from http://compvisionlab.wordpress.com/
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		num_images = std::min(num_images, number_of_images);

		for (int i = 0; i < num_images; ++i)
		{
			std::vector<double> tp;
			for (int r = 0; r < n_rows; ++r)
			{
				for (int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.push_back((double)temp);
				}
			}
			vec.push_back(tp);
		}
	}
}


void read_Mnist_Label(std::string filename, std::vector<double> &vec, int& num_images)
{
	// author : Eric Yuan 
	// my blog: http://eric-yuan.me/
	// part of this code is stolen from http://compvisionlab.wordpress.com/
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		num_images = std::min(number_of_images, num_images);

		vec.resize(num_images);
		for (int i = 0; i < num_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (double)temp;
		}
	}
}


void mnist_test() {

	std::vector<std::vector<float>> data;
	int number_of_images = 60000;
	int minibatch_size = 10000;
	int supervised_minibatch_size = 100;

	const bool debug = false;
	if (debug) {
		number_of_images = 1000;
		minibatch_size = number_of_images;
		supervised_minibatch_size = number_of_images;
	}

	int conditioning_dim = 0;

	{
		std::string filename = "mnist/train-images-idx3-ubyte";
		int image_size = 28 * 28;

		//read MNIST iamge into double vector
		std::vector<std::vector<double> > vec;
		read_Mnist(filename, vec, number_of_images);
		std::cout << vec.size() << std::endl;
		std::cout << vec[0].size() << std::endl;

		float min_val = 100000.0f;
		float max_val = -100000.0f;

		for (int i = 0; i < number_of_images; ++i) {

			if (data.size() <= i) {
				std::vector<float> sample;
				data.push_back(sample);
			}

			for (const double& value : vec[i]) {
				data[i].push_back((float)value / (float)255);

				min_val = std::min(min_val, data[i].back());
				max_val = std::max(max_val, data[i].back());

			}
		}

		std::cout << min_val << std::endl;
		std::cout << max_val << std::endl;

	}




	int noise_dim = 20;
	int data_dim = data[0].size();


	std::vector<unsigned>  layers;
	layers.push_back(noise_dim + conditioning_dim);
	layers.push_back(300);
	layers.push_back(300);
	layers.push_back(300);
	layers.push_back(data_dim);

	std::unique_ptr<MultiLayerPerceptron> mlp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
	mlp->build_network(layers);

	float min_val = -0.1f;
	float max_val = 0.1f;

	mlp->randomize_weights(min_val, max_val);

	mlp->drop_out_stdev_ = 0.0f;
	mlp->max_gradient_norm_ = 0.1f;
	mlp->error_drop_out_prob_ = 0.0f;

	int epochs = 300;

	std::string time_str = get_time_string();



	for (int epoch = 0; epoch < epochs; epoch++) {

		//Saving some predictions to file
		{

			const int plot_samples = 10;

			std::vector<Eigen::VectorXf> noise_inputs;

			for (unsigned i = 0; i < plot_samples; i++) {

				Eigen::VectorXf sample = Eigen::VectorXf::Zero(noise_dim + conditioning_dim);
				sample.tail(noise_dim) = sample_noise(noise_dim);
				noise_inputs.push_back(sample);

			}

			std::vector<std::vector<float>> data_copy;

			for (int i = 0; i < plot_samples; i++) {
				std::vector<float> prediction = std::vector<float>(data_dim, 0.0f);
				mlp->run(noise_inputs[i].data(), prediction.data());
				data_copy.push_back(prediction);
			}

			std::string plotting_time_string = get_time_string();

			std::string filename = "number_figures/";
			filename += plotting_time_string;
			filename += "/";

			_mkdir(filename.data());

			filename += "images.csv";

			write_vector_to_file(filename, data_copy);

		}

		auto start = std::chrono::system_clock::now();


		{
			std::vector<Eigen::VectorXf> noise;

			for (int i = 0; i < number_of_images; ++i) {
				noise.push_back(sample_noise(noise_dim));
			}

			std::vector<float*> data_ptrs = vector_to_ptrs(data);
			std::vector<float*> noise_ptrs = vector_to_ptrs(noise);

			mlp->incremental_matching((const float**)noise_ptrs.data(), (const float**)data_ptrs.data(), conditioning_dim, noise_dim, data_ptrs.size(), minibatch_size, supervised_minibatch_size);

		}



		auto end = std::chrono::system_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Time elapsed: " << (float)elapsed.count() << " ms" << std::endl;

	}
}



void mnist_conditioned_test() {

	//An epoch might take a while (a few minutes), so be patient.

	std::vector<std::vector<float>> data;
	int number_of_images = 60000;
	int minibatch_size = 10000;
	int supervised_minibatch_size = 100;


	int conditioning_dim = 1;

	const bool debug = false;
	if (debug) {
		number_of_images = 100;
		minibatch_size = 100;
		supervised_minibatch_size = 10;
	}

	if (conditioning_dim > 0)
	{
		std::string filename = "mnist/train-labels-idx1-ubyte";

		std::vector<double> vec(number_of_images);
		read_Mnist_Label(filename, vec, number_of_images);
		std::cout << vec.size() << std::endl;

		for (const double& conditioning : vec) {
			std::vector<float> sample;
			sample.push_back((float)conditioning);
			data.push_back(sample);

			if (data.size() >= number_of_images) {
				break;
			}

		}

	}

	{
		std::string filename = "mnist/train-images-idx3-ubyte";
		int image_size = 28 * 28;

		//read MNIST iamge into double vector
		std::vector<std::vector<double> > vec;
		read_Mnist(filename, vec, number_of_images);
		std::cout << vec.size() << std::endl;
		std::cout << vec[0].size() << std::endl;

		float min_val = 10000.0f;
		float max_val = -100000.0f;

		for (int i = 0; i < number_of_images; ++i) {

			if (data.size() <= i) {
				std::vector<float> sample;
				data.push_back(sample);
			}

			for (const double& value : vec[i]) {
				data[i].push_back((float)value / (float)255);

				min_val = std::min(min_val, data[i].back());
				max_val = std::max(max_val, data[i].back());

			}
		}

		std::cout << min_val << std::endl;
		std::cout << max_val << std::endl;

	}



	int noise_dim = 20;
	int data_dim = data[0].size();

	std::vector<unsigned>  layers;
	layers.push_back(noise_dim + conditioning_dim);
	layers.push_back(300);
	layers.push_back(300);
	layers.push_back(300);
	layers.push_back(data_dim);

	std::unique_ptr<MultiLayerPerceptron> mlp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
	mlp->build_network(layers);

	float min_val = -0.1f;
	float max_val = 0.1f;

	mlp->randomize_weights(min_val, max_val);

	mlp->drop_out_stdev_ = 0.0f;
	mlp->max_gradient_norm_ = 0.1f;
	mlp->error_drop_out_prob_ = 0.0f;

	int epochs = 300;

	//Uncomment the following line to load a pretrained model
	//mlp->load_from_file("mnist_network_conditioned_trained.nn");

	std::string time_str = get_time_string();


	for (int epoch = 0; epoch < epochs; epoch++) {

		if (true || epoch % 5 == 0) {

			std::string plotting_time_string = get_time_string();

			//Plotting ones
			for (int number = 0; number <= 9; number++)
			{
				const int plot_samples = 10;

				std::vector<Eigen::VectorXf> noise_inputs;

				for (unsigned i = 0; i < plot_samples; i++) {

					Eigen::VectorXf sample = Eigen::VectorXf::Zero(noise_dim + conditioning_dim);
					if (conditioning_dim > 0) {
						float label = (float)number;
						sample[0] = label;
					}

					sample.tail(noise_dim) = sample_noise(noise_dim);
					noise_inputs.push_back(sample);
				}

				std::vector<std::vector<float>> data_copy;

				for (int i = 0; i < plot_samples; i++) {
					std::vector<float> prediction = std::vector<float>(data_dim, 0.0f);
					mlp->run(noise_inputs[i].data(), prediction.data());

					if (conditioning_dim > 0) {
						prediction[0] = number;
					}

					data_copy.push_back(prediction);

				}

				std::string filename = "number_figures/";
				filename += plotting_time_string;
				filename += "/";

				_mkdir(filename.data());



				switch (number)
				{
				case 0:
					filename += "zeros.csv";
					break;
				case 1:
					filename += "ones.csv";
					break;
				case 2:
					filename += "twos.csv";
					break;
				case 3:
					filename += "threes.csv";
					break;
				case 4:
					filename += "fours.csv";
					break;
				case 5:
					filename += "fives.csv";
					break;
				case 6:
					filename += "sixes.csv";
					break;
				case 7:
					filename += "sevens.csv";
					break;
				case 8:
					filename += "eights.csv";
					break;
				case 9:
					filename += "nines.csv";
					break;
				default:
					break;
				}

				write_vector_to_file(filename, data_copy);
			}

			//std::system("python mnist_plotter_conditioned.py");

			std::string network_name = "networks/" + plotting_time_string;
			network_name += "_network.nn";
			mlp->write_to_file(network_name);

		}

		auto start = std::chrono::system_clock::now();


		{
			std::vector<Eigen::VectorXf> noise;

			for (int i = 0; i < number_of_images; ++i) {
				noise.push_back(sample_noise(noise_dim));
			}

			std::vector<float*> data_ptrs = vector_to_ptrs(data);
			std::vector<float*> noise_ptrs = vector_to_ptrs(noise);

			mlp->incremental_matching((const float**)noise_ptrs.data(), (const float**)data_ptrs.data(), conditioning_dim, noise_dim, data_ptrs.size(), minibatch_size, supervised_minibatch_size);
		}



		auto end = std::chrono::system_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Time elapsed: " << (float)elapsed.count() << " ms" << std::endl;

	}
}




void teaser_test() {

	int noise_dim = 2;
	int data_dim = 2;
	//If you want to condition on the first dimension set this to 1.
	int conditioning_dim = 0;

	float conditioned_dimension_min = 1.0f;
	float conditioned_dimension_max = 5.0f;

	int data_amount = 3000;
	const int minibatch_size = data_amount;
	const int supervised_minibatch_size = 100;

	noise_type = NoiseType::UNIFORM;

	enum TestCase
	{
		THREE_GAUSSIAN, SERPENT, SWISS_ROLL
	};

	TestCase test_case = TestCase::SWISS_ROLL;

	std::vector<Eigen::VectorXf> data;

	switch (test_case)
	{
	case THREE_GAUSSIAN:

		for (unsigned i = 0; i < data_amount; i++) {

			Eigen::VectorXf sample_datum;
			sample_datum.resize(data_dim);
			BoxMuller<float>(sample_datum);

			if (i <= data_amount / 3) {
				sample_datum[0] += 3.0f;
				sample_datum[1] += 7.5f;
			}

			if (i > data_amount / 3) {
				sample_datum[0] += 4.5f;
				sample_datum[1] *= 3.0f;
				sample_datum[1] -= 10.0f;
			}

			if (i > (data_amount * 2 / 3)) {
				BoxMuller<float>(sample_datum);
				sample_datum[0] -= 2.0f;
				sample_datum[1] *= 0.5f;
				sample_datum[1] -= 20.0f;
			}

			data.push_back(sample_datum);

		}
		break;
	case SERPENT:
		for (unsigned i = 0; i < data_amount; i++) {

			Eigen::VectorXf sample_datum;
			sample_datum.resize(data_dim);
			sample_datum.setRandom();
			sample_datum *= 2.0f*M_PI;

			float gaussian_noise = 0.0f;
			BoxMuller<float>(&gaussian_noise, 1);
			gaussian_noise *= 0.1f;

			sample_datum[1] = std::sin(sample_datum[0]) + gaussian_noise;

			data.push_back(sample_datum);

		}
		break;
	case SWISS_ROLL:
	{
		for (unsigned i = 0; i < data_amount; i++) {

			Eigen::VectorXf sample_datum;
			sample_datum.resize(data_dim);

			float x = randu()*8.0f + 2.0f;

			sample_datum[0] = x*std::cos(x);
			sample_datum[1] = x*std::sin(x);

			Eigen::VectorXf gaussian_noise;
			gaussian_noise.resize(data_dim);
			BoxMuller<float>(gaussian_noise);
			gaussian_noise *= 0.5f;

			sample_datum += gaussian_noise;

			data.push_back(sample_datum);

		}
	}
	break;
	default:
		break;
	}





	std::vector<unsigned>  layers;
	layers.push_back(noise_dim + conditioning_dim);
	layers.push_back(50);
	layers.push_back(50);
	layers.push_back(50);
	layers.push_back(data_dim);

	std::unique_ptr<MultiLayerPerceptron> mlp = std::unique_ptr<MultiLayerPerceptron>(new MultiLayerPerceptron());
	mlp->build_network(layers);

	float min_val = -0.5f;
	float max_val = 0.5f;

	mlp->randomize_weights(min_val, max_val);

	mlp->drop_out_stdev_ = 0.0f;
	mlp->max_gradient_norm_ = 0.1f;
	mlp->error_drop_out_prob_ = 0.0f;

	int epochs = 50000;

	std::string time_str = get_time_string();




	for (int epoch = 0; epoch < epochs; epoch++) {

		if (epoch % 200 == 0 && epoch > 0) {

			std::vector<Eigen::VectorXf> noise_inputs;
			for (unsigned i = 0; i < data_amount; i++) {

				Eigen::VectorXf sample = Eigen::VectorXf::Zero(noise_dim + conditioning_dim);
				if (conditioning_dim > 0) {
					int rand_idx = rand() % data.size();
					float conditioning_variable = data[rand_idx][0];

					while (conditioning_variable < conditioned_dimension_min || conditioning_variable > conditioned_dimension_max) {
						rand_idx = rand() % data.size();
						conditioning_variable = data[rand_idx][0];
					}

					sample.head(conditioning_dim) = data[rand_idx].head(conditioning_dim);
				}
				sample.tail(noise_dim) = sample_noise(noise_dim);

				noise_inputs.push_back(sample);
			}

			std::vector<Eigen::VectorXf > data_copy = data;

			for (int i = 0; i < data_copy.size(); i++) {
				mlp->run(noise_inputs[i].data(), data_copy[i].data());

				if (conditioning_dim > 0) {
					data_copy[i].head(conditioning_dim) = noise_inputs[i].head(conditioning_dim);
				}

			}


			std::string  filename = "noise_inputs.csv";
			write_vector_to_file(filename, noise_inputs);

			filename = "predicted.csv";
			write_vector_to_file(filename, data_copy);


			std::system("python teaser_plotter.py");
		}

		auto start = std::chrono::system_clock::now();


		{
			std::vector<Eigen::VectorXf> noise;

			for (int i = 0; i < data_amount; ++i) {
				noise.push_back(sample_noise(noise_dim));
			}

			std::vector<float*> data_ptrs = vector_to_ptrs(data);
			std::vector<float*> noise_ptrs = vector_to_ptrs(noise);

			mlp->incremental_matching((const float**)noise_ptrs.data(), (const float**)data_ptrs.data(), conditioning_dim, noise_dim, data_ptrs.size(), minibatch_size, supervised_minibatch_size);

		}



		auto end = std::chrono::system_clock::now();

		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "Time elapsed: " << (float)elapsed.count() << " ms" << std::endl;


	}



}


int main()
{

	synthetic_data_test();
	//mnist_test();
	//mnist_conditioned_test();
	//discrete_test();
	//teaser_test();

	return 0;
}

